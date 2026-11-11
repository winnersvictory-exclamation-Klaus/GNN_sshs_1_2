"""Shared library for the wave-update test battery (DESIGN.md section 1).

Import this module FIRST in every test script (it caps BLAS threads before heavy numpy work):

    import sys; sys.path.insert(0, "<repo>/battery")
    import common as C          # C.wt = wave_test, C.rx = rank_experiment

Conventions
-----------
* trajectories X have shape (steps+1, R, n); X[0] = u0.  `wt.simulate` zeroes a realisation from the
  step it fails: if fail_step = s >= 0 then X[:s] is valid and X[s:] == 0.  Every metric below masks
  failed realisations using fail_step (see the table in COMMON_API.md for what is NaN after failure).
* graphs are `rx.G` objects.  Treat them as immutable: `make_graph` caches them per process.
* the mean force is F(u) = A(u) - u = -L_rw u with L_rw = I - D^-1 A (D = weighted degree).  L_rw is NOT
  symmetric: eigenvectors / energies / Jacobian symmetry are all done in the D-weighted inner product
  <a,b>_D = a^T D b.
"""
from __future__ import annotations

import contextlib
import csv
import itertools
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

try:  # the env vars above are too late if numpy was imported first; this works at runtime
    from threadpoolctl import threadpool_limits as _tpl
    _tpl(limits=1)
except Exception:  # pragma: no cover
    pass

BATTERY_DIR = Path(__file__).resolve().parent
ROOT = BATTERY_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import wave_test as wt  # noqa: E402

rx = wt.rx
from scipy import stats as _st  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.metrics import adjusted_rand_score, roc_auc_score  # noqa: E402

# --------------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------------
FAIL_AMP = wt.FAIL_AMP                  # 20.0 (rx.FAIL_AMP).  Override per call with fail_amp=...
RMS_STD = 1.0 / math.sqrt(3.0)          # RMS of U(-1,1): all random initial conditions are matched to it
DEV_SEEDS = tuple(range(0, 8))
HELD_SEEDS = tuple(range(100, 116))
HOLD_PARAMS = dict(eps=0.3, delta=0.1, rho=2.0, W=50, stride=25, floor=1e-3)
MAX_PROCS = int(os.environ.get("BATTERY_MAX_PROCS", "2"))
PARTITION_NAMES = ("edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode")


# --------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------
def _rel(a, b, tiny=1e-12):
    return float(a / b) if b > tiny else float("nan")


@contextlib.contextmanager
def fail_amp_override(value):
    """Temporarily set wt.FAIL_AMP (wt.simulate reads the module global at call time)."""
    if value is None:
        yield
        return
    old = wt.FAIL_AMP
    wt.FAIL_AMP = float(value)
    try:
        yield
    finally:
        wt.FAIL_AMP = old


def resolve_params(table, name, over=None):
    """Full parameter dict for a registered force/scheme: registry defaults updated by `over`.
    `table` is wt.FORCES or wt.SCHEMES.  Unknown keys raise (typo guard)."""
    if name not in table:
        kind = "force" if table is wt.FORCES else "scheme"
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)[:40]}...")
    p = dict(table[name][1])
    if over:
        bad = [k for k in over if k not in p]
        if bad:
            raise KeyError(f"{name}: unknown parameter(s) {bad}; valid: {sorted(p)}")
        p.update(over)
    return p


def valid_len(fail_step, T1, R=None):
    """Number of valid time indices per realisation: fail_step if it failed, else T1 (= steps+1)."""
    if fail_step is None:
        return np.full(R, T1)
    fs = np.asarray(fail_step)
    return np.where(fs >= 0, fs, T1)


def _ok(fail_step, R):
    return np.ones(R, bool) if fail_step is None else (np.asarray(fail_step) < 0)


def _rms_rows(a):
    return np.sqrt((a.reshape(a.shape[0], -1) ** 2).mean(1))


# --------------------------------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------------------------------
def wdeg(g):
    """Weighted degree vector d_i = sum_j w_ij (the D of L_rw = I - D^-1 A)."""
    return g.WT.sum(1)


def L_rw(g):
    """Dense row-normalised Laplacian I - D^-1 A (not symmetric)."""
    return np.eye(g.n) - g.A / wdeg(g)[:, None]


def is_connected(g):
    return connected_components(csr_matrix(g.A > 0), directed=False)[0] == 1


def spectrum(g):
    """(lam, Phi, d): eigenvalues (ascending, lam[0] ~ 0) and D-orthonormal eigenvectors (columns) of L_rw.
    Phi^T diag(d) Phi = I and L_rw Phi = Phi diag(lam).  Cached on the graph object."""
    sp = getattr(g, "_bat_spec", None)
    if sp is None:
        d = wdeg(g)
        S = g.A / np.sqrt(np.outer(d, d))
        lam, V = np.linalg.eigh(np.eye(g.n) - S)
        sp = (lam, V / np.sqrt(d)[:, None], d)
        g._bat_spec = sp
    return sp


# --------------------------------------------------------------------------------------------------
# graph zoo
# --------------------------------------------------------------------------------------------------
_DEFAULTS = {
    "two_block": dict(block="sbm", sizes=(24, 24), bridges=2, wb=1.0, seed=0),
    "barbell_path": dict(k=16, L=4, seed=0),
    "grid_neck": dict(a=5, w=2, wb=1.0, seed=0),
    "ring_of_cliques": dict(q=4, s=12, seed=0),
    "sbm_q": dict(q=3, size=16, p_in=0.5, p_out=0.02, seed=0),
    "binary_tree": dict(depth=5, seed=0),
    "lollipop": dict(k=24, L=24, seed=0),
    "watts_strogatz": dict(n=48, k=4, p=0.1, seed=0),
    "random_regular": dict(n=48, d=4, seed=0),
    "er": dict(n=None, sizes=(24, 24), deg=6.0, seed=0),
    "shuffled": dict(sizes=(24, 24), seed=0),
    "sbm_het": dict(sizes=(24, 24), bridges=2, wb=1.0, sigma=0.6, p_in=0.3, seed=0),
}
_LEGACY = ("cycle", "clique", "sbm")          # rx kinds -> two_block
_FREE_KEYS = ("role", "tag", "id")


def normalize_spec(spec):
    """Full spec dict with defaults filled in.  rx-style kinds 'cycle'/'clique'/'sbm' are mapped to
    kind='two_block', block=<kind>.  Unknown keys raise."""
    sp = dict(spec)
    if sp.get("kind") in _LEGACY:
        sp["block"] = sp["kind"]
        sp["kind"] = "two_block"
    kind = sp.get("kind")
    if kind not in _DEFAULTS:
        raise KeyError(f"unknown graph kind {kind!r}; known: {sorted(_DEFAULTS)}")
    out = dict(_DEFAULTS[kind])
    extra = {k: v for k, v in sp.items() if k in _FREE_KEYS}
    bad = [k for k in sp if k != "kind" and k not in out and k not in _FREE_KEYS]
    if bad:
        raise KeyError(f"{kind}: unknown spec key(s) {bad}; valid: {sorted(out)}")
    out.update({k: v for k, v in sp.items() if k not in _FREE_KEYS})
    if "sizes" in out:
        out["sizes"] = tuple(int(x) for x in out["sizes"])
    out["kind"] = kind
    out.update(extra)
    return out


def spec_id(spec):
    """Short readable id, e.g. 'two_block:cycle_24x24_b2_wb1_s0'."""
    sp = normalize_spec(spec)
    kind = sp["kind"]
    parts = []
    for k, v in sp.items():
        if k in ("kind", "seed", "role", "tag", "id", "block") or v is None:
            continue
        if k == "sizes":
            parts.append("x".join(map(str, v)))
        else:
            parts.append(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}")
    head = f"{kind}:{sp['block']}" if kind == "two_block" else kind
    return head + "_" + "_".join(parts) + f"_s{sp['seed']}"


def _spec_key(sp):
    return repr(sorted((k, v) for k, v in sp.items() if k not in _FREE_KEYS))


def _edges_connected(n, edges):
    if not edges:
        return n == 1
    r = np.array([e[0] for e in edges])
    c = np.array([e[1] for e in edges])
    m = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return connected_components(m, directed=False)[0] == 1


def _clique(nodes, w=1.0):
    return [(int(i), int(j), w) for i, j in itertools.combinations(nodes, 2)]


def _b_two_block(sp):
    n0, n1 = sp["sizes"]
    if sp["bridges"] > min(n0, n1):
        raise ValueError(f"bridges={sp['bridges']} > smaller block size {min(n0, n1)}")
    n, e, lab = rx.two_block(sp["block"], sp["sizes"], int(sp["bridges"]), sp["wb"], sp["seed"])
    return n, e, lab


def _b_barbell_path(sp):
    k, L = int(sp["k"]), int(sp["L"])
    n = 2 * k + L
    A, P, B = list(range(k)), list(range(k, k + L)), list(range(k + L, 2 * k + L))
    e = _clique(A) + _clique(B)
    chain = [A[-1]] + P + [B[0]]
    e += [(chain[i], chain[i + 1], 1.0) for i in range(len(chain) - 1)]
    lab = np.zeros(n, int)
    lab[B] = 1
    lab[P[(L + 1) // 2:]] = 1
    return n, e, lab


def _b_grid_neck(sp):
    a, w = int(sp["a"]), int(sp["w"])
    if not (1 <= w <= a):
        raise ValueError("need 1 <= w <= a")
    n = 2 * a * a
    e = []
    for off in (0, a * a):
        for r in range(a):
            for c in range(a):
                i = off + r * a + c
                if c + 1 < a:
                    e.append((i, i + 1, 1.0))
                if r + 1 < a:
                    e.append((i, i + a, 1.0))
    r0 = (a - w) // 2
    for r in range(r0, r0 + w):
        e.append((r * a + a - 1, a * a + r * a, float(sp["wb"])))
    lab = np.array([0] * (a * a) + [1] * (a * a))
    return n, e, lab


def _b_ring_of_cliques(sp):
    q, s = int(sp["q"]), int(sp["s"])
    if q < 3 or s < 2:
        raise ValueError("ring_of_cliques needs q>=3 and s>=2")
    e, lab = [], np.repeat(np.arange(q), s)
    for i in range(q):
        e += _clique(range(i * s, (i + 1) * s))
        e.append((i * s + s - 1, ((i + 1) % q) * s, 1.0))
    return q * s, e, lab


def _b_sbm_q(sp):
    q, s = int(sp["q"]), int(sp["size"])
    n = q * s
    rng = np.random.default_rng(sp["seed"])
    lab = np.repeat(np.arange(q), s)
    iu, ju = np.triu_indices(n, 1)
    P = np.where(lab[iu] == lab[ju], sp["p_in"], sp["p_out"])
    for _ in range(500):
        keep = rng.random(len(iu)) < P
        e = [(int(i), int(j), 1.0) for i, j in zip(iu[keep], ju[keep])]
        if e and _edges_connected(n, e):
            return n, e, lab
    raise RuntimeError("sbm_q: could not draw a connected graph")


def _b_binary_tree(sp):
    d = int(sp["depth"])
    if d < 2:
        raise ValueError("depth >= 2")
    n = 2 ** (d + 1) - 1
    e = [(i, c, 1.0) for i in range(n) for c in (2 * i + 1, 2 * i + 2) if c < n]
    lab = np.zeros(n, int)
    for i in range(1, n):
        a = i
        while (a - 1) // 2 != 0:
            a = (a - 1) // 2
        lab[i] = 0 if a == 1 else 1          # root (node 0) belongs to the left subtree; cut edge = (0, 2)
    return n, e, lab


def _b_lollipop(sp):
    k, L = int(sp["k"]), int(sp["L"])
    if L < 1:
        raise ValueError("L >= 1")
    e = _clique(range(k))
    chain = [k - 1] + list(range(k, k + L))
    e += [(chain[i], chain[i + 1], 1.0) for i in range(L)]
    lab = np.array([0] * k + [1] * L)        # clique vs tail; single bridge edge (k-1, k)
    return k + L, e, lab


def _b_watts_strogatz(sp):
    n, k, p = int(sp["n"]), int(sp["k"]), float(sp["p"])
    rng = np.random.default_rng(sp["seed"])
    base = [(i, (i + j) % n) for i in range(n) for j in range(1, k // 2 + 1)]
    key = lambda a, b: (min(a, b), max(a, b))  # noqa: E731
    for _ in range(200):
        S = {key(a, b) for a, b in base}
        for a, b in base:
            if rng.random() < p:
                for _try in range(50):
                    w = int(rng.integers(n))
                    if w != a and key(a, w) not in S:
                        S.discard(key(a, b)); S.add(key(a, w))
                        break
        e = [(i, j, 1.0) for i, j in sorted(S)]
        if _edges_connected(n, e):
            return n, e, None
    raise RuntimeError("watts_strogatz: could not draw a connected graph")


def _b_random_regular(sp):
    n, d = int(sp["n"]), int(sp["d"])
    if d >= n or (n * d) % 2:
        raise ValueError("need d < n and n*d even")
    S = set()
    for i in range(n):
        for j in range(1, d // 2 + 1):
            a, b = i, (i + j) % n
            S.add((min(a, b), max(a, b)))
    if d % 2:
        for i in range(n // 2):
            S.add((i, i + n // 2))
    base = [(i, j, 1.0) for i, j in sorted(S)]
    for att in range(50):
        _, e = rx.shuffled(n, base, sp["seed"] + 1000 * att)
        if _edges_connected(n, e):
            return n, e, None
    raise RuntimeError("random_regular: not connected")


def _b_er(sp):
    n = int(sp["n"]) if sp["n"] else int(sum(sp["sizes"]))
    n, e = rx.er_graph(n, sp["deg"], sp["seed"])
    return n, e, None


def _b_shuffled(sp):
    n0, e0, _ = rx.two_block("sbm", sp["sizes"], 4, 1.0, sp["seed"])
    n, e = rx.shuffled(n0, e0, sp["seed"] + 7)
    return n, e, None


def _dc_block(n, sigma, p, rng):
    iu, ju = np.triu_indices(n, 1)
    for _ in range(500):
        th = rng.lognormal(0.0, sigma, n)
        th = th / th.mean()
        keep = rng.random(len(iu)) < np.minimum(1.0, p * th[iu] * th[ju])
        e = [(int(i), int(j)) for i, j in zip(iu[keep], ju[keep])]
        if e and _edges_connected(n, e):
            return e
    raise RuntimeError("sbm_het block not connected")


def _b_sbm_het(sp):
    rng = np.random.default_rng(sp["seed"])
    n0, n1 = sp["sizes"]
    b = int(sp["bridges"])
    e = [(i, j, 1.0) for i, j in _dc_block(n0, sp["sigma"], sp["p_in"], rng)]
    e += [(n0 + i, n0 + j, 1.0) for i, j in _dc_block(n1, sp["sigma"], sp["p_in"], rng)]
    if b:
        a = rng.choice(n0, b, replace=False)
        c = n0 + rng.choice(n1, b, replace=False)
        e += [(int(i), int(j), float(sp["wb"])) for i, j in zip(a, c)]
    return n0 + n1, e, np.array([0] * n0 + [1] * n1)


_BUILDERS = {"two_block": _b_two_block, "barbell_path": _b_barbell_path, "grid_neck": _b_grid_neck,
             "ring_of_cliques": _b_ring_of_cliques, "sbm_q": _b_sbm_q, "binary_tree": _b_binary_tree,
             "lollipop": _b_lollipop, "watts_strogatz": _b_watts_strogatz, "random_regular": _b_random_regular,
             "er": _b_er, "shuffled": _b_shuffled, "sbm_het": _b_sbm_het}
_GCACHE: dict = {}


def bridge_mask(g, truth):
    """Boolean (m,) mask of edges whose endpoints have different true labels (None if truth is None)."""
    if truth is None:
        return None
    t = np.asarray(truth)
    return t[g.ei] != t[g.ej]


def make_graph(spec, cache=True):
    """spec dict -> (g: rx.G, truth: int array (n,) | None, meta: dict).

    spec['kind'] in: two_block (block=cycle|clique|sbm, sizes, bridges, wb), barbell_path (k, L),
    grid_neck (a, w, wb), ring_of_cliques (q, s), sbm_q (q, size, p_in, p_out), binary_tree (depth),
    lollipop (k, L), watts_strogatz (n, k, p), random_regular (n, d), er (n or sizes, deg), shuffled (sizes),
    sbm_het (sizes, bridges, wb, sigma, p_in).  rx-style kinds 'cycle'/'clique'/'sbm' are accepted and mean
    two_block with that block.  `seed` is always accepted.  Same spec -> same graph (cached per process;
    do not mutate the returned objects).

    meta keys: id, kind, n, m, seed, params (normalised spec), q (#true blocks or None), connected,
    bridge_mask (bool (m,) over g.ei/g.ej or None), bridge_idx (int array or None), n_bridges, dmax, mean_deg.
    """
    sp = normalize_spec(spec)
    key = _spec_key(sp)
    if cache and key in _GCACHE:
        return _GCACHE[key]
    n, edges, truth = _BUILDERS[sp["kind"]](sp)
    g = rx.G(n, edges)
    if (g.deg < 1).any():
        raise ValueError("graph has isolated nodes (rx.G needs degree >= 1)")
    bm = bridge_mask(g, truth)
    meta = dict(id=spec_id(sp), kind=sp["kind"], n=g.n, m=len(edges), seed=sp["seed"], params=sp,
                q=None if truth is None else int(len(np.unique(truth))), connected=bool(is_connected(g)),
                bridge_mask=bm, bridge_idx=None if bm is None else np.flatnonzero(bm),
                n_bridges=None if bm is None else int(bm.sum()), dmax=g.dmax,
                mean_deg=float(2 * len(edges) / g.n))
    out = (g, None if truth is None else np.asarray(truth), meta)
    if cache:
        _GCACHE[key] = out
    return out


def zoo(name, seeds=(0, 1, 2)):
    """List of graph specs.  names:
      core       3 blocks (cycle, clique, sbm) x bridges {1,2,4,8}, sizes 24+24, wb=1          (12 / seed)
      core_full  core x sizes {24+24, 16+32, 8+40} x wb {1, 0.25}                               (72 / seed)
      wide       barbell_path x2, grid_neck x3, ring_of_cliques x2, sbm_q x2, binary_tree, lollipop,
                 sbm_het sigma {0,.3,.6,1}                                                      (15 / seed)
      neg        er, shuffled, random_regular d{3,4,6,8}, watts_strogatz  (truth None)           (7 / seed)
      disc       two_block with bridges=0 (disconnected positive control, truth known)            (3 / seed)
      scale      n in {48,96,192,384} x block {cycle, clique, sbm}, bridges 2                    (12 / seed)
    """
    sp = []
    for s in seeds:
        if name in ("core", "core_full"):
            sizes_l = [(24, 24)] if name == "core" else [(24, 24), (16, 32), (8, 40)]
            wbs = [1.0] if name == "core" else [1.0, 0.25]
            for blk in ("cycle", "clique", "sbm"):
                for sz in sizes_l:
                    for b in (1, 2, 4, 8):
                        for wb in wbs:
                            sp.append(dict(kind="two_block", block=blk, sizes=sz, bridges=b, wb=wb, seed=s))
        elif name == "wide":
            sp += [dict(kind="barbell_path", k=16, L=4, seed=s), dict(kind="barbell_path", k=16, L=16, seed=s)]
            sp += [dict(kind="grid_neck", a=5, w=w, seed=s) for w in (1, 2, 4)]
            sp += [dict(kind="ring_of_cliques", q=4, s=12, seed=s), dict(kind="ring_of_cliques", q=6, s=8, seed=s)]
            sp += [dict(kind="sbm_q", q=3, size=16, seed=s), dict(kind="sbm_q", q=4, size=12, seed=s)]
            sp += [dict(kind="binary_tree", depth=5, seed=s), dict(kind="lollipop", k=24, L=24, seed=s)]
            sp += [dict(kind="sbm_het", sigma=sg, seed=s) for sg in (0.0, 0.3, 0.6, 1.0)]
        elif name == "neg":
            sp += [dict(kind="er", n=48, deg=6.0, seed=s), dict(kind="shuffled", sizes=(24, 24), seed=s)]
            sp += [dict(kind="random_regular", n=48, d=d, seed=s) for d in (3, 4, 6, 8)]
            sp += [dict(kind="watts_strogatz", n=48, k=4, p=0.3, seed=s)]
        elif name == "disc":
            sp += [dict(kind="two_block", block=b, sizes=(24, 24), bridges=0, seed=s) for b in ("cycle", "clique", "sbm")]
        elif name == "scale":
            for n in (48, 96, 192, 384):
                for blk in ("cycle", "clique", "sbm"):
                    sp.append(dict(kind="two_block", block=blk, sizes=(n // 2, n // 2), bridges=2, seed=s))
        else:
            raise KeyError(f"unknown zoo {name!r}")
    return sp


@dataclass
class Case:
    spec: dict
    g: object
    truth: object
    meta: dict


def build_zoo(name, seeds=(0, 1, 2)):
    """zoo(name, seeds) built into [Case(spec, g, truth, meta)]."""
    out = []
    for sp in zoo(name, seeds):
        g, t, m = make_graph(sp)
        out.append(Case(sp, g, t, m))
    return out


# --------------------------------------------------------------------------------------------------
# initial conditions
# --------------------------------------------------------------------------------------------------
def init(kind, g, R, rng, truth=None, scale=1.0, dc=0.0, rms=RMS_STD, **kw):
    """Initial states (R, n).  All random kinds are matched to RMS `rms` (default = RMS of U(-1,1) = 0.577)
    so that the absolute failure threshold means the same thing for every kind.
      uniform    rng.uniform(-1,1,(R,n)) * rms/RMS_STD   (identical to the rx/wt recipe at default rms)
      gauss, rademacher, student_t3 (clipped at kw clip=5 sigma; clip=None disables; heavy tails otherwise
                 put u0 itself above FAIL_AMP now and then)
      impulse    one node = kw amp (default 1), node = kw node (int or (R,)) else random per realisation
      low_mode   +-Fiedler vector of L_rw (kw k=1 for the k-th) + kw noise_frac=0.1 gaussian noise, RMS=rms
      high_mode  +-top eigenvector (kw k=-1) + noise, RMS=rms
      block_offset  uniform noise (x kw noise=1) + offset per TRUE block (needs truth): block b gets
                 (b-(q-1)/2)*delta/(q-1), kw delta=1  (q=2: -delta/2, +delta/2)
    modifiers (applied in this order): u = u*scale + dc.   rng: np.random.Generator."""
    n = g.n
    if kind == "uniform":
        u = rng.uniform(-1, 1, (R, n)) * (rms / RMS_STD)
    elif kind == "gauss":
        u = rng.normal(0, 1, (R, n)) * rms
    elif kind == "rademacher":
        u = rng.choice([-1.0, 1.0], (R, n)) * rms
    elif kind == "student_t3":
        u = rng.standard_t(3, (R, n)) / math.sqrt(3.0) * rms
        clip = kw.get("clip", 5.0)
        if clip is not None:
            u = np.clip(u, -clip * rms, clip * rms)
    elif kind == "impulse":
        node = kw.get("node")
        node = rng.integers(0, n, R) if node is None else np.broadcast_to(np.asarray(node), (R,))
        u = np.zeros((R, n))
        u[np.arange(R), node] = kw.get("amp", 1.0)
    elif kind in ("low_mode", "high_mode"):
        lam, Phi, d = spectrum(g)
        k = kw.get("k", 1 if kind == "low_mode" else -1)
        v = Phi[:, k]
        v = v / math.sqrt((v ** 2).mean()) * rms
        sgn = rng.choice([-1.0, 1.0], R)[:, None]
        u = sgn * v[None, :] + rng.normal(0, 1, (R, n)) * kw.get("noise_frac", 0.1) * rms
    elif kind == "block_offset":
        if truth is None:
            raise ValueError("block_offset needs truth=")
        truth = np.asarray(truth)
        q = int(truth.max()) + 1
        delta = kw.get("delta", 1.0)
        off = (np.arange(q) - (q - 1) / 2) * delta / max(q - 1, 1)
        u = rng.uniform(-1, 1, (R, n)) * kw.get("noise", 1.0) + off[truth][None, :]
    else:
        raise ValueError(f"unknown init kind {kind!r}")
    return u * scale + dc


# --------------------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------------------
@dataclass
class Run:
    """Result of run()/run_diff()/Runner().  X: (steps+1,R,n); fail_step: (R,) int, -1 if never failed."""
    X: np.ndarray
    fail_step: np.ndarray
    g: object
    u0: np.ndarray
    c: float
    scheme: str
    force: str
    ps: dict
    pf: dict
    steps: int
    fail_amp: float
    eta: float | None = None

    @property
    def fail(self):
        return self.fail_step >= 0

    @property
    def ok(self):
        return self.fail_step < 0

    @property
    def R(self):
        return self.X.shape[1]

    @property
    def valid_len(self):
        return valid_len(self.fail_step, self.X.shape[0])

    def __iter__(self):                      # allows `X, fs = run(...)`-style unpacking if wanted
        yield self.X
        yield self.fail_step


def _as2d(u0):
    u0 = np.asarray(u0, float)
    return u0[None, :] if u0.ndim == 1 else u0


def run(g, scheme, force, u0, c=1.0, steps=rx.T, ps=None, pf=None, fail_amp=None):
    """Wrap wt.simulate.  ps/pf: partial dicts merged over the registry defaults.  fail_amp overrides
    wt.FAIL_AMP (20) for this call.  Returns Run."""
    u0 = _as2d(u0)
    ps = resolve_params(wt.SCHEMES, scheme, ps)
    pf = resolve_params(wt.FORCES, force, pf)
    with fail_amp_override(fail_amp):
        amp = wt.FAIL_AMP
        X, fs = wt.simulate(g, scheme, force, ps, pf, u0, c, steps)
    return Run(X, fs, g, u0, float(c), scheme, force, ps, pf, steps, amp)


def _loop(g, scheme, force, u0, c, steps, ps, pf, fail_amp, diffusion=False, post=None, observer=None,
          early_stop=False, record=True):
    """Generic integrator replicating wt.simulate step-for-step, plus hooks:
    post(t, cur, nxt, ctx): may modify nxt in place BEFORE the failure test;
    observer(t, ctx, cur, prev, F, Fp, nxt, alive).  diffusion=True: u+ = u + c*F (c is eta).
    Returns (X or None, fail_step)."""
    ffn = wt.FORCES[force][0]
    sfn = None if diffusion else wt.SCHEMES[scheme][0]
    amp = wt.FAIL_AMP if fail_amp is None else float(fail_amp)
    Rn, n = u0.shape
    X = None
    if record:
        X = np.zeros((steps + 1, Rn, n))
        X[0] = u0
    prev, cur = u0.copy(), u0.copy()
    fail_step = np.full(Rn, -1)
    ctx = wt.Ctx(g=g, c=c, pf=pf, ps=ps, u0=u0)
    F_prev = None
    with np.errstate(all="ignore"):
        for t in range(steps):
            ctx.t = t
            F = ffn(ctx, cur)
            Fp = F if F_prev is None else F_prev
            nxt = cur + c * F if diffusion else sfn(ctx, cur, prev, F, Fp)
            if post is not None:
                post(t, cur, nxt, ctx)
            if observer is not None:
                observer(t, ctx, cur, prev, F, Fp, nxt, fail_step < 0)
            bad = ~(np.abs(nxt) < amp).all(axis=1) & (fail_step < 0)
            fail_step[bad] = t + 1
            dead = fail_step >= 0
            nxt[dead] = 0.0
            F[dead] = 0.0
            prev, cur, F_prev = cur, nxt, F
            if record:
                X[t + 1] = cur
            if early_stop and dead.all():
                break
    return X, fail_step


def run_diff(g, force, u0, eta=0.5, steps=rx.T, pf=None, fail_amp=None):
    """Diffusion update u+ = u + eta*F(u) with ANY registered wt force (F = A(u)-u for the rx aggregations,
    so force='mean', eta=0.5 reproduces rx.simulate(..., 'diff')).  Forces are evaluated through a wt.Ctx
    (rx.simulate's diffusion path only knows the four rx aggregations).  Returns Run (scheme='diffusion')."""
    u0 = _as2d(u0)
    pf = resolve_params(wt.FORCES, force, pf)
    amp = wt.FAIL_AMP if fail_amp is None else float(fail_amp)
    X, fs = _loop(g, None, force, u0, eta, steps, {}, pf, amp, diffusion=True)
    return Run(X, fs, g, u0, float("nan"), "diffusion", force, {}, pf, steps, amp, eta=float(eta))


@dataclass
class Runner:
    """Picklable callable runner(g, u0, steps=None) -> Run for stimulus()/confounders/pmap jobs.
    kind='scheme' uses (scheme, force, c); kind='diff' uses (force, eta)."""
    kind: str = "scheme"
    scheme: str = "wave"
    force: str = "mean"
    c: float = 1.0
    eta: float = 0.5
    ps: dict = field(default_factory=dict)
    pf: dict = field(default_factory=dict)
    steps: int = rx.T
    fail_amp: float | None = None

    def __call__(self, g, u0, steps=None, post=None):
        steps = self.steps if steps is None else steps
        u0 = _as2d(u0)
        if self.kind == "diff":
            if post is None:
                return run_diff(g, self.force, u0, self.eta, steps, self.pf, self.fail_amp)
            amp = wt.FAIL_AMP if self.fail_amp is None else self.fail_amp
            X, fs = _loop(g, None, self.force, u0, self.eta, steps, {}, self.pf, amp, diffusion=True, post=post)
            return Run(X, fs, g, u0, float("nan"), "diffusion", self.force, {}, self.pf, steps, amp, eta=self.eta)
        if post is None:
            return run(g, self.scheme, self.force, u0, self.c, steps, self.ps, self.pf, self.fail_amp)
        amp = wt.FAIL_AMP if self.fail_amp is None else self.fail_amp
        X, fs = _loop(g, self.scheme, self.force, u0, self.c, steps, self.ps, self.pf, amp, post=post)
        return Run(X, fs, g, u0, self.c, self.scheme, self.force, self.ps, self.pf, steps, amp)

    @property
    def label(self):
        f = f"{self.force}" + (f"[{_kv(self.pf)}]" if self.pf else "")
        if self.kind == "diff":
            return f"diffusion(eta={self.eta:g})+{f}"
        s = f"{self.scheme}" + (f"[{_kv(self.ps)}]" if self.ps else "")
        return f"{s}+{f} c={self.c:g}"


def _kv(d):
    return ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in d.items())


def make_runner(scheme="wave", force="mean", c=1.0, ps=None, pf=None, steps=rx.T, fail_amp=None):
    """Runner for a (scheme, force, c) configuration; ps/pf partial dicts merged over registry defaults."""
    return Runner("scheme", scheme, force, float(c), 0.5, resolve_params(wt.SCHEMES, scheme, ps),
                  resolve_params(wt.FORCES, force, pf), steps, fail_amp)


def make_diff_runner(force="mean", eta=0.5, pf=None, steps=rx.T, fail_amp=None):
    """Runner for the diffusion update with any wt force."""
    return Runner("diff", "diffusion", force, 1.0, float(eta), {}, resolve_params(wt.FORCES, force, pf),
                  steps, fail_amp)


def fail_frac(g, scheme, force, u0, c, steps=2000, ps=None, pf=None, fail_amp=None):
    """Fraction of realisations that fail within `steps` (early-exits once all failed; no trajectory kept)."""
    ps = resolve_params(wt.SCHEMES, scheme, ps)
    pf = resolve_params(wt.FORCES, force, pf)
    _, fs = _loop(g, scheme, force, _as2d(u0), c, steps, ps, pf, fail_amp, early_stop=True, record=False)
    return float((fs >= 0).mean())


# --------------------------------------------------------------------------------------------------
# trajectory metrics
# --------------------------------------------------------------------------------------------------
def ac_rms(X, g=None, weighted=False):
    """(T+1, R): RMS over nodes of the state with the node mean removed at every step
    (weighted=True removes the degree-weighted mean, which is the conserved one for the mean force)."""
    if weighted:
        w = wdeg(g)
        w = w / w.sum()
        Xc = X - (X @ w)[:, :, None]
    else:
        Xc = X - X.mean(2, keepdims=True)
    return np.sqrt((Xc ** 2).mean(2))


def growth_rate(X, fail_step=None, frac=0.5, floor_rel=1e-12, min_pts=4, min_len=6):
    """(R,) least-squares slope of log AC-RMS vs step over the last `frac` of the VALID window (for a failed
    realisation: the window before it failed).  Positive = growing.  Points with AC-RMS < floor_rel*AC-RMS(0)
    are dropped (decayed to round-off); NaN if fewer than min_pts remain or the valid window is < min_len steps
    (owa/vw at c=1 fail within ~8-10 steps, so their rate rests on only 4-5 points).
    Only meaningful for non-dead runs; use ac_final/dead for decayed ones."""
    T1, R, _ = X.shape
    rms = ac_rms(X)
    L = valid_len(fail_step, T1, R)
    out = np.full(R, np.nan)
    for r in range(R):
        if L[r] < min_len:
            continue
        lo = int(L[r] * (1 - frac))
        seg = rms[lo:L[r], r]
        t = np.arange(lo, L[r])
        keep = seg > floor_rel * max(rms[0, r], 1e-300)
        if keep.sum() >= min_pts:
            out[r] = np.polyfit(t[keep], np.log(seg[keep]), 1)[0]
    return out


def dc_drift(X, g, fail_step=None, W=50):
    """dict(plain, dw): (R,) mean over the last W valid steps of the node mean (plain) / degree-weighted mean (dw)
    minus the same quantity at t=0.  For a failed realisation the last W steps BEFORE failure are used.
    dw is exactly conserved (up to roundoff) by the mean force."""
    T1, R, _ = X.shape
    d = wdeg(g)
    mp, mw = X.mean(2), X @ (d / d.sum())
    L = valid_len(fail_step, T1, R)
    plain, dw = np.empty(R), np.empty(R)
    for r in range(R):
        lo = max(0, L[r] - W)
        plain[r] = mp[lo:L[r], r].mean() - mp[0, r]
        dw[r] = mw[lo:L[r], r].mean() - mw[0, r]
    return dict(plain=plain, dw=dw)


def _tail_stat(X, fail_step, W, ac):
    T1, R, _ = X.shape
    out = np.full(R, np.nan)
    ok = _ok(fail_step, R)
    for r in np.flatnonzero(ok):
        tail = X[-W:, r, :]
        if ac:
            tail = tail - tail.mean(1, keepdims=True)
        out[r] = math.sqrt((tail ** 2).mean())
    return out


def ac_final(X, fail_step=None, W=50):
    """(R,) AC-RMS over the last W steps / AC-RMS(u0).  NaN for failed realisations.  (wt.run_one divides by the
    RMS of u0 INCLUDING its mean; this divides by the AC-RMS, which differs only when u0 has a DC offset.)"""
    a0 = ac_rms(X[:1])[0]
    return _tail_stat(X, fail_step, W, True) / np.where(a0 > 0, a0, np.nan)


def rms_final(X, fail_step=None, W=50):
    """(R,) RMS (DC included) over the last W steps / RMS(u0) (wt.run_one's rms_final).  NaN if failed."""
    r0 = _rms_rows(X[0])
    return _tail_stat(X, fail_step, W, False) / np.where(r0 > 0, r0, np.nan)


def dead(X, fail_step=None, W=50, thr=0.05):
    """(R,) bool: not failed and ac_final < thr (wt.run_one's dead criterion)."""
    af = ac_final(X, fail_step, W)
    return np.where(np.isnan(af), False, af < thr)


def ac_peak(X, fail_step=None):
    """(R,) max over valid steps of AC-RMS / AC-RMS(0)."""
    T1, R, _ = X.shape
    rms = ac_rms(X)
    L = valid_len(fail_step, T1, R)
    return np.array([rms[:L[r], r].max() / max(rms[0, r], 1e-300) for r in range(R)])


def traj_metrics(run_, W=50, dead_thr=0.05):
    """All per-realisation trajectory metrics for a Run -> dict of (R,) arrays:
    fail (bool), fail_step, growth_rate, dc_plain, dc_dw, ac_final, rms_final, dead, ac_peak."""
    X, fs, g = run_.X, run_.fail_step, run_.g
    dd = dc_drift(X, g, fs, W)
    af = ac_final(X, fs, W)
    return dict(fail=fs >= 0, fail_step=fs, growth_rate=growth_rate(X, fs), dc_plain=dd["plain"], dc_dw=dd["dw"],
                ac_final=af, rms_final=rms_final(X, fs, W), dead=np.where(np.isnan(af), False, af < dead_thr),
                ac_peak=ac_peak(X, fs))


def traj_summary(run_, W=50, dead_thr=0.05):
    """Scalar summary of a Run for a CSV row: R, n_ok, fail_frac, mean_fail_step, dead_frac, ac_final (median over
    ok), rms_final (median over ok), growth_rate (median over all valid), dc_dw / dc_plain (median), ac_peak."""
    m = traj_metrics(run_, W, dead_thr)
    ok = ~m["fail"]
    med = lambda a, mask=None: float(np.nanmedian(a if mask is None else a[mask])) if np.isfinite(a if mask is None else a[mask]).any() else float("nan")  # noqa: E731
    return dict(R=len(ok), n_ok=int(ok.sum()), fail_frac=float((~ok).mean()),
                mean_fail_step=float(m["fail_step"][~ok].mean()) if (~ok).any() else float("nan"),
                dead_frac=float(m["dead"].mean()), ac_final=med(m["ac_final"], ok), rms_final=med(m["rms_final"], ok),
                growth_rate=med(m["growth_rate"]), dc_dw=med(m["dc_dw"]), dc_plain=med(m["dc_plain"]),
                ac_peak=med(m["ac_peak"]))


def fail_modes(g, scheme, force, u0, c=1.0, steps=rx.T, ps=None, pf=None, fail_amp=None, big=1e6):
    """Drift vs oscillation.  Re-runs with the failure threshold raised to `big`, then finds for each realisation
    the first step at which max_i |u_i| >= A (raw; identical to the normal fail_step) and the first step at which
    max_i |u_i - mean(u)| >= A (AC part only), A = fail_amp.  Returns dict(raw_step, ac_step: (R,) int, -1 if never;
    raw_fail, ac_fail, dc_only = raw_fail & ~ac_fail).  AC crossings are only seen up to `big`."""
    A = wt.FAIL_AMP if fail_amp is None else float(fail_amp)
    r = run(g, scheme, force, u0, c, steps, ps, pf, fail_amp=big)
    X = r.X
    with np.errstate(all="ignore"):
        raw = np.abs(X).max(2) >= A                                  # (T1,R)
        acm = np.abs(X - X.mean(2, keepdims=True)).max(2) >= A
    def first(m):
        any_ = m.any(0)
        return np.where(any_, m.argmax(0), -1)
    raw_step, ac_step = first(raw), first(acm)
    return dict(raw_step=raw_step, ac_step=ac_step, raw_fail=raw_step >= 0, ac_fail=ac_step >= 0,
                dc_only=(raw_step >= 0) & (ac_step < 0))


def energy(X, g, c, fail_step=None, kind="staggered"):
    """(T, R) energy series for the wave update, in the D inner product, M = D - A = D L_rw:
      staggered (default, exactly conserved by wave+mean): E_t = 1/2||u_{t+1}-u_t||_D^2 + 1/2 c^2 u_t^T M u_{t+1}
      naive:  E_t = 1/2||u_{t+1}-u_t||_D^2 + 1/2 c^2 u_t^T M u_t   (oscillates at O(c^2 lambda))
    NaN where u_{t+1} is not valid (failed realisations).  Not conserved for damped/viscous/nonlinear schemes."""
    T1, R, n = X.shape
    d = wdeg(g)
    Xf = X.reshape(T1 * R, n)
    MX = (Xf * d - Xf @ g.A).reshape(T1, R, n)
    dv = X[1:] - X[:-1]
    kin = 0.5 * (d * dv ** 2).sum(2)
    pot = 0.5 * c * c * (X[:-1] * (MX[1:] if kind == "staggered" else MX[:-1])).sum(2)
    E = kin + pot
    L = valid_len(fail_step, T1, R)
    for r in range(R):
        if L[r] < T1:
            E[max(L[r] - 1, 0):, r] = np.nan
    return E


def energy_drift(E):
    """dict(ratio_end, ratio_max, ratio_min): (R,) last-valid / max / min of E_t relative to E_0."""
    E0 = E[0]
    with np.errstate(all="ignore"):
        last = np.array([E[~np.isnan(E[:, r]), r][-1] if (~np.isnan(E[:, r])).any() else np.nan
                         for r in range(E.shape[1])])
        return dict(ratio_end=last / E0, ratio_max=np.nanmax(E, 0) / E0, ratio_min=np.nanmin(E, 0) / E0)


def mode_energy(X, g, fail_step=None, n_low=2, high_frac=0.25, weight="disp", full=False):
    """Projection on the D-orthonormal eigenvectors of L_rw (a_k = phi_k^T D u).  Energy per mode: a_k^2
    (weight='disp') or lam_k a_k^2 (weight='pot').  Mode 0 (consensus) is excluded.  Returns dict:
    low (T+1,R) = fraction in modes 1..n_low, high (T+1,R) = fraction in the top ceil(high_frac*(n-1)) modes,
    total (T+1,R) = sum over modes 1..n-1; (+ 'a': (T+1,R,n) modal coordinates if full).  NaN after failure."""
    T1, R, n = X.shape
    lam, Phi, d = spectrum(g)
    a = (X * d) @ Phi                                               # (T1,R,n)
    e = a ** 2 * (lam if weight == "pot" else 1.0)
    e[..., 0] = 0.0
    tot = e.sum(2)
    nh = max(1, int(math.ceil(high_frac * (n - 1))))
    with np.errstate(all="ignore"):
        low = e[..., 1:1 + n_low].sum(2) / tot
        high = e[..., n - nh:].sum(2) / tot
    L = valid_len(fail_step, T1, R)
    for r in range(R):
        if L[r] < T1:
            low[L[r]:, r] = high[L[r]:, r] = tot[L[r]:, r] = np.nan
    out = dict(low=low, high=high, total=tot)
    if full:
        out["a"] = a
    return out


def temporal_spectrum(X, fail_step=None, start_frac=0.0, nyq_band=0.45):
    """Temporal power spectrum of the AC state (node mean removed, per-node time mean removed, Hann window),
    pooled over nodes, per realisation, over the valid window from start_frac*L to L.  Returns dict(centroid,
    nyquist): (R,); centroid in cycles/step (0..0.5), nyquist = power fraction at f >= nyq_band (period ~2
    checkerboard flicker).  NaN if the window is < 16 steps or has no power."""
    T1, R, _ = X.shape
    L = valid_len(fail_step, T1, R)
    cen, nyq = np.full(R, np.nan), np.full(R, np.nan)
    for r in range(R):
        s = int(start_frac * L[r])
        seg = X[s:L[r], r, :]
        if len(seg) < 16:
            continue
        seg = seg - seg.mean(1, keepdims=True)
        seg = (seg - seg.mean(0)) * np.hanning(len(seg))[:, None]
        P = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).sum(1)
        f = np.fft.rfftfreq(len(seg))
        tot = P[1:].sum()
        if tot > 1e-30:
            cen[r] = (f[1:] * P[1:]).sum() / tot
            nyq[r] = P[f >= nyq_band].sum() / tot
    return dict(centroid=cen, nyquist=nyq)


def lyapunov(g, scheme, force, u0, c=1.0, steps=500, ps=None, pf=None, eps=1e-8, seed=0, fail_amp=None,
             sat=1e-3, dfloor=1e-13):
    """Divergence rate of a twin started eps away (random direction, ||delta||_2 = eps, cur=prev perturbed).
    d(t) = ||X_twin(t) - X(t)||_2.  rate = LS slope of log d over steps 0..t_sat where t_sat = first step with
    d > sat*||X(t)||_2, using only points with d > dfloor.  Returns dict(rate (R,) per step, t_sat (R,), d
    (T+1,R)).  NaN rate if either run failed or < 10 usable points (e.g. decaying runs reach round-off at
    once).  Caveat: the twin is not renormalised, so slow Lyapunov exponents may be underestimated.  A
    non-chaotic neutral wave gives ~0 (or slightly positive, polynomial growth)."""
    u0 = _as2d(u0)
    rng = np.random.default_rng(seed)
    dl = rng.normal(0, 1, u0.shape)
    dl *= eps / np.linalg.norm(dl, axis=1, keepdims=True)
    a = run(g, scheme, force, u0, c, steps, ps, pf, fail_amp)
    b = run(g, scheme, force, u0 + dl, c, steps, ps, pf, fail_amp)
    d = np.linalg.norm(b.X - a.X, axis=2)
    nx = np.linalg.norm(a.X, axis=2)
    T1, R = d.shape
    rate, tsat = np.full(R, np.nan), np.full(R, -1)
    for r in range(R):
        if a.fail_step[r] >= 0 or b.fail_step[r] >= 0:
            continue
        over = d[:, r] > sat * np.maximum(nx[:, r], 1e-300)
        ts = int(over.argmax()) if over.any() else T1
        tsat[r] = ts
        t = np.arange(ts)
        keep = d[:ts, r] > dfloor
        if keep.sum() >= 10:
            rate[r] = np.polyfit(t[keep], np.log(d[:ts, r][keep]), 1)[0]
    return dict(rate=rate, t_sat=tsat, d=d)


def activation(g, scheme, force, u0, c=1.0, steps=500, ps=None, pf=None, thr=0.1, fail_amp=None):
    """How often a scheme modifies the plain wave step.  At every step and node the modification
    |u_next - u_wave| / max(|u_wave - u|, 1e-6*RMS(u0)) is compared with thr (u_wave = 2u - u_prev + c^2 F with
    the same F).  Relevant for vel_shrink/drag/limiter/cfl/energy/saturate (for damp/viscous it is trivially on).
    A realisation that has converged to exact consensus (static state) reads 0 from then on.
    Returns dict(frac (steps,R) fraction of nodes active at each step (NaN once failed),
    mean_frac (R,), late_frac (R,) over the last half, first_quiet (R,) first step after which frac < 0.01 for
    good (-1 never))."""
    u0 = _as2d(u0)
    ps = resolve_params(wt.SCHEMES, scheme, ps)
    pf = resolve_params(wt.FORCES, force, pf)
    Rn = u0.shape[0]
    frac = np.full((steps, Rn), np.nan)
    floor = 1e-6 * _rms_rows(u0)[:, None]

    def obs(t, ctx, cur, prev, F, Fp, nxt, alive):
        plain = 2 * cur - prev + c * c * F
        a = np.abs(nxt - plain) / np.maximum(np.abs(plain - cur), floor)
        fr = (a > thr).mean(1)
        fr[~alive] = np.nan
        frac[t] = fr

    _, fs = _loop(g, scheme, force, u0, c, steps, ps, pf, fail_amp, observer=obs, record=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(frac, 0)
        late = np.nanmean(frac[steps // 2:], 0)
    fq = np.full(Rn, -1)
    for r in range(Rn):
        f = frac[:, r]
        if fs[r] >= 0:
            continue
        loud = np.flatnonzero(np.nan_to_num(f, nan=0.0) >= 0.01)
        if not np.isnan(f).all():
            fq[r] = 0 if len(loud) == 0 else (loud[-1] + 1 if loud[-1] + 1 < steps else -1)
    return dict(frac=frac, mean_frac=mean, late_frac=late, first_quiet=fq, fail_step=fs)


# --------------------------------------------------------------------------------------------------
# force property probes
# --------------------------------------------------------------------------------------------------
def eval_force(g, force, U, pf=None, t=0, c=1.0):
    """F(U) for a registered wt force; U (n,) or (R,n) -> (R,n).  A FRESH wt.Ctx is created per call, so
    stateful forces (edge_memory) are evaluated from their initial state (weights = 1, i.e. as the mean)."""
    U = _as2d(U)
    pf = resolve_params(wt.FORCES, force, pf)
    ctx = wt.Ctx(g=g, c=c, pf=pf, ps={}, u0=U, t=t)
    with np.errstate(all="ignore"):
        return wt.FORCES[force][0](ctx, U)


def _jac_parts(g, force, u, pf, h, t):
    u = np.asarray(u, float)
    n = g.n
    hh = h * max(1.0, float(np.sqrt((u ** 2).mean())))
    U = np.tile(u, (2 * n + 1, 1))
    U[np.arange(n), np.arange(n)] += hh
    U[n + np.arange(n), np.arange(n)] -= hh
    F = eval_force(g, force, U, pf, t)                 # rows: u+h e_j (n), u-h e_j (n), u (1); fresh Ctx
    Fp, Fm, F0 = F[:n], F[n:2 * n], F[2 * n]
    return (Fp - Fm).T / (2 * hh), (Fp - F0).T / hh, (F0 - Fm).T / hh


def jacobian(g, force, u, pf=None, h=1e-6, t=0, return_kink=False):
    """Central-difference Jacobian J[i,j] = dF_i/du_j at u (n,), one batched evaluation of 2n+1 states (fresh Ctx).
    Step h*max(1, rms(u)).  WARNING: rank/sort/clip based forces (trim, owa, median, symowa_*, huber*, clipped
    edges, vw/gauss at ties) are discontinuous or have kinks; J is then an arbitrary one-sided mix of the
    derivatives and is wrong if a kink lies within +-h of u (tied values, Rademacher states, exact saturation).
    return_kink=True also returns kink = ||J_fwd - J_bwd||_F / ||J||_F (~h for smooth points, O(1) at kinks/ties)."""
    Jc, Jf, Jb = _jac_parts(g, force, u, pf, h, t)
    if return_kink:
        return Jc, _rel(np.linalg.norm(Jf - Jb), np.linalg.norm(Jc))
    return Jc


def modal_roots(mu, c, k=0.0, nu=0.0):
    """Roots z of the per-mode characteristic polynomial of damp/viscous/damp_visc applied to a linear force
    F = mu*u (mu = -lambda for the mean force):  z^2 - (2 - k + (c^2+nu) mu) z + (1 - k + nu mu) = 0.
    mu may be a real/complex array; returns complex array (..., 2) via the closed form (exact double roots)."""
    mu = np.asarray(mu, complex)
    b = 2 - k + (c * c + nu) * mu
    q = 1 - k + nu * mu
    s = np.sqrt(b * b - 4 * q)
    return np.stack([(b + s) / 2, (b - s) / 2], -1)


def modal_rho(mu, c, k=0.0, nu=0.0):
    """max |z| over modes and both roots."""
    return float(np.abs(modal_roots(mu, c, k, nu)).max())


def c_crit_linear(mu, k=0.0, nu=0.0, c_hi=4.0, rho_tol=1e-7, n_scan=200):
    """Smallest c with modal_rho(mu, c) > 1 + rho_tol (scan then bisection); inf if stable up to c_hi.
    For the mean force and k=nu=0 this is 2/sqrt(lambda_max)."""
    cs = np.linspace(c_hi / n_scan, c_hi, n_scan)
    bad = [modal_rho(mu, c, k, nu) > 1 + rho_tol for c in cs]
    if not any(bad):
        return float("inf")
    i = int(np.argmax(bad))
    lo, hi = (cs[i - 1] if i > 0 else 0.0), cs[i]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if modal_rho(mu, mid, k, nu) > 1 + rho_tol:
            hi = mid
        else:
            lo = mid
    return float(hi)


def force_props(g, force, u=None, pf=None, rng=None, h=1e-6, shift_a=1.0, scales=(0.1, 10.0), return_jac=False,
                t=0):
    """Property probes of a force at the 1-D state u (n,) (default: one U(-1,1) draw).  All floats.
    Defect keys (zero for the mean force on any state):
      oddness       ||F(u)+F(-u)|| / ||F(u)||
      shift_abs/rel ||F(u+a1)-F(u)||  (a = shift_a)  and divided by ||F(u)||
      homog_abs/rel max_s ||F(su)/s - F(u)||, s in `scales`, and divided by ||F(u)||
      const_defect  ||F(1.7*1)||  (F of a constant state)
      momentum / momentum_rel   sum_i d_i F_i  and |sum d F| / sum d|F|   (degree-weighted zero-sum)
      dsym_abs/rel  ||DJ - (DJ)^T||_F  and divided by ||DJ||_F       (D-symmetry of the Jacobian)
    Linear stability (D-metric, S = sym(DJ), T = D^-1/2 S D^-1/2):
      lam_max       max eig of -T  (stiffness; = lambda_max(L_rw) for mean)
      eig_sym_max   max eig of T (> ~1e-8 => anti-restoring direction exists; ~0 for shift-invariant forces)
      c_crit_pred   2/sqrt(lam_max)  (wave scheme; inf if lam_max <= 0)
      rho_J         spectral radius of J (true, possibly complex eigenvalues)
      c_crit_eig    c_crit_linear(eig(J))  using the actual eigenvalues (differs from c_crit_pred when J non-normal)
      jac_kink      ||J_fwd - J_bwd||_F / ||J||_F  (~1e-6 at smooth points; O(1) when u sits on a kink/tie of a
                    rank/clip force, in which case every Jacobian-derived number above is unreliable)
    Jacobian is from a CENTRAL difference (see jacobian()); discontinuous forces: read the warning there.
    return_jac=True adds key 'J' (n,n)."""
    n = g.n
    if u is None:
        u = (rng or np.random.default_rng(0)).uniform(-1, 1, n)
    u = np.asarray(u, float).reshape(-1)
    d = wdeg(g)
    f = lambda v: eval_force(g, force, v, pf, t)[0]  # noqa: E731
    F0 = f(u)
    nF = np.linalg.norm(F0)
    out = {}
    out["oddness"] = _rel(np.linalg.norm(F0 + f(-u)), nF)
    sh = np.linalg.norm(f(u + shift_a) - F0)
    out["shift_abs"], out["shift_rel"] = float(sh), _rel(sh, nF)
    hom = max(np.linalg.norm(f(s * u) / s - F0) for s in scales)
    out["homog_abs"], out["homog_rel"] = float(hom), _rel(hom, nF)
    out["const_defect"] = float(np.linalg.norm(f(np.full(n, 1.7))))
    out["momentum"] = float((d * F0).sum())
    out["momentum_rel"] = _rel(abs((d * F0).sum()), (d * np.abs(F0)).sum())
    J, kink = jacobian(g, force, u, pf, h, t, return_kink=True)
    DJ = d[:, None] * J
    out["dsym_abs"] = float(np.linalg.norm(DJ - DJ.T))
    out["dsym_rel"] = _rel(out["dsym_abs"], np.linalg.norm(DJ))
    S = 0.5 * (DJ + DJ.T)
    Tm = S / np.sqrt(np.outer(d, d))
    ev = np.linalg.eigvalsh(0.5 * (Tm + Tm.T))
    out["lam_max"] = float(-ev[0])
    out["eig_sym_max"] = float(ev[-1])
    out["c_crit_pred"] = float(2 / math.sqrt(-ev[0])) if -ev[0] > 1e-12 else float("inf")
    mu = np.linalg.eigvals(J)
    out["rho_J"] = float(np.abs(mu).max())
    out["c_crit_eig"] = c_crit_linear(mu)
    out["jac_kink"] = kink
    if return_jac:
        out["J"] = J
    return out


def force_props_rows(g, force, U, pf=None, **kw):
    """force_props for every row of U (R,n) -> list of dicts (ready for write_csv)."""
    return [force_props(g, force, u, pf, **kw) for u in _as2d(U)]


def c_crit_empirical(g, scheme, force, u0=None, ps=None, pf=None, steps=2000, c_lo=0.02, c_hi=3.0, n_scan=16,
                     rtol=0.01, R=8, seed=0, fail_amp=None):
    """Empirical critical wave speed.  Scans c on a geometric grid c_lo..c_hi (n_scan points); the first grid
    point with fail_frac > 0 (any realisation failing within `steps`) brackets the threshold, which is then
    bisected to relative width rtol.  u0 default: U(-1,1) with R realisations, seed.
    Returns dict(c_crit, status, non_monotone, scan_c, scan_fail):
      status 'ok' | 'unstable_at_lo' (c_crit=nan) | 'stable_to_hi' (c_crit=inf).
    non_monotone=True if some c above c_crit was stable again (bisection finds only the FIRST threshold).
    Uses early exit, so unstable settings are cheap, stable ones cost steps*(n_scan+log2) updates."""
    if u0 is None:
        u0 = init("uniform", g, R, np.random.default_rng(seed))
    ff = lambda c: fail_frac(g, scheme, force, u0, c, steps, ps, pf, fail_amp)  # noqa: E731
    cs = np.geomspace(c_lo, c_hi, n_scan)
    fl = np.array([ff(c) for c in cs])
    res = dict(scan_c=cs, scan_fail=fl)
    bad = np.flatnonzero(fl > 0)
    if len(bad) == 0:
        return dict(res, c_crit=float("inf"), status="stable_to_hi", non_monotone=False)
    i = bad[0]
    if i == 0:
        return dict(res, c_crit=float("nan"), status="unstable_at_lo", non_monotone=False)
    lo, hi = cs[i - 1], cs[i]
    while hi / lo > 1 + rtol:
        mid = math.sqrt(lo * hi)
        if ff(mid) > 0:
            hi = mid
        else:
            lo = mid
    return dict(res, c_crit=float(math.sqrt(lo * hi)), status="ok", non_monotone=bool((fl[i:] == 0).any()))


# --------------------------------------------------------------------------------------------------
# partition readouts
# --------------------------------------------------------------------------------------------------
def edge_score(X, g, fail_step=None, frac=0.5, lowpass_fc=None, per_run=False, chunk=64):
    """Edge score = RMS over the last `frac` of the trajectory of u_i - u_j, median over non-failed realisations
    (rx's procedure, but with the window taken from X's own length: rx.edge_scores hard-codes T//2 with T=500).
    lowpass_fc: apply rx.lowpass(X, fc) first.  Returns (m,) (or (R_ok,m) if per_run); None if every run failed."""
    ok = _ok(fail_step, X.shape[1])
    if not ok.any():
        return None
    Xo = X[:, ok]
    if lowpass_fc is not None:
        Xo = rx.lowpass(Xo, lowpass_fc)
    s0 = int(Xo.shape[0] * (1 - frac))
    seg = Xo[s0:]
    acc = np.zeros((seg.shape[1], len(g.ei)))
    for a in range(0, seg.shape[0], chunk):
        b = seg[a:a + chunk]
        acc += ((b[:, :, g.ei] - b[:, :, g.ej]) ** 2).sum(0)
    sc = np.sqrt(acc / seg.shape[0])
    return sc if per_run else np.median(sc, axis=0)


def _split_labels(F, q, seed=0):
    from sklearn.cluster import KMeans
    if F.shape[1] == 1 and q == 2:
        return rx.split_1d(F[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KMeans(n_clusters=q, n_init=10, random_state=seed).fit(F).labels_


def _njw(Aff, q):
    d = Aff.sum(1)
    S = Aff / np.sqrt(np.outer(d, d))
    w, V = np.linalg.eigh(S)
    E = V[:, -q:]
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return _split_labels(E, q)


def _cluster_avg(Xo, g, q, frac):
    s0 = int(Xo.shape[0] * (1 - frac))
    A = Xo[s0:]
    A = A - A.mean(2, keepdims=True)
    F = A.mean(0).T                                                  # (n, R_ok)
    nrm = np.linalg.norm(F, axis=0, keepdims=True)
    return _split_labels(F / np.maximum(nrm, 1e-300), q)


def _cluster_corr(Xo, g, q, frac):
    s0 = int(Xo.shape[0] * (1 - frac))
    C = np.zeros((g.n, g.n))
    for r in range(Xo.shape[1]):
        x = Xo[s0:, r, :]
        x = x - x.mean(1, keepdims=True)
        x = x - x.mean(0)
        nr = np.linalg.norm(x, axis=0)
        x = x / np.where(nr > 1e-300, nr, np.inf)
        C += x.T @ x
    C /= Xo.shape[1]
    return _njw((C + 1) / 2, q)


def _cluster_fft(Xo, g, q):
    Z = Xo - Xo.mean(0)                                              # time mean per node (as rx.wave_fft_labels)
    Fz = np.fft.rfft(Z, axis=0)                                      # (nf, R, n)
    P = (np.abs(Fz) ** 2).sum((1, 2))
    P[:2] = 0
    peaks = [k for k in range(2, len(P) - 1) if P[k] >= P[k - 1] and P[k] >= P[k + 1] and P[k] > 0.1 * P.max()]
    k = peaks[0] if peaks else int(np.argmax(P))
    cols = np.stack([rx._align(Fz[k, r]) for r in range(Fz.shape[1])], 1)
    nrm = np.linalg.norm(cols, axis=0, keepdims=True)
    return _split_labels(cols / np.maximum(nrm, 1e-300), q)


def partitions(X, g, q=2, fail_step=None, which=None, fc=0.02, frac=0.5):
    """dict readout-name -> labels (n,) int, or None where a readout cannot produce q parts / all runs failed.
    Uses only non-failed realisations, pooled.  Readouts (PARTITION_NAMES):
      edge_cut            rx procedure: edge score = median_r RMS_t(u_i-u_j) over the last `frac`, single-linkage cut
      edge_cut_lp         same after rx.lowpass(X, fc=0.02) of the whole trajectory
      timeavg_2means      k-means(q) on node features = time-averaged AC state over the last `frac` (one column per
                          realisation, column-normalised)
      traj_corr_spectral  NJW spectral clustering of the node-trajectory correlation matrix (AC state, last `frac`,
                          averaged over realisations), affinity (corr+1)/2
      fft_lowmode         Sahai-style: lowest dominant temporal frequency (pooled power), phase-aligned (SVD) real
                          component per realisation, 2-means (= rx.wave_fft_labels for one realisation, q=2)
    which: subset of names."""
    which = which or PARTITION_NAMES
    ok = _ok(fail_step, X.shape[1])
    out = {nm: None for nm in which}
    if not ok.any():
        return out
    Xo = X[:, ok]
    for nm in which:
        try:
            if nm == "edge_cut":
                out[nm] = rx.cut_partition(g, edge_score(X, g, fail_step, frac), q)
            elif nm == "edge_cut_lp":
                out[nm] = rx.cut_partition(g, edge_score(X, g, fail_step, frac, lowpass_fc=fc), q)
            elif nm == "timeavg_2means":
                out[nm] = _cluster_avg(Xo, g, q, frac)
            elif nm == "traj_corr_spectral":
                out[nm] = _cluster_corr(Xo, g, q, frac)
            elif nm == "fft_lowmode":
                out[nm] = _cluster_fft(Xo, g, q)
            else:
                raise KeyError(nm)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            out[nm] = None
    return out


def partition_metrics(g, labels, truth=None):
    """dict(ari, nmi, cond, minfrac, ok, n_parts) (rx.make_metrics + n_parts); NaNs and ok=0 if labels is None."""
    m = rx.make_metrics(g, None if labels is None else np.asarray(labels), truth)
    m["n_parts"] = 0 if labels is None else int(len(np.unique(labels)))
    return m


def edge_margin(score, g, truth):
    """median(score over true bridge edges) / median(score over intra-block edges).  >1: bridges stand out.
    NaN without truth, bridges, or if the intra median is 0."""
    bm = bridge_mask(g, truth)
    if bm is None or score is None or not bm.any() or bm.all():
        return float("nan")
    return _rel(np.median(score[bm]), np.median(score[~bm]), tiny=1e-300)


def bridge_auc(score, bmask):
    """AUC of the edge score for classifying true bridge edges (ties count 1/2).  NaN if one class is empty."""
    if score is None or bmask is None or bmask.all() or not bmask.any():
        return float("nan")
    return float(roc_auc_score(bmask.astype(int), score))


def evaluate_partitions(X, g, truth, fail_step=None, q=None, which=None, fc=0.02, frac=0.5):
    """One flat dict for a CSV row: n_ok, then for each readout ari_<name>, nmi_<name>, cond_<name>,
    minfrac_<name>, ok_<name>; plus edge_margin and bridge_auc (raw edge_score) and the same two for the
    low-passed score.  q default = number of true blocks (2 if no truth)."""
    q = q or (2 if truth is None else int(len(np.unique(truth))))
    labs = partitions(X, g, q, fail_step, which, fc, frac)
    row = dict(n_ok=int(_ok(fail_step, X.shape[1]).sum()))
    for nm, lab in labs.items():
        m = partition_metrics(g, lab, truth)
        for k in ("ari", "nmi", "cond", "minfrac", "ok"):
            row[f"{k}_{nm}"] = m[k]
    bm = bridge_mask(g, truth)
    for tag, kw in (("", {}), ("_lp", dict(lowpass_fc=fc))):
        sc = edge_score(X, g, fail_step, frac, **kw)
        row["edge_margin" + tag] = edge_margin(sc, g, truth)
        row["bridge_auc" + tag] = bridge_auc(sc, bm)
    return row


def ari_over_time(X, g, truth, window=100, stride=None, q=2, fail_step=None):
    """ARI of the edge_cut readout computed on sliding windows [s, s+window) of the trajectory (edge score = median
    over realisations that are still valid at the window end of the RMS edge difference in the window).
    Returns dict(t_start, t_end, ari): (n_windows,) arrays; ari NaN if no valid realisation or cut fails."""
    T1, R, _ = X.shape
    stride = stride or max(1, window // 2)
    fs = np.full(R, -1) if fail_step is None else np.asarray(fail_step)
    starts = np.arange(0, T1 - window + 1, stride)
    ari = np.full(len(starts), np.nan)
    for i, s in enumerate(starts):
        e = s + window
        ok = (fs < 0) | (fs >= e)
        if not ok.any():
            continue
        seg = X[s:e][:, ok]
        d = seg[:, :, g.ei] - seg[:, :, g.ej]
        sc = np.median(np.sqrt((d ** 2).mean(0)), axis=0)
        lab = rx.cut_partition(g, sc, q)
        if lab is not None and truth is not None:
            ari[i] = adjusted_rand_score(truth, lab)
    return dict(t_start=starts, t_end=starts + window, ari=ari)


def mst_edge_scores(g, score):
    """Edge scores of the minimum spanning forest (Kruskal on `score`, ascending) sorted DESCENDING.
    Removing the top j of them gives j+1 single-linkage clusters (rx.cut_partition)."""
    order = np.argsort(score, kind="stable")
    parent = list(range(g.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    kept = []
    for e in order:
        a, b = find(g.ei[e]), find(g.ej[e])
        if a != b:
            parent[a] = b
            kept.append(score[e])
    return np.sort(np.array(kept))[::-1]


def q_select(score, g, qmax=8, min_rel_gap=None):
    """Number of clusters by the largest gap: with MST scores s_1>=s_2>=... (mst_edge_scores), q = j+1 where
    j = argmax_{1<=j<qmax} (s_j - s_{j+1}) (absolute gap).  If min_rel_gap is given and the largest gap
    < min_rel_gap * median(s), returns q=1 ('no structure').  Returns dict(q, gaps (qmax-1,), top (qmax,))."""
    s = mst_edge_scores(g, score)
    top = s[:qmax]
    gaps = top[:-1] - top[1:]
    j = int(np.argmax(gaps)) + 1
    q = j + 1
    if min_rel_gap is not None and gaps[j - 1] < min_rel_gap * np.median(s):
        q = 1
    return dict(q=q, gaps=gaps, top=top)


# --------------------------------------------------------------------------------------------------
# quasi-equilibrium
# --------------------------------------------------------------------------------------------------
def window_stats(X, g, labels, W=50, stride=25, norm="rms", chunk=64):
    """rx.window_stats with a free window and chunked (memory-safe) edge sums.  X must contain only valid
    realisations.  Returns (din, dout, rms): (R, n_windows), din = max over parts of the RMS intra-part edge
    difference / norm, dout = RMS inter-part edge difference / norm, rms = window RMS of the state (norm='rms':
    includes DC, like rx; 'ac': node-mean removed).  Identical to rx at the defaults."""
    labels = np.asarray(labels)
    T1, R, n = X.shape
    li, lj = labels[g.ei], labels[g.ej]
    parts = np.unique(labels)
    classes = [(li == lj) & (li == p) for p in parts] + [li != lj]
    M = np.stack([c.astype(float) for c in classes], 1)               # (m, P+1)
    cnt = np.maximum(M.sum(0), 1)
    S = np.zeros((T1, R, M.shape[1]))
    for a in range(0, T1, chunk):
        b = X[a:a + chunk]
        S[a:a + chunk] = ((b[:, :, g.ei] - b[:, :, g.ej]) ** 2) @ M
    cs = np.concatenate([np.zeros_like(S[:1]), np.cumsum(S, 0)], 0)
    Y = X if norm == "rms" else X - X.mean(2, keepdims=True)
    ns = np.concatenate([np.zeros((1, R)), np.cumsum((Y ** 2).sum(-1), 0)], 0)
    starts = np.arange(0, T1 - W + 1, stride)
    win = (cs[starts + W] - cs[starts]) / cnt                          # mean over edges of windowed sums
    rms = np.sqrt((ns[starts + W] - ns[starts]) / (W * n)) + 1e-12
    din = np.zeros((len(starts), R))
    for pi, p in enumerate(parts):
        if classes[pi].any():
            din = np.maximum(din, np.sqrt(win[:, :, pi] / W) / rms)
    dout = np.sqrt(win[:, :, -1] / W) / rms if classes[-1].any() else np.zeros_like(din)
    return din.T, dout.T, rms.T


def hold_steps(stats, eps, delta, rho, stride=25, floor=1e-3):
    """rx.hold_steps with a free stride: longest run of windows with D_in<=eps, D_out>=delta, D_out/D_in>=rho and
    rms>=floor, in steps ((run+1)*stride; 0 if none).  (R,)"""
    din, dout, rms = stats
    ok = (din <= eps) & (dout >= delta) & (dout / (din + 1e-12) >= rho) & (rms >= floor)
    res = np.zeros(ok.shape[0])
    for r in range(ok.shape[0]):
        best = cur = 0
        for v in ok[r]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        res[r] = (best + 1) * stride if best else 0
    return res


def hold(X, g, labels, params=None, min_frac=0.1, fail_step=None, norm="rms"):
    """Quasi-equilibrium hold time with the min_frac fix.  Criterion (rx): in a window, D_in <= eps, D_out >= delta,
    D_out/D_in >= rho, rms >= floor; hold = longest run of such windows (steps).  FIX: if any part of `labels` holds
    less than min_frac of the nodes (e.g. a single cut-off node, for which D_in is trivially tiny) the labels are
    rejected and hold = 0.  params default HOLD_PARAMS = eps 0.3, delta 0.1, rho 2, W 50, stride 25, floor 1e-3.
    Only non-failed realisations are scored (NaN for failed ones).  labels None / one part -> rejected.
    norm: 'rms' (rx; DC counts in the normaliser, so a drifting mean can flatter D_in) or 'ac'.
    Returns dict(hold (R,), hold_pos (R,) float 0/1 (NaN failed), din, dout, rms: (R,n_windows) NaN rows for failed,
    t_start (n_windows,), rejected (bool), minfrac, params)."""
    p = dict(HOLD_PARAMS)
    p.update(params or {})
    T1, R, _ = X.shape
    ok = _ok(fail_step, R)
    nw = len(np.arange(0, T1 - p["W"] + 1, p["stride"]))
    out = dict(hold=np.full(R, np.nan), hold_pos=np.full(R, np.nan), din=np.full((R, nw), np.nan),
               dout=np.full((R, nw), np.nan), rms=np.full((R, nw), np.nan),
               t_start=np.arange(0, T1 - p["W"] + 1, p["stride"]), rejected=False, minfrac=float("nan"), params=p)
    if labels is None or len(np.unique(labels)) < 2 or not ok.any():
        out["rejected"] = True
        out["hold"][ok] = 0.0
        out["hold_pos"][ok] = 0.0
        return out
    labels = np.asarray(labels)
    out["minfrac"] = float(np.bincount(np.unique(labels, return_inverse=True)[1]).min() / g.n)
    st = window_stats(X[:, ok], g, labels, p["W"], p["stride"], norm)
    out["din"][ok], out["dout"][ok], out["rms"][ok] = st
    if out["minfrac"] < min_frac:
        out["rejected"] = True
        out["hold"][ok] = 0.0
        out["hold_pos"][ok] = 0.0
        return out
    h = hold_steps(st, p["eps"], p["delta"], p["rho"], p["stride"], p["floor"])
    out["hold"][ok] = h
    out["hold_pos"][ok] = (h > 0).astype(float)
    return out


# --------------------------------------------------------------------------------------------------
# stimulus
# --------------------------------------------------------------------------------------------------
def _runX(r):
    if isinstance(r, Run):
        return r.X, r.fail_step
    return r[0], np.asarray(r[1])


def stimulus(g, n0, runner, amps=(0.2, 1, 5), signs=(+1, -1), sigma=0.05, R=8, steps=200, seed=0,
             det_trials=32, far=None, src_pool=None):
    """Stimulus-transmission experiment (rx.run_stimulus generalised).  Near side = nodes < n0 (two-block graphs),
    far side = nodes >= n0 (override with far = bool mask / src_pool = candidate source nodes).
    runner(g, u0 (R,n), steps) -> Run, e.g. make_runner(...) / make_diff_runner(...) / with_outliers(...).
    Response metrics (R paired trials per amp x sign): stimulus run = noise + sign*amp at a random near node,
    reference = the same noise alone; delta = difference; over trials where neither run failed:
      far_peak (max_t max_far |delta|), far_peak_signed (max of sign*delta), near_peak (incl. the source),
      transmit_ratio = far_peak/near_peak, arrival (first t with max_far|delta| > max(3*noise_far_rms,1e-3), mean
      over trials that arrive), arrived_frac, snr = far_peak/noise_far_rms, noise_far_rms (RMS of the noise-only
      run on the far side), noise_gain = noise_far_rms/sigma.
    Detection: det_trials independent stimulus trials and det_trials independent noise-only trials (new noise);
    statistic = max over t and far nodes of |u| (what an observer sees, no reference); det_auc = Mann-Whitney AUC
    P(stat_stim > stat_noise) (+0.5 ties), failed trials dropped (det_n = min #kept).
    Returns list of dict rows, one per (amp, sign); also fail_frac and n_ok (response trials)."""
    n = g.n
    far_m = (np.arange(n) >= n0) if far is None else np.asarray(far, bool)
    near_m = ~far_m
    pool = np.arange(n0) if src_pool is None else np.asarray(src_pool)
    rng = np.random.default_rng([seed, 4242])
    src = rng.choice(pool, R)
    noise = rng.normal(0, sigma, (R, n))
    Xn, fn = _runX(runner(g, noise, steps))
    nz_all = np.sqrt((Xn[:, :, far_m] ** 2).mean((0, 2)))
    nz_n = rng.normal(0, sigma, (det_trials, n))
    nz_s = rng.normal(0, sigma, (det_trials, n))
    src_d = rng.choice(pool, det_trials)
    Xnull, fnull = _runX(runner(g, nz_n, steps))
    stat_null = np.abs(Xnull[:, :, far_m]).max((0, 2))[fnull < 0]
    rows = []
    for a in amps:
        for s in signs:
            u = noise.copy()
            u[np.arange(R), src] += s * a
            Xs, fs = _runX(runner(g, u, steps))
            ok = (fs < 0) & (fn < 0)
            row = dict(amp=float(a), sign=int(s), R=R, n_ok=int(ok.sum()), fail_frac=float(1 - ok.mean()))
            if ok.any():
                dl = (Xs - Xn)[:, ok]
                pk_t = np.abs(dl[:, :, far_m]).max(-1)               # (T1,Rok)
                fpk = pk_t.max(0)
                npk = np.abs(dl[:, :, near_m]).max((0, 2))
                nz = nz_all[ok]
                thr = np.maximum(3 * nz, 1e-3)
                arr = np.array([np.argmax(pk_t[:, r] > thr[r]) if (pk_t[:, r] > thr[r]).any() else np.nan
                                for r in range(pk_t.shape[1])])
                row.update(far_peak=float(fpk.mean()), far_peak_signed=float((s * dl[:, :, far_m]).max((0, 2)).mean()),
                           near_peak=float(npk.mean()), transmit_ratio=float((fpk / np.maximum(npk, 1e-12)).mean()),
                           arrival=float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
                           arrived_frac=float(np.isfinite(arr).mean()),
                           snr=float((fpk / np.maximum(nz, 1e-12)).mean()), noise_far_rms=float(nz.mean()),
                           noise_gain=float((nz / sigma).mean()))
            else:
                row.update({k: float("nan") for k in ("far_peak", "far_peak_signed", "near_peak", "transmit_ratio",
                                                      "arrival", "arrived_frac", "snr", "noise_far_rms", "noise_gain")})
            ud = nz_s.copy()
            ud[np.arange(det_trials), src_d] += s * a
            Xd, fd = _runX(runner(g, ud, steps))
            stat_stim = np.abs(Xd[:, :, far_m]).max((0, 2))[fd < 0]
            if len(stat_stim) >= 2 and len(stat_null) >= 2:
                y = np.r_[np.ones(len(stat_stim)), np.zeros(len(stat_null))]
                row["det_auc"] = float(roc_auc_score(y, np.r_[stat_stim, stat_null]))
            else:
                row["det_auc"] = float("nan")
            row["det_n"] = int(min(len(stat_stim), len(stat_null)))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
# confounders
# --------------------------------------------------------------------------------------------------
@dataclass
class PerturbedRunner:
    """Runner wrapper that perturbs the state after every update (before the failure test).
    mode 'stuck'  : nodes `idx` are held at `value` ('init' = their initial value, or a float) every step
    mode 'noisy'  : nodes `idx` get additive heavy-tailed noise (Student-t3, unit variance) * scale every step
    mode 'bridge' : nodes incident to the edges in bridge_mask get additive gaussian noise * scale every step
    idx is a random frac*n subset drawn from default_rng(seed) (same subset for every call -> paired stimulus runs
    see the same outliers).  The per-step noise stream is seeded by (seed, call counter): successive calls
    (e.g. noise-only vs stimulus run) get different noise."""
    inner: Runner
    mode: str = "stuck"
    frac: float = 0.05
    seed: int = 0
    scale: float = 1.0
    value: object = "init"
    bridge_mask: object = None
    _calls: int = 0

    def __call__(self, g, u0, steps=None):
        u0 = _as2d(u0)
        Rn, n = u0.shape
        if self.mode == "bridge":
            if self.bridge_mask is None or len(self.bridge_mask) != len(g.ei):
                raise ValueError("bridge mode needs bridge_mask matching g")
            idx = np.unique(np.r_[g.ei[self.bridge_mask], g.ej[self.bridge_mask]])
        else:
            k = max(1, int(round(self.frac * n)))
            idx = np.sort(np.random.default_rng(self.seed).choice(n, k, replace=False))
        nrng = np.random.default_rng([self.seed, self._calls, 99])
        self._calls += 1
        vals = u0[:, idx].copy() if self.value == "init" else np.full((Rn, len(idx)), float(self.value or 0.0))
        mode, scale = self.mode, self.scale

        def post(t, cur, nxt, ctx):
            if mode == "stuck":
                nxt[:, idx] = vals
            elif mode == "noisy":
                nxt[:, idx] += scale * nrng.standard_t(3, (Rn, len(idx))) / math.sqrt(3.0)
            else:
                nxt[:, idx] += scale * nrng.normal(0, 1, (Rn, len(idx)))

        r = self.inner(g, u0, steps, post=post)
        r.outlier_idx = idx
        return r

    @property
    def label(self):
        return f"{self.inner.label}|{self.mode}({self.frac:g},{self.scale:g})"


def with_outliers(runner, frac=0.05, mode="stuck", seed=0, scale=1.0, value="init"):
    """Wrap a Runner so that a random `frac` of nodes are outliers: mode='stuck' (frozen at `value`, default their
    initial value) or 'noisy' (per-step Student-t3 noise of std `scale`).  See PerturbedRunner."""
    if mode not in ("stuck", "noisy"):
        raise ValueError("mode must be 'stuck' or 'noisy'")
    return PerturbedRunner(runner, mode, frac, seed, scale, value)


def bridge_noise(runner, bridge_mask, sigma=0.05, seed=0):
    """Wrap a Runner so that the nodes incident to true bridge edges (bridge_mask: bool (m,), e.g.
    meta['bridge_mask']) receive additive gaussian noise of std sigma at every step."""
    return PerturbedRunner(runner, "bridge", 0.0, seed, sigma, "init", np.asarray(bridge_mask, bool))


# --------------------------------------------------------------------------------------------------
# stats / IO
# --------------------------------------------------------------------------------------------------
def boot_ci(x, n_boot=2000, alpha=0.05, seed=0, func=np.mean):
    """(estimate, lo, hi): percentile bootstrap CI of func over the finite entries of x (resampling units =
    graphs/seeds: pass ONE value per graph).  (nan, nan, nan) if empty, CI nan if n < 2."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    est = float(func(x))
    if len(x) < 2:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    b = func(x[idx], axis=1) if func in (np.mean, np.median) else np.array([func(r) for r in x[idx]])
    return est, float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def paired_wilcoxon(a, b, alternative="two-sided"):
    """Paired Wilcoxon signed-rank on a-b after dropping pairs with a NaN.  Returns dict(n, n_nonzero, mean_diff,
    median_diff, stat, p).  All-zero differences -> p = 1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[m]
    out = dict(n=int(m.sum()), n_nonzero=int((d != 0).sum()), mean_diff=float(d.mean()) if len(d) else float("nan"),
               median_diff=float(np.median(d)) if len(d) else float("nan"), stat=float("nan"), p=float("nan"))
    if len(d) == 0:
        return out
    if (d != 0).sum() == 0:
        out["p"] = 1.0
        return out
    r = _st.wilcoxon(d, zero_method="wilcox", alternative=alternative)
    out["stat"], out["p"] = float(r.statistic), float(r.pvalue)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone step-down, capped at 1; NaN entries ignored and kept NaN)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def paired_vs_baseline(df, value, config, baseline, pair=("graph", "seed"), n_boot=2000, alpha=0.05):
    """Each config vs `baseline` config on metric `value`, paired by the columns `pair` (pandas DataFrame in long
    format: one row per pair x config).  Returns DataFrame: config, n, mean_diff, ci_lo, ci_hi (bootstrap over pairs),
    median_diff, p (Wilcoxon), p_holm (Holm over all compared configs)."""
    import pandas as pd
    base = df[df[config] == baseline].set_index(list(pair))[value]
    rows = []
    for cfg, sub in df[df[config] != baseline].groupby(config, sort=False):
        s = sub.set_index(list(pair))[value]
        j = s.to_frame("a").join(base.rename("b"), how="inner")
        w = paired_wilcoxon(j["a"].values, j["b"].values)
        est, lo, hi = boot_ci((j["a"] - j["b"]).values, n_boot, alpha)
        rows.append(dict(config=cfg, n=w["n"], mean_diff=est, ci_lo=lo, ci_hi=hi, median_diff=w["median_diff"],
                         p=w["p"]))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res["p"].values)
    return res


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def write_csv(path, rows):
    """rows: list of dicts (columns = union of keys in first-seen order) or a pandas DataFrame.  Creates parent
    directories, converts numpy scalars, returns the Path.  Raw rows please (one per graph x config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _py(v) for k, v in r.items()})
    return path


def pmap(fn, jobs, procs=2, chunksize=1, progress=False, start="spawn"):
    """Ordered parallel map with a hard cap of MAX_PROCS (=2, env BATTERY_MAX_PROCS) worker processes.
    fn must be a picklable top-level function and the calling script must be guarded by
    `if __name__ == "__main__":` (spawn re-imports it).  procs<=1 or a single job runs serially in-process.
    Exceptions propagate.  Workers import common themselves (BLAS threads pinned to 1)."""
    jobs = list(jobs)
    procs = max(1, min(int(procs), MAX_PROCS))
    if procs == 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    t0 = time.time()
    out = []
    with mp.get_context(start).Pool(procs) as pool:
        for i, r in enumerate(pool.imap(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if progress and (i % max(1, len(jobs) // 20) == 0 or i == len(jobs)):
                print(f"  pmap {i}/{len(jobs)} [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    return out
