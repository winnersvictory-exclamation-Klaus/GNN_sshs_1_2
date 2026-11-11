"""Shared helpers for idea 4 (high-order even series).  Import this FIRST in every t4xx script (it imports common,
registers the extra forces / schemes with wave_test and defines the theory helpers).

Registered here (all re-registered at import in spawn workers):
  force  series:<base>   pf: w=(a_1..a_K)   total force = sum_k a_k (-1)^{k+1} F^{k}   (generalised weight vector)
  force  gk:<base>       pf: k=int          pure G_k = (-1)^{k+1} F^{k}  (for restoring-ness probes)
  scheme t6g             ps: eps=(e4,e6,...) any order, same code path as wt.time_even_step
  scheme t6su            ps: e4, e6, m   plain-wave start-up for the first m steps, true history afterwards
  scheme t6su_sym        ps: e4, e6      history u(-j) filled by backward-in-time leapfrog (u(-1)=u(1)=u0+c^2F/2 ...)
"""
import sys
from pathlib import Path
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402  (pins BLAS threads)
import numpy as np  # noqa: E402

wt = C.wt
rx = C.rx
HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# ---------------------------------------------------------------------------------------------------
# registered extras
# ---------------------------------------------------------------------------------------------------


def _mk_series(kind):
    def fn(ctx, u):
        return wt.series_force(ctx.g, u, kind, ctx.pf["w"])
    return fn


def _mk_gk(kind):
    def fn(ctx, u):
        k = int(ctx.pf["k"])
        G = u
        for _ in range(k):
            G = wt.base_force(ctx.g, G, kind)
        return (-1) ** (k + 1) * G
    return fn


for _b in wt.BASES:
    if f"series:{_b}" not in wt.FORCES:
        wt.force(f"series:{_b}", w=(1.0,))(_mk_series(_b))
        wt.force(f"gk:{_b}", k=1)(_mk_gk(_b))


@wt.scheme("t6g", eps=(0.1, 0.01))
def _t6g(ctx, cur, prev, F, Fp):
    return wt.time_even_step(ctx, cur, prev, F, list(ctx.ps["eps"]))


@wt.scheme("t6su", e4=0.1, e6=0.01, m=6)
def _t6su(ctx, cur, prev, F, Fp):
    """Start-up: plain wave for the first max(m, nh) steps (recording the true history), then the series scheme
    with that true history instead of u(-j)=u0."""
    ev = [ctx.ps["e4"], ctx.ps["e6"]]
    nh = 2 * (len(ev) + 1)
    hist = ctx.state.setdefault("su_hist", [])
    hist.insert(0, cur)
    del hist[nh + 1:]
    if ctx.t < max(int(ctx.ps["m"]), nh):
        return 2 * cur - prev + ctx.c ** 2 * F
    if not ctx.state.get("su_on"):
        ctx.state["s_hist"] = list(hist[1:nh + 1])        # [u_{t-1}, ..., u_{t-nh}]
        ctx.state["su_on"] = True
    return wt.time_even_step(ctx, cur, prev, F, ev)


@wt.scheme("t6su_sym", e4=0.1, e6=0.01, force="mean")
def _t6su_sym(ctx, cur, prev, F, Fp):
    """Start-up by time symmetry (zero initial velocity => u(-j)=u(j)): pre-run the plain wave with the
    second-order start u(1)=u0+c^2F(u0)/2 for nh steps, mirror it to negative times as the history."""
    ev = [ctx.ps["e4"], ctx.ps["e6"]]
    nh = 2 * (len(ev) + 1)
    if "s_hist" not in ctx.state:
        ffn = wt.FORCES[ctx.ps["force"]][0]
        c2 = ctx.c ** 2
        pp, cc = cur, cur + 0.5 * c2 * ffn(ctx, cur)
        fut = [cc]
        for _ in range(nh - 1):
            pp, cc = cc, 2 * cc - pp + c2 * ffn(ctx, cc)
            fut.append(cc)
        ctx.state["s_hist"] = fut                         # [u(1), ..., u(nh)] = [u(-1), ..., u(-nh)]
    return wt.time_even_step(ctx, cur, prev, F, ev)


# ---------------------------------------------------------------------------------------------------
# series helpers
# ---------------------------------------------------------------------------------------------------
NAMED = {
    "hi_acc": (1, 1 / 12, 1 / 90, 1 / 560),
    "hi_alt": (1, -1 / 12, 1 / 90, -1 / 560),
}


def weights(name, eps=None, K=4):
    """weight vector of a registered series name (hi_acc, hi_alt, hi_geo, hi_geoneg)."""
    if name in NAMED:
        return tuple(NAMED[name])
    e = eps if eps is not None else (0.2 if name == "hi_geo" else -0.2)
    return tuple(e ** k for k in range(K))


def g_of(lam, w):
    lam = np.asarray(lam, float)
    return sum(a * lam ** k for k, a in enumerate(w, 1))


def spec_lam(g):
    lam, Phi, d = C.spectrum(g)
    return lam, Phi, d


def c_crit_pred(lam, w):
    """2/sqrt(max g) if g>0 on the (nonzero) spectrum else 0 (some mode unstable for any c)."""
    gv = g_of(lam[1:], w)
    if (gv < -1e-12).any():
        return 0.0          # a mode with negative stiffness: exponential growth at any c
    return 2.0 / np.sqrt(max(gv.max(), 1e-300))   # g==0 modes are neutral (z=1 double root, no growth from zero velocity)


def modal_rho_wave(s):
    """roots of z^2-(2-s)z+1 for s=c^2 g; returns max|z| (1 if 0<=s<=4)."""
    s = np.asarray(s, float)
    disc = (2 - s) ** 2 - 4 + 0j
    z1 = ((2 - s) + np.sqrt(disc)) / 2
    z2 = ((2 - s) - np.sqrt(disc)) / 2
    return np.maximum(abs(z1), abs(z2))


# ---------------------------------------------------------------------------------------------------
# time-scheme characteristic polynomial, derived from wt.time_even_step code as written
#   lead*u_{t+1} + [-2 u_t + u_{t-1}] + sum_k e_k sum_{j=1}^{2k} (-1)^j C(2k,j) u_{t+1-j} = c^2 F(u_t) = -s u_t
#   with u_{t+1-j} <-> z^{N-j}, N = 2(m+1), m=len(eps):   P(z) = sum_j b_j z^{N-j} + s z^{N-1}
# ---------------------------------------------------------------------------------------------------

def time_poly(eps, s):
    """coefficients (highest degree first) of the characteristic polynomial for one mode with s=c^2*lambda."""
    m = len(eps)
    N = 2 * (m + 1)
    b = np.zeros(N + 1)
    b[0] = 1.0 + sum(eps)
    b[1] += -2.0
    b[2] += 1.0
    for k, e in enumerate(eps, 2):
        for j in range(1, 2 * k + 1):
            b[j] += e * (-1) ** j * comb(2 * k, j)
    poly = b.copy()                      # coefficient of z^{N-j} is b[j]
    poly[1] += s                         # + s z^{N-1}
    return poly


def time_roots(eps, s):
    return np.roots(time_poly(eps, s))


def time_rho(eps, s):
    """max |z| over roots (array in s)."""
    s = np.atleast_1d(np.asarray(s, float))
    out = np.empty(len(s))
    for i, si in enumerate(s):
        out[i] = np.abs(time_roots(eps, si)).max()
    return out


def time_phys_root(eps, s):
    """root continuously connected to e^{i sqrt(s)}: the one closest to it (physical branch)."""
    r = time_roots(eps, s)
    tgt = np.exp(1j * np.sqrt(s)) if s > 0 else 1.0
    return r[np.argmin(abs(r - tgt))]


def ari_or_nan(x):
    return float(x) if x is not None else np.nan


# ---------------------------------------------------------------------------------------------------
# steps-to-partition helpers
# ---------------------------------------------------------------------------------------------------
READOUTS = ("fft_lowmode", "edge_cut_lp", "edge_cut", "timeavg_2means")


def prefix_ari(run, g, truth, Ts, which=READOUTS):
    """ARI of each readout computed on the trajectory prefix X[:T+1] for T in Ts. -> {readout: array(len(Ts))}"""
    out = {w: np.full(len(Ts), np.nan) for w in which}
    for i, T in enumerate(Ts):
        if T > run.X.shape[0] - 1:
            continue
        X = run.X[:T + 1]
        fs = np.where((run.fail_step >= 0) & (run.fail_step <= T), run.fail_step, -1)
        labs = C.partitions(X, g, 2, fs, which=which)
        for w in which:
            m = C.partition_metrics(g, labs[w], truth)
            out[w][i] = m["ari"] if labs[w] is not None else np.nan
    return out


def first_reach(Ts, ari, thr=0.9):
    """(first T with ARI>=thr, first T from which ARI>=thr at every later grid point); NaN if never."""
    ok = np.asarray(ari) >= thr
    first = next((T for T, o in zip(Ts, ok) if o), np.nan)
    sust = np.nan
    for i in range(len(Ts)):
        if ok[i:].all():
            sust = Ts[i]
            break
    return first, sust


# ---------------------------------------------------------------------------------------------------
# generic configuration evaluation (T4.11 / T4.13 / T4.15 / T4.16)
# cfg = dict(label, kind='scheme'|'diff', scheme, force, ps, pf, c (float | ('cp', frac)), w (weights for c_pred), K (aggregations per step))
# ---------------------------------------------------------------------------------------------------

def resolve_c(cfg, lam):
    c = cfg.get("c", 0.5)
    if isinstance(c, tuple):
        return c[1] * c_crit_pred(lam, cfg["w"])
    return float(c)


def make_cfg_runner(cfg, lam, steps):
    if cfg.get("kind") == "diff":
        return C.make_diff_runner(cfg.get("force", "mean"), eta=cfg.get("eta", 0.5), pf=cfg.get("pf"), steps=steps)
    return C.make_runner(cfg.get("scheme", "wave"), cfg["force"], c=resolve_c(cfg, lam), ps=cfg.get("ps"), pf=cfg.get("pf"), steps=steps)


def eval_cfg(g, truth, meta, lam, u0, cfg, steps=500, stim=False, n0=24, dead_ok=True):
    runner = make_cfg_runner(cfg, lam, steps)
    run = runner(g, u0)
    row = dict(graph=meta["id"], seed=meta["seed"], config=cfg["label"], c=getattr(run, "c", np.nan), K=cfg.get("K", 1),
               n=meta["n"], bridges=meta["n_bridges"], block=meta["params"].get("block"))
    row.update(C.traj_summary(run))
    if run.ok.any():
        if truth is not None:
            row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))
        labs = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut", "edge_cut_lp"))
        for nm, lab in labs.items():
            if lab is None:
                row[f"hold_{nm}"] = np.nan
                row[f"degen_{nm}"] = np.nan
                continue
            h = C.hold(run.X, g, lab, fail_step=run.fail_step)
            row[f"hold_{nm}"] = float(np.nanmean(h["hold"]))
            row[f"false_{nm}"] = float(np.nanmean(h["hold"]) > 0)       # a part >= min_frac AND quasi-equilibrium held
            row[f"minfrac_{nm}"] = float(C.partition_metrics(g, lab, truth)["minfrac"]) if truth is not None else float(min(np.bincount(lab)) / len(lab))
    if stim and truth is not None:
        st = C.stimulus(g, n0, runner, R=8)
        df = [dict(s) for s in st]
        for key in ("far_peak", "transmit_ratio", "arrival", "snr", "noise_gain", "det_auc"):
            vals = [d[key] for d in df if d["amp"] == 1]
            row[f"stim_{key}"] = float(np.nanmean(vals))
    return row
