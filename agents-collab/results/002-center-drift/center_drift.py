#!/usr/bin/env python3
"""002-center-drift: plan section 3 E stability diagnostic for legacy `huber_local` wave (development only).

Part 1  legacy replay (zeroing, archived t310 RNG) on the 48 t310 graphs + a non-zeroed diagnostic replay that
        tracks the degree-weighted center m, its acceleration and the centered amplitudes.
Part 2  12-graph slice: undamped central differences at h=0.5 / 0.25 to T=250, zero-velocity Taylor startup,
        original vs degree-weighted-mean-projected huber_local.
Part 3  same slice: edge-symmetric local Huber candidate, Mean / fixed Huber references, single-mode frequency
        and first-bridge-force hand checks, OWA / odd_owa (original + projected) controls on two graphs.
Part 4  matched raw / low-pass latter-half edge-RMS q=2 cut readouts on the slice (descriptive + plan scores).

Run from anywhere; see README.md for the exact command and environment.  Writes CSVs, gzipped trace CSVs and
summary.json into --out (default: this directory).  Exits 1 if a hard validation check fails.
"""
import sys

sys.dont_write_bytecode = True  # do not leave __pycache__ in the shared reference-code directories

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.environ.setdefault("GNN_REPO", str(REPO))          # wave_test.py locates rank_experiment.py through this
sys.path.insert(0, str(REPO / "GNN_testing" / "battery"))
import common as C  # noqa: E402  (imports wave_test as C.wt and rank_experiment as C.rx)
import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

wt, rx = C.wt, C.rx
ARCHIVE = REPO / "GNN_testing" / "battery" / "idea3" / "results" / "t310_partition.csv"

# ---- fixed settings (all chosen before running; none are tuned) ------------------------------------------
AMP = 20.0             # legacy failure predicate: not all(|u| < AMP) on an updated frame (nonfinite fails too)
EMERGENCY = 1e8        # diagnostic-only safety censor: stop integrating an init once max|u| exceeds this
C_LEGACY = 0.5         # legacy wave speed; u+ = 2u - u- + c^2 F is central differences with h = c
LEGACY_STEPS = 500
T_PHYS = 250.0         # = LEGACY_STEPS * C_LEGACY
HS = (0.5, 0.25)
R = 8
FC = 0.02              # low-pass cutoff, cycles per saved frame (rx.lowpass, as archived)
FRAC = 0.5             # latter-half scoring window (frames 250..500 of 501)
REL_GROWTH = 2.0       # relative-growth evidence: peak centered RMS >= 2 x initial centered RMS
NOISE_FLOOR = 1e-6     # bridge and internal median edge RMS both below this -> noise-floor readout
NONINFORMATIVE_ARI = 0.5
SENS_DARI = 0.1        # |ARI(h=.5) - ARI(h=.25)| above this flags discretization sensitivity
CHECK_EVERY = 5.0      # physical time between saved checkpoints (10 legacy frames)
NEAR = 10              # substeps saved either side of a first crossing
SENS_TIMES = (25.0, 50.0, 125.0, 250.0)
TOL_CONSERVE = 1e-12   # |sum d F| / sum d |F|
TOL_CENTER_EQ = 1e-10  # discrete center recursion residual / max(1, max|u|)
TOL_CENTER_KEEP = 1e-10  # |m(t) - m(0)| / max(1, max|u|) for exactly mean-free forces
TOL_RECENTER = 1e-8    # |u_proj - (u_orig - m_orig + m0)| / max(1, max|u_orig|), full horizon (reported, not fatal)
TOL_RECENTER_EARLY = 1e-12  # same deviation over physical t <= RECENTER_EARLY_T
RECENTER_EARLY_T = 50.0
TWIN_FACTOR = 10.0     # full-horizon projection deviation <= this x the 1-ulp-perturbed-start deviation
TOL_FREQ = 1e-9
TOL_HAND = 1e-15

MAIN_CONFIGS = [  # (force, variant, role)
    ("huber_local", "orig", "main"),
    ("huber_local", "proj", "projection_control"),
    ("huber_local_sym", "orig", "candidate"),
    ("mean", "orig", "reference"),
    ("huber", "orig", "reference"),
]
OWA_CONFIGS = [(f, v, "E_control") for f in ("owa", "odd_owa") for v in ("orig", "proj")]
OWA_GRAPHS = ("two_block:cycle_24x24_bridges1_wb1_s0", "two_block:clique_24x24_bridges1_wb1_s0")


# ---- graphs, initial states, forces --------------------------------------------------------------------
def t310_specs():
    """The 48 archived t310 graphs, exactly as t310_partition.py builds them."""
    return C.zoo("core", seeds=C.DEV_SEEDS[:4])


def slice_specs():
    """Preselected 12-graph slice: 24+24, cycle/clique/SBM, unit bridge weight, 1 or 4 bridges, seeds 0/1."""
    return [s for s in t310_specs() if s["bridges"] in (1, 4) and s["seed"] in (0, 1)]


def t310_init(spec, g):
    """The archived t310 initial states: eight U(-1,1) vectors from default_rng(seed*1000+310)."""
    return C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 310))


def huber_local_sym(g, kappa):
    """Edge-symmetric local Huber: same endpoint thresholds tau_i = kappa*median_j|u_j-u_i| + EPS as huber_local,
    conductance-weighted flux w/2*(clip(s,+-tau_i) + clip(s,+-tau_j)) with s = u_j-u_i added to i and subtracted
    from j, then divided by weighted degree.  Written in the padded neighbour layout of wave_test."""
    def F(u):
        x = wt.diffs(g, u)                                     # (R, n, dmax): u_j - u_i, padding 0
        tau = kappa * wt.nb_median(g, np.abs(x)) + wt.EPS      # (R, n), the original neighbour rule
        ti, tj = tau[..., None], tau[:, g.NB]
        return wt.edge_mean(g, 0.5 * (np.clip(x, -ti, ti) + np.clip(x, -tj, tj)))
    return F


def make_force(g, name, variant):
    """Callable u (R,n) -> F (R,n).  variant 'proj' removes the degree-weighted mean: F - sum(dF)/sum(d)."""
    if name == "huber_local_sym":
        base = huber_local_sym(g, C.resolve_params(wt.FORCES, "huber_local")["kappa"])
    else:
        ctx = wt.Ctx(g=g, c=float("nan"), pf=C.resolve_params(wt.FORCES, name), ps={}, u0=None)
        fn = wt.FORCES[name][0]
        base = lambda u: fn(ctx, u)  # noqa: E731
    if variant == "orig":
        return base
    d = C.wdeg(g)
    D = d.sum()
    return lambda u: (lambda F: F - (F @ d / D)[:, None])(base(u))


# ---- integration ---------------------------------------------------------------------------------------
def integrate(g, force, u0, h, steps, startup):
    """Undamped central differences u_{k+1} = 2u_k - u_{k-1} + h^2 F(u_k) without zeroing.
    startup 'legacy': prev = cur (so u_1 = u_0 + h^2 F(u_0), bit-identical to wt 'wave' with c = h);
    startup 'taylor': u_1 = u_0 + 0.5 h^2 F(u_0) (zero initial velocity).
    first_abs: first updated frame failing the legacy predicate (nonfinite included).  An init is censored
    (state NaN from then on) at its first nonfinite frame or once max|u| > EMERGENCY.
    A[t] = sum(d F(u_t)) / sum(d); cons[t] = |sum(d F)| / sum(d |F|)."""
    d = C.wdeg(g)
    D = d.sum()
    Rn, n = u0.shape
    X = np.full((steps + 1, Rn, n), np.nan)
    A = np.full((steps + 1, Rn), np.nan)
    cons = np.full((steps + 1, Rn), np.nan)
    X[0] = u0
    first_abs = np.full(Rn, -1)
    nonfinite = np.full(Rn, -1)
    emergency = np.full(Rn, -1)
    alive = np.ones(Rn, bool)
    prev, cur = u0.copy(), u0.copy()
    h2 = h ** 2
    with np.errstate(all="ignore"):
        for t in range(steps + 1):
            F = np.full((Rn, n), np.nan)
            if alive.any():
                F[alive] = force(cur[alive])
            A[t] = F @ d / D
            cons[t] = np.abs(F @ d) / np.maximum(np.abs(F) @ d, 1e-300)
            if t == steps:
                break
            if t == 0 and startup == "taylor":
                nxt = cur + 0.5 * h2 * F
            else:
                nxt = 2 * cur - prev + h2 * F
            bad = ~(np.abs(nxt) < AMP).all(1) & (first_abs < 0) & alive
            first_abs[bad] = t + 1
            nf = alive & ~np.isfinite(nxt).all(1)
            nonfinite[nf] = t + 1
            em = alive & ~nf & (np.abs(nxt).max(1) > EMERGENCY)
            emergency[em] = t + 1
            alive &= ~(nf | em)
            nxt[~alive] = np.nan
            X[t + 1] = nxt
            prev, cur = cur, nxt
    return dict(X=X, A=A, cons=cons, first_abs=first_abs, nonfinite=nonfinite, emergency=emergency, h=h,
                steps=steps, startup=startup)


def center_series(X, d):
    """m (degree-weighted center), max|u|, centered max max|u-m|, degree-weighted centered RMS; each (T1, R)."""
    D = d.sum()
    with np.errstate(all="ignore"):
        m = X @ d / D
        V = X - m[..., None]
        return m, np.abs(X).max(-1), np.abs(V).max(-1), np.sqrt((V ** 2) @ d / D)


def center_eq_residual(run, m):
    """max over inits/frames of the discrete center recursion residual relative to max(1, max|u|):
    m_{k+1} - 2 m_k + m_{k-1} = h^2 A_k (k >= 1) and m_1 - m_0 = (1 or 1/2) h^2 A_0."""
    h2 = run["h"] ** 2
    A = run["A"]
    lhs0 = m[1] - m[0]
    rhs0 = (0.5 if run["startup"] == "taylor" else 1.0) * h2 * A[0]
    res = [np.abs(lhs0 - rhs0)]
    res.append(np.abs((m[2:] - 2 * m[1:-1] + m[:-2]) - h2 * A[1:-1]))
    scale = max(1.0, float(np.nanmax(np.abs(run["X"]))))
    vals = np.concatenate([np.ravel(r) for r in res])
    vals = vals[np.isfinite(vals)]
    return float(vals.max() / scale) if vals.size else float("nan")


def first_index(mask):
    """(T1, R) bool -> first True index per column, -1 if none."""
    any_ = mask.any(0)
    return np.where(any_, mask.argmax(0), -1)


def init_rows(run, g, base):
    """Per-initialization diagnostic dicts (and the centered first-hit array)."""
    d = C.wdeg(g)
    X, A, h = run["X"], run["A"], run["h"]
    m, mx, cmx, crms = center_series(X, d)
    with np.errstate(invalid="ignore"):
        cen_hit = first_index(~(cmx < AMP) & np.isfinite(cmx))
    rows = []
    for r in range(X.shape[1]):
        fa, fc_, nf, em = int(run["first_abs"][r]), int(cen_hit[r]), int(run["nonfinite"][r]), int(run["emergency"][r])
        valid = np.isfinite(mx[:, r])
        last = int(np.flatnonzero(valid).max())
        pk = lambda s: (float(np.nanmax(s[:, r])), int(np.nanargmax(s[:, r])))  # noqa: E731
        p_mx, p_mx_k = pk(mx)
        p_m, p_m_k = pk(np.abs(m))
        p_c, p_c_k = pk(cmx)
        p_rms, p_rms_k = pk(crms)
        ratio = p_rms / crms[0, r]
        censored = nf >= 0 or em >= 0
        grow = ratio >= REL_GROWTH
        if censored:
            cls = "censored"
        elif fc_ >= 0:
            cls = "relative_crossing"
        elif fa >= 0:
            cls = "mixed_center_crossing_relative_growth" if grow else "center_only_crossing"
        else:
            cls = "no_crossing_relative_growth" if grow else "no_crossing"
        at = lambda s, k: float(s[k, r]) if k >= 0 and np.isfinite(s[k, r]) else float("nan")  # noqa: E731
        T = run["steps"]
        rows.append(dict(base, init=r, h=h, startup=run["startup"], steps=T,
                         first_abs_step=fa, first_abs_t=fa * h if fa >= 0 else float("nan"),
                         m_at_abs=at(m, fa), cmax_at_abs=at(cmx, fa), crms_at_abs=at(crms, fa),
                         first_centered_step=fc_, first_centered_t=fc_ * h if fc_ >= 0 else float("nan"),
                         maxabs_at_centered=at(mx, fc_), m_at_centered=at(m, fc_),
                         nonfinite_step=nf, emergency_step=em, censored=int(censored), last_valid_step=last,
                         m0=float(m[0, r]), crms0=float(crms[0, r]),
                         peak_maxabs=p_mx, peak_maxabs_step=p_mx_k, peak_abs_m=p_m, peak_abs_m_step=p_m_k,
                         peak_cmax=p_c, peak_cmax_step=p_c_k, peak_crms=p_rms, peak_crms_step=p_rms_k,
                         crms_ratio=ratio, peak_abs_a=float(np.nanmax(np.abs(A[:, r]))),
                         m_drift_T=at(m, T) - float(m[0, r]), maxabs_T=at(mx, T), cmax_T=at(cmx, T), crms_T=at(crms, T),
                         rel_growth=int(grow), cls=cls))
    return rows, (m, mx, cmx, crms)


def series_rows(run, series, base, first_abs, first_cen):
    """Trace rows at common physical checkpoints (every CHECK_EVERY) and within NEAR substeps of first crossings."""
    m, mx, cmx, crms = series
    A, h = run["A"], run["h"]
    T1 = m.shape[0]
    every = int(round(CHECK_EVERY / h))
    rows = []
    for r in range(m.shape[1]):
        fa, fc_ = int(first_abs[r]), int(first_cen[r])
        ks = set(range(0, T1, every))
        for f in (fa, fc_):
            if f >= 0:
                ks |= set(range(max(0, f - NEAR), min(T1, f + NEAR + 1)))
        for k in sorted(ks):
            rows.append(dict(base, init=r, h=h, step=k, t=k * h, checkpoint=int(k % every == 0),
                             near_abs=int(fa >= 0 and abs(k - fa) <= NEAR),
                             near_centered=int(fc_ >= 0 and abs(k - fc_) <= NEAR),
                             m=float(m[k, r]), a=float(A[k, r]), maxabs=float(mx[k, r]), cmax=float(cmx[k, r]),
                             crms=float(crms[k, r])))
    return rows


# ---- readout -------------------------------------------------------------------------------------------
def readout(Xs, g, truth, use):
    """Raw and low-pass latter-half edge-RMS scores (median over the `use` inits), q=2 single-linkage cut."""
    bm = C.bridge_mask(g, truth)
    out = {}
    for tag, fc in (("raw", None), ("lp", FC)):
        if not use.any():
            out.update({f"{tag}_{k}": float("nan") for k in ("bridge_med", "internal_med", "margin", "auc", "ari")})
            out.update({f"{tag}_cut_ok": 0, f"{tag}_noise_floor": 0})
            continue
        sc = C.edge_score(Xs[:, use], g, None, FRAC, lowpass_fc=fc)
        lab = rx.cut_partition(g, sc, 2)
        b, i = float(np.median(sc[bm])), float(np.median(sc[~bm]))
        out.update({f"{tag}_bridge_med": b, f"{tag}_internal_med": i, f"{tag}_margin": C.edge_margin(sc, g, truth),
                    f"{tag}_auc": C.bridge_auc(sc, bm),
                    f"{tag}_ari": float(adjusted_rand_score(truth, lab)) if lab is not None else float("nan"),
                    f"{tag}_cut_ok": int(lab is not None), f"{tag}_noise_floor": int(max(b, i) < NOISE_FLOOR)})
    return out


# ---- output helpers ------------------------------------------------------------------------------------
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if str(path).endswith(".gz"):   # mtime=0 keeps reruns byte-identical
        opener = lambda p, m, newline: io.TextIOWrapper(gzip.GzipFile(p, "wb", mtime=0), newline=newline)  # noqa: E731
    else:
        opener = open
    with opener(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()})


def read_archive():
    with open(ARCHIVE, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["force"] == "huber_local" and r["scheme"] == "wave"]
    out = {r["graph"]: r for r in rows}
    assert len(out) == len(rows) == 48, "archive must hold exactly one huber_local/wave row per t310 graph"
    return out


def fnum(s):
    return float(s) if s not in ("", None) else float("nan")


def same(a, b, tol=1e-12):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol


# ---- part 1 --------------------------------------------------------------------------------------------
def part1(checks):
    arch = read_archive()
    specs = t310_specs()
    inits, graphs, trace = [], [], []
    for spec in specs:
        g, truth, meta = C.make_graph(spec)
        u0 = t310_init(spec, g)
        legacy = C.run(g, "wave", "huber_local", u0, c=C_LEGACY, steps=LEGACY_STEPS)
        diag = integrate(g, make_force(g, "huber_local", "orig"), u0, C_LEGACY, LEGACY_STEPS, "legacy")
        fs = legacy.fail_step
        prefix_ok = all(np.array_equal(legacy.X[:(f if f >= 0 else LEGACY_STEPS + 1), r],
                                       diag["X"][:(f if f >= 0 else LEGACY_STEPS + 1), r]) for r, f in enumerate(fs))
        base = dict(part=1, graph=meta["id"], family=spec["block"], bridges=spec["bridges"], seed=spec["seed"],
                    force="huber_local", variant="orig")
        rows, ser = init_rows(diag, g, base)
        for row, f in zip(rows, fs):
            row["legacy_fail_step"] = int(f)
        inits += rows
        cen = np.array([r["first_centered_step"] for r in rows])
        trace += series_rows(diag, ser, base, diag["first_abs"], cen)
        a = arch[meta["id"]]
        n_ok = int((fs < 0).sum())
        mfs = float(fs[fs >= 0].mean()) if (fs >= 0).any() else float("nan")
        ro = readout(legacy.X, g, truth, fs < 0)                    # archived readout: survivors of zeroed runs
        cls = [r["cls"] for r in rows]
        graphs.append(dict(
            graph=meta["id"], family=spec["block"], bridges=spec["bridges"], seed=spec["seed"],
            archive_n_ok=int(a["n_ok"]), replay_n_ok=n_ok, n_ok_match=int(n_ok == int(a["n_ok"])),
            archive_mean_fail_step=fnum(a["mean_fail_step"]), replay_mean_fail_step=mfs,
            mean_fail_step_match=int(same(mfs, fnum(a["mean_fail_step"]), 1e-9)),
            archive_ari_raw=fnum(a["ari_edge_cut"]), replay_ari_raw=ro["raw_ari"],
            archive_ari_lp=fnum(a["ari_edge_cut_lp"]), replay_ari_lp=ro["lp_ari"],
            archive_auc_raw=fnum(a["bridge_auc"]), replay_auc_raw=ro["raw_auc"],
            archive_auc_lp=fnum(a["bridge_auc_lp"]), replay_auc_lp=ro["lp_auc"],
            readout_match=int(all(same(ro[k], fnum(a[c])) for k, c in (("raw_ari", "ari_edge_cut"),
                              ("lp_ari", "ari_edge_cut_lp"), ("raw_auc", "bridge_auc"), ("lp_auc", "bridge_auc_lp")))),
            diag_first_abs_matches_legacy=int(np.array_equal(diag["first_abs"], fs)), legacy_prefix_bitwise=int(prefix_ok),
            n_fail_abs=int((diag["first_abs"] >= 0).sum()), n_fail_centered=int((cen >= 0).sum()),
            n_censored=sum(r["censored"] for r in rows), n_rel_growth=sum(r["rel_growth"] for r in rows),
            **{f"cls_{c}": cls.count(c) for c in ("center_only_crossing", "mixed_center_crossing_relative_growth",
                                                 "relative_crossing", "no_crossing_relative_growth", "no_crossing",
                                                 "censored")},
            max_center_eq_residual=center_eq_residual(diag, ser[0]),
            archive_dc_dw=fnum(a["dc_dw"])))
    checks.append(dict(check="part1_graph_count", value=len(graphs), tol="==48", passed=int(len(graphs) == 48 and
                       {r["graph"] for r in graphs} == set(arch)), hard=1))
    checks.append(dict(check="part1_diag_first_abs_equals_legacy_fail_step", value=sum(r["diag_first_abs_matches_legacy"]
                       for r in graphs), tol="48/48 graphs", passed=int(all(r["diag_first_abs_matches_legacy"] for r in graphs)), hard=1))
    checks.append(dict(check="part1_diag_prefix_bitwise_equals_legacy", value=sum(r["legacy_prefix_bitwise"] for r in graphs),
                       tol="48/48 graphs", passed=int(all(r["legacy_prefix_bitwise"] for r in graphs)), hard=1))
    res = max(r["max_center_eq_residual"] for r in graphs)
    checks.append(dict(check="part1_center_recursion_residual", value=res, tol=TOL_CENTER_EQ, passed=int(res <= TOL_CENTER_EQ), hard=1))
    for k in ("n_ok_match", "mean_fail_step_match", "readout_match"):
        n = sum(r[k] for r in graphs)
        checks.append(dict(check=f"part1_archive_{k}", value=n, tol="48/48 graphs (discrepancies reported, not fatal)",
                           passed=int(n == 48), hard=0))
    return inits, graphs, trace


# ---- parts 2-4 -----------------------------------------------------------------------------------------
def run_config(spec, g, truth, meta, u0, force, variant, role, h):
    steps = int(round(T_PHYS / h))
    run = integrate(g, make_force(g, force, variant), u0, h, steps, "taylor")
    base = dict(part=2 if force == "huber_local" else 3, graph=meta["id"], family=spec["block"],
                bridges=spec["bridges"], seed=spec["seed"], force=force, variant=variant, role=role)
    rows, ser = init_rows(run, g, base)
    cen = np.array([r["first_centered_step"] for r in rows])
    trace = series_rows(run, ser, base, run["first_abs"], cen)
    k = int(round(C_LEGACY / h))
    Xs = run["X"][::k]                                     # common physical frames t = 0, 0.5, ..., 250
    assert Xs.shape[0] == LEGACY_STEPS + 1
    censored = np.array([r["censored"] for r in rows], bool)
    ro = readout(Xs, g, truth, ~censored)
    n_fail = int((run["first_abs"] >= 0).sum())
    cls = [r["cls"] for r in rows]
    m = ser[0]
    scale = max(1.0, float(np.nanmax(np.abs(run["X"]))))
    g_row = dict(base, h=h, startup="taylor_zero_velocity", steps=steps, n_init=R, n_fail_abs=n_fail,
                 n_fail_centered=int((cen >= 0).sum()), n_censored=int(censored.sum()),
                 n_rel_growth=sum(r["rel_growth"] for r in rows), n_scored=int((~censored).sum()),
                 **{f"cls_{c}": cls.count(c) for c in ("center_only_crossing", "mixed_center_crossing_relative_growth",
                                                      "relative_crossing", "no_crossing_relative_growth",
                                                      "no_crossing", "censored")},
                 max_abs_center_accel=float(np.nanmax(np.abs(run["A"]))),
                 max_conservation_ratio=float(np.nanmax(run["cons"])),
                 max_center_shift_rel=float(np.nanmax(np.abs(m - m[0])) / scale),
                 center_eq_residual=center_eq_residual(run, m))
    g_row.update(ro)
    for tag in ("raw", "lp"):
        g_row[f"{tag}_ari_plan"] = 0.0 if n_fail > 0 else g_row[f"{tag}_ari"]
        g_row[f"{tag}_plan_unscorable"] = int(n_fail == 0 and math.isnan(g_row[f"{tag}_ari"]))
    g_row["graph_class"] = graph_class(g_row, rows)
    return run, rows, trace, g_row


def graph_class(g_row, rows):
    """Descriptive per-graph label; see README for the rule order."""
    if g_row["n_censored"]:
        return "censored"
    if any(r["cls"] in ("relative_crossing", "mixed_center_crossing_relative_growth", "no_crossing_relative_growth")
           for r in rows):
        return "relative_growth"
    if g_row["n_fail_abs"]:
        return "center_only_crossing"
    if g_row["raw_noise_floor"] and g_row["lp_noise_floor"]:
        return "stable_noise_floor"
    aris = [g_row["raw_ari"], g_row["lp_ari"]]
    if all(math.isnan(a) or a < NONINFORMATIVE_ARI for a in aris):
        return "stable_noninformative"
    return "stable_ari_ge_0.5"


def recenter_check(orig, proj, twin, g):
    """Per init: deviation of the projected run from the re-centered original,
    max |u_proj - (u_orig - m_orig + m0)| / max(1, max|u_orig|) over frames where both are finite (full horizon
    and t <= RECENTER_EARLY_T), the first frame with absolute deviation > 1e-9, and the same relative deviation
    between the centered states of the original run and of a twin started one ulp above u0 (rounding-level
    sensitivity of the original dynamics)."""
    d = C.wdeg(g)
    Xo, Xp, Xt = orig["X"], proj["X"], twin["X"]
    mo = Xo @ d / d.sum()
    target = Xo - (mo - mo[0])[..., None]
    with np.errstate(invalid="ignore"):
        dev = np.abs(Xp - target).max(-1)                       # (T1, R), NaN where either censored
        tdev = np.abs((Xt - (Xt @ d / d.sum())[..., None]) - (Xo - mo[..., None])).max(-1)
    early = np.arange(Xo.shape[0]) * orig["h"] <= RECENTER_EARLY_T
    out = []
    for r in range(Xo.shape[1]):
        ok = np.isfinite(dev[:, r])
        okt = np.isfinite(tdev[:, r])
        scale = max(1.0, float(np.nanmax(np.abs(Xo[ok, r]))))
        big = np.flatnonzero(ok & (dev[:, r] > 1e-9))
        out.append(dict(max_rel_dev=float(dev[ok, r].max() / scale),
                        early_max_rel_dev=float(dev[ok & early, r].max() / scale), common_finite_frames=int(ok.sum()),
                        first_frame_absdev_over_1e_9=int(big[0]) if big.size else -1,
                        twin_max_rel_dev=float(tdev[okt, r].max() / scale)))
    return out


def freq_check(spec, g, meta, h):
    """Mean wave from a single D-orthonormal L_rw eigenvector with Taylor startup must follow cos(w n h) exactly,
    w = 2 asin(h sqrt(lam)/2)/h.  Rows for the Fiedler and top modes."""
    lam, Phi, d = C.spectrum(g)
    steps = int(round(T_PHYS / h))
    rows = []
    for k in (1, g.n - 1):
        run = integrate(g, make_force(g, "mean", "orig"), Phi[:, k][None, :], h, steps, "taylor")
        a = run["X"][:, 0, :] @ (d * Phi[:, k])
        w = 2 * math.asin(h * math.sqrt(lam[k]) / 2) / h
        expect = np.cos(w * h * np.arange(steps + 1))
        resid = np.abs(run["X"][:, 0, :] - a[:, None] * Phi[:, k][None, :]).max()
        sel = np.flatnonzero(np.abs(a[1:-1]) > 0.5) + 1
        w_est = float(np.median(np.arccos(np.clip((a[sel + 1] + a[sel - 1]) / (2 * a[sel]), -1, 1)) / h))
        rows.append(dict(graph=meta["id"], h=h, mode=k, lam=float(lam[k]), h2lam=float(h * h * lam[k]),
                         omega_discrete=w, omega_continuous=math.sqrt(lam[k]), omega_est=w_est,
                         max_amp_dev=float(np.abs(a - expect).max()), max_offmode=float(resid)))
    return rows


def hand_check():
    """First bridge force for +1/-1 blocks on two 24-cliques joined by one bridge of weight 0.25."""
    spec = dict(kind="two_block", block="clique", sizes=(24, 24), bridges=1, wb=0.25, seed=0)
    g, truth, meta = C.make_graph(spec)
    u = np.where(truth == 0, 1.0, -1.0)[None, :]
    e = int(meta["bridge_idx"][0])
    a, b = int(g.ei[e]), int(g.ej[e])
    if truth[a] != 0:
        a, b = b, a
    da = 23 + 0.25                                    # clique degree 23 plus the bridge weight
    rows = []
    for name, phi in (("mean", -2.0), ("huber", float(np.clip(-2.0, -0.5, 0.5)))):
        F = make_force(g, name, "orig")(u)[0]
        hand_a = 0.25 * phi / da
        others = np.delete(F, [a, b])
        for h in HS:
            rows.append(dict(graph=meta["id"], force=name, h=h, bridge_node_plus=a, bridge_node_minus=b,
                             F_plus_code=float(F[a]), F_plus_hand=hand_a, F_minus_code=float(F[b]), F_minus_hand=-hand_a,
                             max_abs_F_other=float(np.abs(others).max()),
                             u1_plus_taylor=float(1 + 0.5 * h * h * F[a]), u1_plus_hand=1 + 0.5 * h * h * hand_a,
                             err=float(max(abs(F[a] - hand_a), abs(F[b] + hand_a), np.abs(others).max())),
                             label="sanity check only (block startup, not a partition score)"))
    return rows


def parts234(checks):
    sl = slice_specs()
    assert len(sl) == 12
    inits, graphs, trace, sens, recenter, freq = [], [], [], [], [], []
    for spec in sl:
        g, truth, meta = C.make_graph(spec)
        u0 = t310_init(spec, g)
        configs = MAIN_CONFIGS + (OWA_CONFIGS if meta["id"] in OWA_GRAPHS else [])
        by = {}
        for force, variant, role in configs:
            for h in HS:
                run, rows, tr, g_row = run_config(spec, g, truth, meta, u0, force, variant, role, h)
                by[(force, variant, h)] = (run, rows, g_row)
                inits += rows
                trace += tr
                graphs.append(g_row)
        for force, variant, role in configs:          # projection == re-centered original
            if variant != "proj":
                continue
            for h in HS:
                twin = integrate(g, make_force(g, force, "orig"), np.nextafter(u0, np.inf), h, int(round(T_PHYS / h)),
                                 "taylor")
                for r, res in enumerate(recenter_check(by[(force, "orig", h)][0], by[(force, "proj", h)][0], twin, g)):
                    recenter.append(dict(graph=meta["id"], force=force, h=h, init=r, **res))
        for force, variant, role in configs:          # h sensitivity at common physical times
            (ra, rows_a, ga), (rb, rows_b, gb) = by[(force, variant, 0.5)], by[(force, variant, 0.25)]
            d = C.wdeg(g)
            row = dict(graph=meta["id"], family=spec["block"], bridges=spec["bridges"], seed=spec["seed"], force=force,
                       variant=variant, role=role)
            for tag, gr in (("h0.5", ga), ("h0.25", gb)):
                for k in ("n_fail_abs", "n_fail_centered", "n_censored", "n_rel_growth", "graph_class", "raw_ari",
                          "lp_ari", "raw_auc", "lp_auc", "raw_margin", "lp_margin"):
                    row[f"{k}_{tag}"] = gr[k]
            for tt in SENS_TIMES:
                ka, kb = int(round(tt / 0.5)), int(round(tt / 0.25))
                va, vb = ra["X"][ka], rb["X"][kb]
                va = va - (va @ d / d.sum())[:, None]
                vb = vb - (vb @ d / d.sum())[:, None]
                rel = np.sqrt(((va - vb) ** 2) @ d) / np.sqrt((vb ** 2) @ d)
                row[f"centered_reldiff_t{tt:g}_median"] = float(np.median(rel)) if np.isfinite(rel).all() else float("nan")
            row["discretization_sensitive"] = int(ga["graph_class"] != gb["graph_class"] or ga["n_fail_abs"] != gb["n_fail_abs"]
                                                  or any(abs(ga[k] - gb[k]) > SENS_DARI for k in ("raw_ari", "lp_ari")
                                                         if not (math.isnan(ga[k]) or math.isnan(gb[k]))))
            sens.append(row)
        for h in HS:
            freq += freq_check(spec, g, meta, h)
    hand = hand_check()

    def chk(name, vals, tol, hard=1):
        v = max(vals)
        checks.append(dict(check=name, value=v, tol=tol, passed=int(v <= tol), hard=hard))
    free = [r for r in graphs if r["variant"] == "proj" or r["force"] in ("huber_local_sym", "mean", "huber")]
    chk("mean_free_forces_conservation_ratio", [r["max_conservation_ratio"] for r in free], TOL_CONSERVE)
    chk("mean_free_forces_center_preserved", [r["max_center_shift_rel"] for r in free], TOL_CENTER_KEEP)
    chk("slice_center_recursion_residual", [r["center_eq_residual"] for r in graphs], TOL_CENTER_EQ)
    chk("projection_equals_recentered_original_full_horizon", [r["max_rel_dev"] for r in recenter], TOL_RECENTER, hard=0)
    chk("projection_equals_recentered_original_t_le_50", [r["early_max_rel_dev"] for r in recenter], TOL_RECENTER_EARLY)
    ratios = []
    for key in sorted({(r["force"], r["h"]) for r in recenter}):
        rs = [r for r in recenter if (r["force"], r["h"]) == key]
        ratios.append(max(r["max_rel_dev"] for r in rs) / max(r["twin_max_rel_dev"] for r in rs))
    chk("projection_dev_over_1ulp_twin_dev", ratios, TWIN_FACTOR)
    chk("linear_mode_frequency_amplitude_dev", [r["max_amp_dev"] for r in freq], TOL_FREQ)
    chk("linear_mode_offmode_leak", [r["max_offmode"] for r in freq], TOL_FREQ)
    chk("bridge_force_hand_check", [r["err"] for r in hand], TOL_HAND)
    checks.append(dict(check="slice_graph_count", value=len(sl), tol="==12", passed=int(len(sl) == 12), hard=1))
    return inits, graphs, trace, sens, recenter, freq, hand


def summarize(graphs, keys):
    """Counts / means over graphs for each configuration key."""
    out = {}
    for r in graphs:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    rows = []
    for key, rs in out.items():
        row = dict(zip(keys, key), n_graphs=len(rs), runs=R * len(rs), run_fail_abs=sum(r["n_fail_abs"] for r in rs),
                   graphs_any_fail_abs=sum(r["n_fail_abs"] > 0 for r in rs),
                   run_fail_centered=sum(r["n_fail_centered"] for r in rs), run_censored=sum(r["n_censored"] for r in rs),
                   run_rel_growth=sum(r["n_rel_growth"] for r in rs))
        if "raw_ari" in rs[0]:
            for tag in ("raw", "lp"):
                for k in ("ari", "ari_plan"):
                    v = [r[f"{tag}_{k}"] for r in rs if not math.isnan(r[f"{tag}_{k}"])]
                    row[f"{tag}_{k}_mean"] = float(np.mean(v)) if v else float("nan")
                    row[f"{tag}_{k}_n"] = len(v)
                for k in ("auc", "bridge_med", "internal_med"):
                    v = [r[f"{tag}_{k}"] for r in rs if not math.isnan(r[f"{tag}_{k}"])]
                    row[f"{tag}_{k}_median"] = float(np.median(v)) if v else float("nan")
            for c in ("censored", "relative_growth", "center_only_crossing", "stable_noise_floor",
                      "stable_noninformative", "stable_ari_ge_0.5"):
                row[f"graphs_{c}"] = sum(r["graph_class"] == c for r in rs)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: the script's directory)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert wt.FAIL_AMP == AMP and C_LEGACY ** 2 == 0.25
    checks = []
    p1_inits, p1_graphs, p1_trace = part1(checks)
    write_csv(args.out / "replay_inits.csv", p1_inits)
    write_csv(args.out / "replay_graphs.csv", p1_graphs)
    write_csv(args.out / "replay_trace.csv.gz", p1_trace)
    p1_sum = summarize(p1_graphs, ["family"]) + summarize(p1_graphs, ["bridges"]) + \
        summarize([dict(r, all="all") for r in p1_graphs], ["all"])
    write_csv(args.out / "replay_summary.csv", p1_sum)

    inits, graphs, trace, sens, recenter, freq, hand = parts234(checks)
    write_csv(args.out / "slice_inits.csv", inits)
    write_csv(args.out / "slice_graphs.csv", graphs)
    write_csv(args.out / "slice_trace.csv.gz", trace)
    write_csv(args.out / "slice_h_sensitivity.csv", sens)
    write_csv(args.out / "slice_recenter_check.csv", recenter)
    write_csv(args.out / "check_linear_frequency.csv", freq)
    write_csv(args.out / "check_bridge_force.csv", hand)
    write_csv(args.out / "slice_summary.csv", summarize(graphs, ["force", "variant", "role", "h"]))
    write_csv(args.out / "checks.csv", checks)

    cls = [r["cls"] for r in p1_inits]
    summary = dict(
        environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                         scipy=__import__("scipy").__version__, sklearn=__import__("sklearn").__version__),
        part1=dict(runs=len(p1_inits), run_fail_abs=sum(r["first_abs_step"] >= 0 for r in p1_inits),
                   graphs_any_fail=sum(r["n_fail_abs"] > 0 for r in p1_graphs),
                   archive_run_fail=sum(8 - r["archive_n_ok"] for r in p1_graphs),
                   init_classes={c: cls.count(c) for c in sorted(set(cls))}),
        checks=checks, hard_failures=[c["check"] for c in checks if c["hard"] and not c["passed"]])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["hard_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
