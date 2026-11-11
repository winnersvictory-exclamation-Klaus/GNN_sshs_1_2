"""idea-5 helpers: registered extras + small analysis utilities (imports common FIRST)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
import numpy as np
wt, rx = C.wt, C.rx
RES = Path(__file__).parent / "results"
BASES = ("mean", "trim", "owa", "vw", "huber")


# ---- pure biharmonic ("plate"): force = -(F o F)*... = series weights [0,1]: -L^2 u for the mean base ----------
def _make_plate(kind):
    def fn(ctx, u):
        return wt.series_force(ctx.g, u, kind, [0.0, 1.0])
    fn.__doc__ = f"idea5 plate: -F(F(u)) (= -L^2 u for mean). base={kind}"
    return fn


for _b in BASES:
    wt.force(f"plate:{_b}")(_make_plate(_b))


# ---- t4 + extra velocity damping (T5.13), and t4 with consistent start-up (T5.06) ----------------------------
@wt.scheme("t4damp", eps=0.1, k=0.1)
def _(ctx, cur, prev, F, Fp):
    """t4 step minus k*(cur-prev): explicit velocity damping added to the t4 update"""
    return wt.time_even_step(ctx, cur, prev, F, [ctx.ps["eps"]]) - ctx.ps["k"] * (cur - prev)


@wt.scheme("t4plain_start", eps=0.1, nstart=4)
def _(ctx, cur, prev, F, Fp):
    """t4, but the first nstart steps use the plain wave update (history is then consistent, not u0 repeated).
    The history buffer is still filled by time_even_step's own bookkeeping, so we maintain it here."""
    n = int(ctx.ps["nstart"])
    if ctx.t < n:
        # keep history in sync with time_even_step's convention: h[0]=f_t, h[1]=f_{t-1}, ...
        h = ctx.state.get("s_hist")
        n_hist = 4
        h = [cur] * n_hist if h is None else [cur] + h[:-1]
        ctx.state["s_hist"] = h
        return 2 * cur - prev + ctx.c ** 2 * F
    return wt.time_even_step(ctx, cur, prev, F, [ctx.ps["eps"]])


# ---- modal / linear helpers ---------------------------------------------------------------------------------
def t4_poly(e, s):
    """characteristic polynomial coefficients (descending) of time_even_step(eps_list=[e]) with F=-lam f, s=c^2 lam.
    Derived from the code: lead*f_{t+1} + (-2-4e) f_t + (1+6e) f_{t-1} - 4e f_{t-2} + e f_{t-3} = -s f_t, lead = 1+e."""
    return np.array([1 + e, s - 2 - 4 * e, 1 + 6 * e, -4 * e, e], float)


def t4_rho(e, s):
    return float(np.max(np.abs(np.roots(t4_poly(e, s)))))


def spatial_g(lam, eps):
    return lam * (1 + eps * lam)


def wave_modal_fail(g, gfun, c, u0, T, fail_amp=None):
    """Exact linear prediction (mean base): per mode a_{t+1} = 2a_t - a_{t-1} - c^2 g(lam) a_t, a_{-1}=a_0.
    Returns predicted fail_step (R,), same rule as wt.simulate (|u|>=FAIL_AMP or non-finite)."""
    amp = wt.FAIL_AMP if fail_amp is None else fail_amp
    lam, Phi, d = C.spectrum(g)
    a = (u0 * d) @ Phi
    ap = a.copy()
    s = c * c * gfun(lam)
    fail = np.full(u0.shape[0], -1)
    with np.errstate(all="ignore"):
        for t in range(T):
            nxt = 2 * a - ap - s * a
            u = nxt @ Phi.T
            bad = ~(np.abs(u) < amp).all(1) & (fail < 0)
            fail[bad] = t + 1
            nxt[fail >= 0] = 0.0
            ap, a = a, nxt
    return fail


def pred_ln_rho(s):
    """log spectral radius of z^2-(2-s)z+1 (s=c^2 g): 0 if 0<=s<=4 else ln of the larger root."""
    s = np.asarray(s, float)
    b = 2 - s
    disc = b * b - 4
    z = np.where(disc > 0, (np.abs(b) + np.sqrt(np.maximum(disc, 0))) / 2, 1.0)
    return np.log(z)


# ---- common per-config evaluation ------------------------------------------------------------------------------
HOLD_READOUTS = ("edge_cut", "edge_cut_lp", "fft_lowmode")
ALL_READOUTS = C.PARTITION_NAMES


def eval_run(run, g, truth, meta, label, readouts=ALL_READOUTS, hold_readouts=HOLD_READOUTS, extra=None):
    """Flat CSV row for one (graph, config): trajectory summary, all readouts, edge_margin, hold, mode energy."""
    row = dict(graph=meta["id"], seed=meta["seed"], kind=meta["kind"], n=meta["n"], bridges=meta["n_bridges"], config=label)
    row.update(C.traj_summary(run))
    if extra:
        row.update(extra)
    if run.ok.any():
        row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step, which=readouts))
        labs = C.partitions(run.X, g, 2, run.fail_step, which=hold_readouts)
        for nm, lab in labs.items():
            h = C.hold(run.X, g, lab, fail_step=run.fail_step)
            row[f"hold_{nm}"] = float(np.nanmean(h["hold"]))
            row[f"holdpos_{nm}"] = float(np.nanmean(h["hold_pos"]))
        me = C.mode_energy(run.X, g, run.fail_step, weight="pot")
        T = run.X.shape[0]
        row["high_pot_late"] = float(np.nanmean(me["high"][T // 2:]))
        row["low_pot_late"] = float(np.nanmean(me["low"][T // 2:]))
        md = C.mode_energy(run.X, g, run.fail_step, weight="disp")
        row["high_disp_late"] = float(np.nanmean(md["high"][T // 2:]))
    return row


def spectral_row(g, truth, meta, label="spectral"):
    lab = rx.spectral_labels(g)
    m = C.partition_metrics(g, lab, truth)
    return dict(graph=meta["id"], seed=meta["seed"], kind=meta["kind"], n=meta["n"], bridges=meta["n_bridges"], config=label,
                ari_edge_cut=m["ari"], ari_spectral=m["ari"])


def ari_vs_T(run, g, truth, Ts, which=("edge_cut", "edge_cut_lp", "fft_lowmode"), thr=0.9):
    """ARI of each readout computed on the prefix X[:T+1] for T in Ts; first T with ARI>=thr (nan if never)."""
    out = {}
    for nm in which:
        out[f"ari_T_{nm}"] = []
    for T in Ts:
        Xp = run.X[:T + 1]
        fs = np.where((run.fail_step >= 0) & (run.fail_step <= T), run.fail_step, -1)
        labs = C.partitions(Xp, g, 2, fs, which=which)
        for nm in which:
            m = C.partition_metrics(g, labs[nm], truth)
            out[f"ari_T_{nm}"].append(m["ari"])
    res = {}
    for nm in which:
        arr = np.array(out[f"ari_T_{nm}"], float)
        for T, a in zip(Ts, arr):
            res[f"ari_{nm}_T{T}"] = a
        hit = np.where(arr >= thr)[0]
        res[f"stp_{nm}"] = float(Ts[hit[0]]) if len(hit) else np.nan
    return res


def graph_job_setup(spec, R=8, kind="uniform"):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init(kind, g, R, rng, truth=truth)
    return g, truth, meta, u0


def baselines(steps=500):
    return [("wave+mean c=0.5", C.make_runner("wave", "mean", 0.5, steps=steps)),
            ("wave+mean c=1", C.make_runner("wave", "mean", 1.0, steps=steps)),
            ("damp(k=.1)+mean c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=steps)),
            ("diffusion+mean", C.make_diff_runner("mean", eta=0.5, steps=steps))]


def summarise(df, value, by="config", ci=True):
    """mean + bootstrap CI over graphs for each config."""
    rows = []
    for k, sub in df.groupby(by, sort=False):
        x = sub[value].dropna().values
        if len(x):
            est, lo, hi = C.boot_ci(x)
        else:
            est = lo = hi = np.nan
        rows.append(dict(config=k, n=len(x), mean=est, lo=lo, hi=hi))
    return pd.DataFrame(rows)


import pandas as pd  # noqa: E402


def truncate(run, T):
    """Run restricted to the first T steps (failures after T are masked as non-failed)."""
    from dataclasses import replace
    fs = np.where((run.fail_step >= 0) & (run.fail_step <= T), run.fail_step, -1)
    return replace(run, X=run.X[:T + 1], fail_step=fs, steps=T)
