"""Shared helpers for idea-6 tests (spread transforms)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C          # noqa: E402  (pins BLAS threads)
import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

wt = C.wt
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BASES = wt.BASES
TRANSFORMS = list(wt.SPREAD)            # quad_g, sq_g, quad_l, sq_l, sqrt_g, invq_g, tanh_g, sqrt_l, invq_l, tanh_l
MODE = {k: v[0] for k, v in wt.SPREAD.items()}
SCOPE = {k: v[1] for k, v in wt.SPREAD.items()}
HAS_GAMMA = {"quad", "invq"}


def fname(tr, base="mean"):
    return f"{tr}:{base}"


def pf_for(tr, gamma=1.0):
    return {"gamma": gamma} if MODE[tr] in HAS_GAMMA else None


def core_specs(seeds, blocks=("cycle", "clique", "sbm"), bridges=(1, 2, 4, 8)):
    return [dict(kind="two_block", block=b, sizes=(24, 24), bridges=k, wb=1.0, seed=s)
            for s in seeds for b in blocks for k in bridges]


def gid(meta):
    p = meta["params"]
    return meta["id"]


def base_row(meta, **kw):
    r = dict(graph=meta["id"], seed=meta["seed"], kind=meta["kind"], n=meta["n"], bridges=meta["n_bridges"])
    r.update(kw)
    return r


def block_of(spec):
    return spec.get("block", spec["kind"])


def ci(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0:
        return "n=0"
    e, lo, hi = C.boot_ci(x)
    return f"{e:.3f} [{lo:.3f},{hi:.3f}] n={len(x)}"


def pretty(df, by, vals):
    out = []
    for k, sub in df.groupby(by, sort=False):
        out.append((k, {v: ci(sub[v]) for v in vals}))
    return out


def full_row(run, g, truth, meta, spec_block=None, readouts=("edge_cut",), with_hold=True, **kw):
    """One row: traj_summary + partition metrics + hold (edge_cut labels, min_frac fix)."""
    row = base_row(meta, block=spec_block, **kw)
    row.update(C.traj_summary(run))
    if run.ok.any() and truth is not None:
        try:
            ev = C.evaluate_partitions(run.X, g, truth, run.fail_step, which=readouts)
            row.update(ev)
        except Exception as e:                   # noqa
            row["eval_err"] = str(e)[:50]
        if with_hold:
            labs = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))
            lab = labs.get("edge_cut")
            if lab is not None:
                h = C.hold(run.X, g, lab, fail_step=run.fail_step)
                row["hold_rec"] = float(np.nanmean(h["hold"]))
                h2 = C.hold(run.X, g, lab, fail_step=run.fail_step, norm="ac")
                row["hold_rec_ac"] = float(np.nanmean(h2["hold"]))
    return row


def contrast_metrics(run, g, truth, W=50):
    """Block-contrast survival using TRUE labels. Returns scalar medians over ok runs."""
    out = {}
    if not run.ok.any() or truth is None:
        return out
    X = run.X[:, run.ok]
    T1 = X.shape[0]
    a = truth == truth.min()
    dm = X[:, :, a].mean(2) - X[:, :, ~a].mean(2)            # (T+1, R)
    nw = T1 // W
    env = np.sqrt((dm[:nw * W].reshape(nw, W, -1) ** 2).mean(1))   # (nw, R)
    acr = np.sqrt(((X - X.mean(2, keepdims=True)) ** 2).mean(2))[:nw * W].reshape(nw, W, -1)
    acw = np.sqrt((acr ** 2).mean(1))                          # AC-RMS per window
    out["env0"] = float(np.median(env[0]))
    out["env_ret_250"] = float(np.median(env[min(4, nw - 1)] / (env[0] + 1e-300)))
    out["env_ret_end"] = float(np.median(env[-1] / (env[0] + 1e-300)))
    out["rel_contrast_early"] = float(np.median(env[min(1, nw - 1)] / (acw[min(1, nw - 1)] + 1e-300)))   # contrast / AC-rms
    out["rel_contrast_end"] = float(np.median(env[-1] / (acw[-1] + 1e-300)))
    out["ac_rms_end"] = float(np.median(acw[-1]))
    h = C.hold(run.X, g, truth, fail_step=run.fail_step)
    h2 = C.hold(run.X, g, truth, fail_step=run.fail_step, norm="ac")
    out["hold_true"] = float(np.nanmean(h["hold"]))
    out["hold_true_ac"] = float(np.nanmean(h2["hold"]))
    din, dout = h2["din"], h2["dout"]
    out["dout_din_end"] = float(np.nanmedian(dout[:, -1] / (din[:, -1] + 1e-12)))
    out["dout_din_mid"] = float(np.nanmedian(dout[:, len(dout[0]) // 2] / (din[:, len(dout[0]) // 2] + 1e-12)))
    return out


# ---- frozen-scale spread forces (T6.07) --------------------------------------------------------
def _make_frozen(base, mode, scope):
    def fn(ctx, u):
        g = ctx.g
        gam = ctx.pf.get("gamma", 1.0)
        if scope == "global":
            m = u.mean(1, keepdims=True)
            d = u - m
            if "s_fz" not in ctx.state:
                ctx.state["s_fz"] = wt.rms_rows(d) + wt.EPS
            ut = m + wt._phi(d, ctx.state["s_fz"], mode, gam)
            return wt.base_force(g, ut, base)
        V = wt.gather(g, u)
        m = wt.edge_mean(g, V)
        dV = np.where(g.valid, V - m[..., None], 0.0)
        ds = u - m
        if "s_fz" not in ctx.state:
            ctx.state["s_fz"] = np.sqrt(((g.WT * dV ** 2).sum(-1) + ds ** 2) / (g.WT.sum(-1) + 1)) + wt.EPS
        s = ctx.state["s_fz"]
        Vt = m[..., None] + wt._phi(dV, s[..., None], mode, gam)
        return wt.nb_force(g, Vt, m + wt._phi(ds, s, mode, gam), base)
    fn.__doc__ = f"idea6 frozen-scale {mode} {scope} base {base}"
    return fn


def register_frozen():
    for tr, (mode, scope) in wt.SPREAD.items():
        for b in BASES:
            nm = f"fz_{tr}:{b}"
            if nm not in wt.FORCES:
                wt.force(nm, **({"gamma": 1.0} if mode in HAS_GAMMA else {}))(_make_frozen(b, mode, "global" if scope == "global" else "local"))


register_frozen()


# ---- composite schemes (T6.14) ----------------------------------------------------------------
@wt.scheme("limdamp", vmax=0.3, k=0.1)
def _limdamp(ctx, cur, prev, F, Fp):
    """limiter + linear damping: u+ = u + vm tanh((1-k)v/vm) + c^2 F"""
    vm = ctx.ps["vmax"] * wt.rms_rows(ctx.u0)
    return cur + vm * np.tanh((1 - ctx.ps["k"]) * (cur - prev) / vm) + ctx.c ** 2 * F


@wt.scheme("satdamp", kappa=1.0, k=0.1)
def _satdamp(ctx, cur, prev, F, Fp):
    """saturate + linear damping"""
    if "phi" not in ctx.state:
        ctx.state["phi"] = ctx.ps["kappa"] * wt.rms_rows(F) + wt.EPS
    phi = ctx.state["phi"]
    return cur + (1 - ctx.ps["k"]) * (cur - prev) + ctx.c ** 2 * phi * np.tanh(F / phi)


# ---- config dicts (used by T6.15 / T6.18) -------------------------------------------------------
def build_runner(cfg, steps=500, fail_amp=None):
    """cfg: dict(label, kind in {wave,damp,diff}, force, pf, ps, par)."""
    if cfg["kind"] == "diff":
        return C.make_diff_runner(cfg["force"], eta=cfg["par"], pf=cfg.get("pf"), steps=steps, fail_amp=fail_amp)
    return C.make_runner(cfg["kind"], cfg["force"], c=cfg["par"], ps=cfg.get("ps"), pf=cfg.get("pf"), steps=steps, fail_amp=fail_amp)


BASELINES = [
    dict(label="wave+mean c=.5", kind="wave", force="mean", par=0.5),
    dict(label="damp.1+mean c=.5", kind="damp", force="mean", par=0.5, ps=dict(k=0.1)),
    dict(label="diff+mean eta=.25", kind="diff", force="mean", par=0.25),
    dict(label="diff+mean eta=.5", kind="diff", force="mean", par=0.5),
]
