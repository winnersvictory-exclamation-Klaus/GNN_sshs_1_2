"""T5.01 fine eps sweep (space): wave + d4:mean, eps in linspace(-1,1,41) x c in {.25,.5,1,1.4}.
Theory: modal stiffness g(lam)=lam(1+eps lam); stable iff 0 < c^2 g(lam) < 4 for every non-consensus mode.
Outputs results/t01_spatial_theory.csv (+ results/t01_ccrit.csv with --ccrit)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *
import pandas as pd

EPS = np.round(np.linspace(-1, 1, 41), 4)
CS = [0.25, 0.5, 1.0, 1.4]
T = 1000


def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, R, rng)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for e in EPS:
        gv = spatial_g(lam[1:], e)
        for c in CS:
            s = c * c * gv
            ln_rho = pred_ln_rho(s)
            pred_stable = bool((s > 1e-9).all() and (s < 4 - 1e-9).all())
            r = C.run(g, "wave", "d4:mean", u0, c, T, pf=dict(eps=float(e)))
            pfail = wave_modal_fail(g, lambda l: spatial_g(l, e), c, u0, T)
            tm = C.traj_metrics(r)
            fs = r.fail_step
            rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=spec["bridges"], n=meta["n"],
                             eps=float(e), c=c, lam_max=lam[-1], gmin=gv.min(), gmax=gv.max(),
                             c_crit_pred=(2 / np.sqrt(gv.max()) if gv.min() > 0 else 0.0),
                             pred_stable=pred_stable, pred_ln_rho=float(ln_rho.max()),
                             pred_nonpos_modes=int((s <= 1e-9).sum()),
                             fail_frac=float((fs >= 0).mean()), pred_fail_frac=float((pfail >= 0).mean()),
                             exact_match=bool(np.array_equal(fs, pfail)), max_fail_diff=int(np.abs(fs - pfail).max()),
                             med_fail_step=float(np.median(fs[fs >= 0])) if (fs >= 0).any() else np.nan,
                             growth_rate=float(np.nanmedian(tm["growth_rate"])) if np.isfinite(tm["growth_rate"]).any() else np.nan,
                             ac_final=float(np.nanmedian(C.ac_final(r.X, fs))) if r.ok.any() else np.nan))
    return rows


CC_EPS = [-0.4, -0.2, -0.1, -0.05, 0.1, 0.3, 1.0]


def job_cc(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for e in CC_EPS + [0.0]:
        gv = spatial_g(lam[1:], e)
        pred = 2 / np.sqrt(gv.max()) if gv.min() > 0 else 0.0
        out = C.c_crit_empirical(g, "wave", "d4:mean", pf=dict(eps=e), steps=2000, c_lo=0.02, c_hi=3.0, R=8)
        rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], eps=e, c_crit_pred=pred,
                         c_crit_emp=out["c_crit"], status=out["status"], non_monotone=out["non_monotone"]))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    seeds = C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4]
    specs = C.zoo("core", seeds=seeds)
    if "--ccrit" in sys.argv:
        sp = [s for s in C.zoo("core", seeds=C.DEV_SEEDS[:4]) if s.get("bridges") == 2]
        out = C.pmap(job_cc, sp, procs=2)
        df = pd.DataFrame([r for rs in out for r in rs]); C.write_csv(RES / "t01_ccrit.csv", df)
        df["relerr"] = (df.c_crit_emp - df.c_crit_pred) / df.c_crit_pred
        print(df.groupby("eps")[["c_crit_pred", "c_crit_emp", "relerr"]].agg(["median", "max"]).round(4))
        sys.exit()
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t01_spatial_theory.csv" if not small else "t01_small.csv"), df)
    print("rows", len(df), "exact fail-step match", df.exact_match.mean())
    print("pred_stable vs emp fail: ")
    print(pd.crosstab(df.pred_stable, df.fail_frac > 0))
