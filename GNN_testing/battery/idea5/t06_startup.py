"""T5.06 start-up sensitivity (time term). t4 as written (history = u0 repeated) vs plain-wave start-up for the
first nstart in {4, 50} steps then t4 (scheme t4plain_start from lib5).  eps in {-0.1,-1/12,-0.05,0.05,0.1,0.5},
c in {0.25,0.5,1.0}; cycle/clique/sbm bridges=2, dev seeds 0-7; R=8; T=2000.
Output results/t06_startup.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS = [-0.1, -1 / 12, -0.05, 0.05, 0.1, 0.5]
CS = [0.25, 0.5, 1.0]
T = 2000
STARTS = [("asis", None), ("plain4", 4), ("plain50", 50)]


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for e in EPS:
        for c in CS:
            lnr = np.log(max(t4_rho(e, c * c * l) for l in lam[1:]))
            for nm, ns in STARTS:
                if ns is None:
                    r = C.run(g, "t4", "mean", u0, c, T, ps=dict(eps=e))
                else:
                    r = C.run(g, "t4plain_start", "mean", u0, c, T, ps=dict(eps=e, nstart=ns))
                s = C.traj_summary(r)
                fs = r.fail_step
                rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], eps=e, c=c, start=nm,
                                 pred_ln_rho=lnr, first_fail=float(fs[fs >= 0].min()) if (fs >= 0).any() else np.nan, **s))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    seeds = C.DEV_SEEDS[:1] if small else C.DEV_SEEDS
    specs = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, specs, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t06_small.csv" if small else "t06_startup.csv"), df)
    print(df.pivot_table(index=["eps", "c"], columns="start", values="mean_fail_step", aggfunc="mean").round(1))
    print(df.pivot_table(index=["eps", "c"], columns="start", values="fail_frac", aggfunc="mean").round(2))
