"""T1.11 long horizon: T=5000 re-run of marginal T1.02 cells (|growth_rate|<=1e-3, fail_frac<1 at T=1000)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *
import pandas as pd

T = 5000; R = 6; MAXCELLS = 200

def job(args):
    spec, cells = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for _, r in cells.iterrows():
        sch, ps = scheme_for(r.k, r.nu)
        run = erun(g, sch, r.force, u0, r.c, T, ps)
        fs = run.fail_step
        row = dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=r.force, k=r.k, nu=r.nu, c=r.c,
                   growth_T1000=r.growth_rate, fail_T1000_old=r.fail_frac)
        for t in (500, 1000, 2000, 5000):
            row[f"fail_T{t}"] = float(((fs >= 0) & (fs <= t)).mean())
        a = C.ac_rms(run.X)
        row["ac_ratio_T5000"] = float(np.median(a[-1] / a[0])) if run.ok.any() else np.nan
        row["growth_T5000"] = C.traj_summary(run)["growth_rate"]
        rows.append(row)
    return rows

if __name__ == "__main__":
    t2 = pd.read_csv(OUT / "t02_phase.csv")
    m = t2[(t2.growth_rate.abs() <= 1e-3) & (t2.fail_frac < 1) & (t2.dead_frac < 1)]
    print("marginal cells", len(m), "of", len(t2))
    specs = {C.make_graph(s)[2]["id"]: s for s in core_specs([0, 1])}
    jobs = []
    for gid, grp in m.groupby("graph"):
        grp = grp.sample(min(len(grp), MAXCELLS), random_state=0)
        jobs.append((specs[gid], grp[["force", "k", "nu", "c", "growth_rate", "fail_frac"]]))
    if len(sys.argv) > 1:
        j = (jobs[0][0], jobs[0][1].head(5)); t0 = time.time(); r = job(j); print(pd.DataFrame(r).round(4).to_string(), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, jobs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t11_long.csv", rows)
    print(len(rows), time.time() - t0)
