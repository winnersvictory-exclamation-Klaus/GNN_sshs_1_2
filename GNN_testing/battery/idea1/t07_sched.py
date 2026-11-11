"""T1.07 schedules sched_decay (k0 x Ts) and sched_grow (kinf x Ts): fail_step relative to Ts."""
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

FORCES = ["mean", "owa", "vw", "trim", "median", "huber", "symowa_ext"]
TS = [10, 30, 100, 300]
K0 = [0.1, 0.3, 0.5, 0.9]
KINF = [0.05, 0.1, 0.3, 0.6]
CS = [0.5, 1.0]
T = 1000; R = 8

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    cfgs = [("wave", {}, "wave", np.nan, np.nan)]
    for k0, ts in itertools.product(K0, TS): cfgs.append(("sched_decay", dict(k0=k0, Ts=ts), "sched_decay", k0, ts))
    for ki, ts in itertools.product(KINF, TS): cfgs.append(("sched_grow", dict(kinf=ki, Ts=ts), "sched_grow", ki, ts))
    for k in sorted(set(K0 + KINF)): cfgs.append(("damp", dict(k=k), "damp_const", k, np.nan))
    for f in FORCES:
        for c in CS:
            for sch, ps, fam, kk, ts in cfgs:
                run = erun(g, sch, f, u0, c, T, ps)
                s = C.traj_summary(run)
                fs = run.fail_step[run.fail_step >= 0]
                rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=f, c=c, family=fam, kk=kk, Ts=ts,
                                 fail_frac=s["fail_frac"], fail_step_med=float(np.median(fs)) if len(fs) else np.nan,
                                 fail_step_min=float(fs.min()) if len(fs) else np.nan,
                                 dead_frac=s["dead_frac"], ac_final=s["ac_final"], growth_rate=s["growth_rate"]))
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3])
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t07_sched.csv", rows)
    print(len(rows), time.time() - t0)
