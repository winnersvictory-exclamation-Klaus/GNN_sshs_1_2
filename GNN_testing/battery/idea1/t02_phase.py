"""T1.02 phase diagram: k x nu x c x force (damp_visc family)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *
import itertools

FORCES = ["mean", "trim", "owa", "vw", "median", "gauss", "absvw", "symowa_ext", "symowa_med", "huber"]
KS = [0.0] + list(np.geomspace(1e-3, 0.5, 10))   # k=0 column added (reference); 10 logspace values as designed
NUS = [0, 0.05, 0.1, 0.2, 0.4]
CS = [0.25, 0.5, 1.0, 1.4, 1.8]
T = 1000; R = 6

def job(args):
    spec, forces, ks, nus, cs, T, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for f in forces:
        for k, nu, c in itertools.product(ks, nus, cs):
            sch, ps = scheme_for(k, nu)
            run = erun(g, sch, f, u0, c, T, ps)
            s = C.traj_summary(run)
            rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=f, k=k, nu=nu, c=c, T=T,
                             fail_frac=s["fail_frac"], mean_fail_step=s["mean_fail_step"], growth_rate=s["growth_rate"],
                             dead_frac=s["dead_frac"], ac_final=s["ac_final"], dc_dw=s["dc_dw"], ac_peak=s["ac_peak"]))
    return rows

if __name__ == "__main__":
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    specs = core_specs([0, 1])
    if test:
        t0 = time.time()
        rows = job((specs[0], ["owa", "mean"], KS[:3], NUS[:2], CS[:2], T, R)); print(len(rows), time.time() - t0); sys.exit()
    # one job per (graph, force) to balance load
    jobs = [(s, [f], KS, NUS, CS, T, R) for s in specs for f in FORCES]
    t0 = time.time()
    rs = C.pmap(job, jobs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t02_phase.csv", rows)
    print(len(rows), time.time() - t0)
