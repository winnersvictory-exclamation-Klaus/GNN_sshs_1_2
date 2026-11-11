"""T1.10 stimulus cost of damping: far_peak, arrival, SNR, detection AUC vs k, nu."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

STEPS = 200
MEAN_CFGS = [("wave", {}), ] + [("damp", dict(k=k)) for k in (0.003, 0.01, 0.03, 0.1, 0.3)] + \
            [("viscous", dict(nu=v)) for v in (0.05, 0.2, 0.4)] + [("damp_visc", dict(k=0.03, nu=0.1))]
NL_FORCES = ["huber", "gauss", "trim", "owa", "median"]
NL_CFGS = [("damp", dict(k=0.1)), ("damp", dict(k=0.3)), ("damp", dict(k=0.6))]

def job(spec):
    g, truth, meta = C.make_graph(spec)
    n0 = spec["sizes"][0]
    rows = []
    cfgs = [("mean", sch, ps, 0.5) for sch, ps in MEAN_CFGS] + [("mean", "damp", dict(k=0.1), 1.0)]
    cfgs += [(f, sch, ps, 0.5) for f in NL_FORCES for sch, ps in NL_CFGS]
    runners = [(f, sch, ps, c, C.make_runner(sch, f, c, ps, steps=STEPS)) for f, sch, ps, c in cfgs]
    runners.append(("mean", "diffusion", {}, np.nan, C.make_diff_runner("mean", 0.5, steps=STEPS)))
    for f, sch, ps, c, rn in runners:
        for r in C.stimulus(g, n0, rn, R=8, steps=STEPS, seed=spec["seed"]):
            r.update(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=f, scheme=sch,
                     ps=",".join(f"{k}={v}" for k, v in ps.items()), c=c)
            rows.append(r)
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3])
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0); sys.exit()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t10_stimulus.csv", rows)
    print(len(rows))
