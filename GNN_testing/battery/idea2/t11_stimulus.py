"""T2.11 stimulus saturation curve.  far_peak vs amp in {0.2,1,5,20} (both signs), detection AUC, mean force, c=1, R=8, det_trials 16, T=200.
Compare relative (RMS(u0)-normalised -> state-scale-invariant, so it cannot clip) schemes with *_abs (fixed scale sigma=0.577) variants.
Graphs: cycle/clique/sbm two_block 24+24 bridges 2, seeds 0-2 (9).  fail threshold 1e4 so the amp=20 stimulus is not 'failed' by definition."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
STEPS = 200
FA = 1e4

def configs():
    cf = [("wave", "wave", {}), ("damp0.1", "damp", {"k": 0.1})]
    for v in (0.1, 0.3, 1.0):
        cf += [(f"limiter{v:g}", "limiter", {"vmax": v}), (f"limiter_abs{v:g}", "limiter_abs", {"vmax": v})]
    for v in (0.3, 1.0, 3.0):
        cf += [(f"saturate{v:g}", "saturate", {"kappa": v}), (f"saturate_abs{v:g}", "saturate_abs", {"kappa": v})]
    for v in (0.5, 3.0):
        cf += [(f"drag{v:g}", "drag", {"alpha": v}), (f"drag_abs{v:g}", "drag_abs", {"alpha": v})]
    for v in (1.0, 10.0):
        cf += [(f"vel_shrink{v:g}", "vel_shrink", {"alpha": v}), (f"vel_shrink_abs{v:g}", "vel_shrink_abs", {"alpha": v})]
    cf += [("cfl0.9", "cfl", {"s": 0.9}), ("energy0.5", "energy", {"kmax": 0.5}), ("energy_abs0.5", "energy_abs", {"kmax": 0.5})]
    return cf

def job(args):
    spec, c = args
    g, truth, meta = C.make_graph(spec)
    rows = []
    for name, sch, ps in configs():
        runner = C.make_runner(sch, "mean", c, ps=ps, steps=STEPS, fail_amp=FA)
        st = C.stimulus(g, 24, runner, amps=(0.2, 1, 5, 20), signs=(+1, -1), sigma=0.05, R=8, steps=STEPS, seed=spec["seed"], det_trials=16)
        for r in st:
            r = dict(r); r.update(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], config=name, scheme=sch, c=c)
            rows.append(r)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1, 2))
    out = C.pmap(job, [(s, 1.0) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t11_stimulus.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
