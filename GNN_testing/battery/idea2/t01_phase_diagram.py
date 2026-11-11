"""T2.01 phase diagram: 6 schemes x 6 params x c in {0.5,1,1.4,1.8} x 10 forces (+ baselines wave, damp k=.1)
graphs: cycle/clique/sbm two_block 24+24, bridges 2, dev seeds 0-3 (12 graphs), R=8, T=500."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
CS = (0.5, 1.0, 1.4, 1.8)
STEPS = 500

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    cfgs = [("wave", {}, None), ("damp", {"k": 0.1}, None)]
    for b in L.ORIG:
        for v in L.GRID6[b]:
            cfgs.append((b, {L.PARAM[b][0]: v}, v))
    rows = []
    for f in L.FORCES10:
        for c in CS:
            for sch, ps, val in cfgs:
                run = C.run(g, sch, f, u0, c, STEPS, ps)
                row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=sch, param=val if val is not None else np.nan,
                           pname=L.PARAM[sch][0] if sch in L.PARAM else "", force=f, c=c)
                row.update(C.traj_summary(run))
                row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
                ari = np.nan
                if run.ok.any():
                    lab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
                    if lab is not None:
                        ari = C.partition_metrics(g, lab, truth)["ari"]
                row["ari_edge_cut"] = ari
                rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1, 2, 3))
    if len(sys.argv) > 1 and sys.argv[1] == "tiny":
        specs = specs[:1]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / ("t01_phase_diagram%s.csv" % ("_tiny" if len(sys.argv) > 1 else "")), rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
