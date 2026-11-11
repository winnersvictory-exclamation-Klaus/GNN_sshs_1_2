"""T2.07 minimal rescue + alive/dead Pareto.  For each (graph kind, force, scheme) scan a 10-value strength grid (weak->strong) at c=1,
T=2000, R=8, graphs = kind x bridges 2 x seeds 0-2; 'rescue' = fail_frac == 0 on all 3 seeds.  At the minimal rescuing value m and at
2x / 4x strength (parameter scaled in the strong direction; energy kmax capped at 1) record alive/dead metrics.
Control scheme 'damp' (idea 1) included with its own k grid.  Output: rows type=scan (one per graph x scheme x force x value) and
type=rescue (one per graph x scheme x force x factor)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 2000
L.PARAM["damp"] = ("k", +1)
L.GRID10["damp"] = [0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
SCHEMES = list(L.ORIG) + ["damp"]
SEEDS = (0, 1, 2)

def job(args):
    kind, force = args
    c = 1.0
    specs = L.graphs_dev(kinds=(kind,), seeds=SEEDS)
    G = []
    for sp in specs:
        g, truth, meta = C.make_graph(sp)
        G.append((sp, g, truth, meta, L.make_u0(sp, g, 8)))
    rows = []
    for sch in SCHEMES:
        pn = L.PARAM[sch][0]
        grid = L.GRID10[sch]
        ff = np.zeros((len(grid), len(G)))
        for i, v in enumerate(grid):
            for j, (sp, g, truth, meta, u0) in enumerate(G):
                ff[i, j] = C.fail_frac(g, sch, force, u0, c, T, {pn: v})
                rows.append(dict(type="scan", kind=kind, force=force, scheme=sch, graph=meta["id"], seed=sp["seed"], idx=i, param=v, fail_frac=ff[i, j]))
        okidx = [i for i in range(len(grid)) if (ff[i] == 0).all()]
        m = okidx[0] if okidx else -1
        # baseline wave (no scheme)
        if sch == SCHEMES[0]:
            for j, (sp, g, truth, meta, u0) in enumerate(G):
                rows.append(dict(type="scan", kind=kind, force=force, scheme="wave", graph=meta["id"], seed=sp["seed"], idx=-1, param=np.nan,
                                 fail_frac=C.fail_frac(g, "wave", force, u0, c, T)))
        if m < 0:
            continue
        for fac in (1, 2, 4):
            v = L.scaled_strength(sch, grid[m], fac)
            if sch == "energy":
                v = min(v, 1.0)
            for j, (sp, g, truth, meta, u0) in enumerate(G):
                run = C.run(g, sch, force, u0, c, T, {pn: v})
                row = dict(type="rescue", kind=kind, force=force, scheme=sch, graph=meta["id"], seed=sp["seed"], factor=fac, idx=m, param=v,
                           min_param=grid[m], min_idx=m, n_minok=len(okidx))
                row.update(C.traj_summary(run))
                row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
                row["alive_frac"] = float(((run.ok) & (L.lifetime(run.X, g, run.fail_step) >= T - 1)).mean())
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
    jobs = [(k, f) for k in ("cycle", "clique", "sbm") for f in L.FORCES10]
    out = C.pmap(job, jobs, procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t07_minimal_rescue.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
