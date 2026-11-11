"""T2.12 combination with linear damping.  vel_shrink alpha x k grid; limiter_damp vmax x k; saturate_visc kappa x nu (composites registered in
i2lib); controls = pure damp(k) and pure viscous(nu) (idea 1) to test whether the nonlinear part adds anything beyond the linear one.
Forces trim, owa, vw, median at c=1; cycle/clique/sbm bridges 2 seeds 0-2 (9 graphs); R=8, T=1000."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 1000
FORCES = ["trim", "owa", "vw", "median"]

def configs():
    cf = []
    for a in (0.3, 1, 3, 10, 30):
        for k in (0.0, 0.02, 0.05, 0.1, 0.2):
            cf.append(("vel_shrink", dict(alpha=a, k=k), "vel_shrink", a, k))
    for vm in (0.1, 0.3, 1, 3):
        for k in (0.0, 0.02, 0.05, 0.1, 0.2):
            cf.append(("limiter_damp", dict(vmax=vm, k=k), "limiter_damp", vm, k))
    for ka in (0.3, 1, 3, 10):
        for nu in (0.0, 0.05, 0.1, 0.2, 0.4):
            cf.append(("saturate_visc", dict(kappa=ka, nu=nu), "saturate_visc", ka, nu))
    for k in (0.02, 0.05, 0.1, 0.2):
        cf.append(("damp", dict(k=k), "damp", np.nan, k))
    for nu in (0.05, 0.1, 0.2, 0.4):
        cf.append(("viscous", dict(nu=nu), "viscous", np.nan, nu))
    cf.append(("wave", {}, "wave", np.nan, 0.0))
    return cf

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    rows = []
    for f in FORCES:
        for sch, ps, fam, p1, p2 in configs():
            run = C.run(g, sch, f, u0, 1.0, T, ps)
            row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], force=f, family=fam, scheme=sch, p1=p1, p2=p2)
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
    specs = L.graphs_dev(seeds=(0, 1, 2))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t12_combinations.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
