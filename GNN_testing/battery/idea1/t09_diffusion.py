"""T1.09 is it just diffusion? damp(k) -> diffusion as k->1 (c^2 = 0.5 so damp(k=1) == diffusion eta=0.5)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

T = 500; R = 8
KS = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0]
CS = [np.sqrt(0.5), 0.5]

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rd = C.run_diff(g, "mean", u0, 0.5, T)
    sd = C.edge_score(rd.X, g, rd.fail_step)
    ld = C.partitions(rd.X, g, 2, rd.fail_step, which=("edge_cut",))["edge_cut"]
    evd = C.evaluate_partitions(rd.X, g, truth, rd.fail_step)
    rows = []
    for c in CS:
        for k in KS:
            sch, ps = scheme_for(k, 0)
            run = erun(g, sch, "mean", u0, c, T, ps)
            row = dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], c=c, k=k, fail_frac=float(run.fail.mean()))
            if run.ok.any():
                s = C.traj_summary(run); row["ac_ratio_end"] = s["ac_final"]
                sc = C.edge_score(run.X, g, run.fail_step)
                lab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
                ev = C.evaluate_partitions(run.X, g, truth, run.fail_step)
                row.update(ari=ev["ari_edge_cut"], ari_lp=ev["ari_edge_cut_lp"], edge_margin=ev["edge_margin"], bridge_auc=ev["bridge_auc"],
                           ari_diff=evd["ari_edge_cut"], edge_margin_diff=evd["edge_margin"], bridge_auc_diff=evd["bridge_auc"])
                if sc is not None and np.std(sc) > 0 and np.std(sd) > 0:
                    row["spearman_vs_diff"] = float(spearmanr(sc, sd)[0])
                if lab is not None and ld is not None:
                    row["ari_agree_vs_diff"] = float(adjusted_rand_score(ld, lab))
                row["maxabs_traj_diff"] = float(np.abs(run.X - rd.X).max()) if k == 1.0 else np.nan
            rows.append(row)
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3], bridges=2) + core_specs([0, 1, 2, 3], bridges=4)
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0)
        import pandas as pd; print(pd.DataFrame(r).drop(columns=["graph"]).round(3).to_string()); sys.exit()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t09_diffusion.csv", rows)
    print(len(rows))
