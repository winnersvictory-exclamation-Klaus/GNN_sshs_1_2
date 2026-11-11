"""T2.10 partition + quasi-equilibrium hold.  zoo `core` (cycle/clique/sbm x bridges 1,2,4,8; 24+24; wb 1) x dev seeds 0-2 (36 graphs; REDUCED from 4 seeds for time), R=8, T=500, c=1.
Configs: 6 schemes x 2 strengths (GRID6 idx 2,4; REDUCED from 3) x forces {mean,trim,owa,vw}  + baselines wave+mean, damp(k=.1)+mean, diffusion+mean, spectral.
All 5 readouts (ARI/NMI/cond/minfrac), edge_margin, bridge_auc, hold (min_frac fix) on edge_cut labels, ARI over time (window 100)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
from sklearn.metrics import adjusted_rand_score
C = L.C
rx = L.rx
T = 500
FORCES = ["mean", "trim", "owa", "vw"]
LEVELS = (("L2", 2), ("L4", 4))

def configs():
    cf = [("wave+mean", C.make_runner("wave", "mean", 1.0, steps=T)),
          ("damp0.1+mean", C.make_runner("damp", "mean", 1.0, ps={"k": 0.1}, steps=T)),
          ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=T))]
    for f in FORCES:
        for b in L.ORIG:
            for lv, i in LEVELS:
                ps = {L.PARAM[b][0]: L.GRID6[b][i]}
                cf.append((f"{b}[{L.PARAM[b][0]}={ps[L.PARAM[b][0]]:g}]+{f}", C.make_runner(b, f, 1.0, ps=ps, steps=T)))
    return cf

def evaluate(run, g, truth, spec_id):
    row = {}
    row.update(C.traj_summary(run))
    row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
    if not run.ok.any():
        return row
    row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))
    labs = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut", "edge_cut_lp"))
    for nm, lab in labs.items():
        if lab is None:
            continue
        h = C.hold(run.X, g, lab, fail_step=run.fail_step)
        row["hold_" + nm] = float(np.nanmean(h["hold"]))
        row["holdpos_" + nm] = float(np.nanmean(h["hold_pos"]))
        row["hold_rej_" + nm] = float(np.mean(h["rejected"])) if np.ndim(h["rejected"]) else float(h["rejected"])
    at = C.ari_over_time(run.X, g, truth, window=100, stride=100, fail_step=run.fail_step)
    for k, a in enumerate(at["ari"]):
        row[f"ari_w{k}"] = float(a)
    return row

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    rows = []
    for name, runner in configs():
        run = runner(g, u0)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=meta["n_bridges"], config=name)
        row.update(evaluate(run, g, truth, meta["id"]))
        rows.append(row)
    lab = rx.spectral_labels(g)
    row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=meta["n_bridges"], config="spectral",
               ari_edge_cut=adjusted_rand_score(truth, lab))
    rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:3])
    if len(sys.argv) > 1:
        specs = specs[:1]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / ("t10_partition%s.csv" % ("_tiny" if len(sys.argv) > 1 else "")), rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
