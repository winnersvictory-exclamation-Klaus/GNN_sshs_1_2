"""T2.17 held-out statistics.  Top-3 dev-selected configs (results/selected.json entries flagged top3) + baselines on held-out seeds 100-115,
zoo `core` (192 graphs), R=8, T=500.  Metrics: fail, dead, ARI (5 readouts), edge_margin, bridge_auc, hold (min_frac fix), transmit_ratio and
det_auc at amp=1 (stimulus, sigma .05).  Paired Wilcoxon + Holm vs each baseline."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
from sklearn.metrics import adjusted_rand_score
C = L.C; rx = L.rx
T = 500
RES = Path(__file__).parent / "results"

def runners():
    sel = [s for s in json.load(open(RES / "selected.json")) if s.get("top3")]
    cf = [("wave+mean", C.make_runner("wave", "mean", 1.0, steps=T)), ("damp0.1+mean", C.make_runner("damp", "mean", 1.0, ps={"k": 0.1}, steps=T)),
          ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=T))]
    for s in sel:
        cf.append((s["name"], C.make_runner(s["scheme"], s["force"], s["c"], ps=s["ps"], steps=T)))
    return cf

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, 8)
    n0 = spec["sizes"][0] if "sizes" in spec else 24
    rows = []
    for name, runner in runners():
        run = runner(g, u0)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=meta["n_bridges"], config=name)
        row.update(C.traj_summary(run)); row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
        if run.ok.any():
            row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))
            lab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
            if lab is not None:
                h = C.hold(run.X, g, lab, fail_step=run.fail_step); row["hold_rec"] = float(np.nanmean(h["hold"]))
        st = C.stimulus(g, n0, runner, amps=(1.0,), signs=(+1,), sigma=0.05, R=8, steps=200, seed=spec["seed"], det_trials=16)[0]
        for k in ("far_peak", "transmit_ratio", "arrival", "snr", "noise_gain", "det_auc", "fail_frac"):
            row["stim_" + k] = st[k]
        rows.append(row)
    rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], bridges=meta["n_bridges"], config="spectral",
                     ari_edge_cut=adjusted_rand_score(truth, rx.spectral_labels(g))))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = C.zoo("core", seeds=C.HELD_SEEDS)
    out = C.pmap(job, specs, procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(RES / "t17_heldout.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
