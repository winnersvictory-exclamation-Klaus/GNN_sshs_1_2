"""T5.17 held-out statistics: seeds 100-115 x zoo 'core' (12 graphs/seed = 192 graphs), the dev-selected top-3 (results/top3.json)
vs baselines wave+mean c=0.5, damp(.1)+mean c=1, diffusion, spectral.  ARI (edge_cut, edge_cut_lp, fft_lowmode), edge_margin, hold, transmit_ratio /
det_auc (C.stimulus, amp 1, sign +, n0=24), fail_frac.  Paired Wilcoxon vs baseline (Holm within metric).  Output results/t17_heldout.csv.
NO selection is done on these seeds."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 500


def load_configs():
    top = json.load(open(RES / "top3.json"))
    L = []
    for t in top:
        L.append((t["label"], C.make_runner(t["scheme"], t["force"], t["c"], ps=t.get("ps"), pf=t.get("pf"), steps=T), dict(family="top3")))
    L += [("wave+mean c=0.5", C.make_runner("wave", "mean", 0.5, steps=T), dict(family="base")),
          ("damp(k=.1)+mean c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=T), dict(family="base")),
          ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=T), dict(family="base"))]
    return L


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng, truth=truth)
    rows = []
    for label, runner, ex in load_configs():
        r = runner(g, u0)
        row = eval_run(r, g, truth, meta, label, extra=ex)
        sr = C.stimulus(g, 24, runner.__class__(**{**runner.__dict__, "steps": 200}), amps=(1.0,), signs=(+1,), R=8, steps=200, seed=spec["seed"], det_trials=16)[0]
        row.update(transmit_ratio=sr["transmit_ratio"], det_auc=sr["det_auc"], far_peak=sr["far_peak"], stim_noise_gain=sr["noise_gain"], stim_arrival=sr["arrival"])
        rows.append(row)
    rows.append(dict(spectral_row(g, truth, meta), family="spectral"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("core", seeds=C.HELD_SEEDS[:1] if small else C.HELD_SEEDS)
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t17_small.csv" if small else "t17_heldout.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "ari_edge_cut", "edge_margin", "hold_edge_cut", "transmit_ratio", "det_auc"]].mean().round(3).to_string())
