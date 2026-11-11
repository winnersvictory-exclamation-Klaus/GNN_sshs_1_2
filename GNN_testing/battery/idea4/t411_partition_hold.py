"""T4.11 partition + hold for stable configs (mean base unless stated).  Core zoo, dev seeds 0-3 (48 graphs), R=8, T=500 (rx standard), all readouts,
edge_margin, bridge_auc, hold (min_frac=0.1 fix) of the edge_cut labels.  Baselines: wave c=.5 / c=1, damp(.1) c=1, diffusion(eta .5).
Series at c=0.5 and c=0.9*c_pred; time-series t6g at stable parameters (from T4.07) at c=0.5.  Paired (graph,seed) Wilcoxon vs wave c=0.5, Holm within metric."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C
W = i4.weights

CONFIGS = [
    dict(label="wave c.5", force="mean", c=0.5, w=(1.0,)),
    dict(label="wave c1", force="mean", c=1.0, w=(1.0,)),
    dict(label="damp.1 c1", scheme="damp", ps=dict(k=0.1), force="mean", c=1.0, w=(1.0,)),
    dict(label="diffusion", kind="diff", force="mean", eta=0.5),
    dict(label="hi_acc c.5", force="hi_acc:mean", c=0.5, w=W("hi_acc"), K=4),
    dict(label="hi_alt c.5", force="hi_alt:mean", c=0.5, w=W("hi_alt"), K=4),
    dict(label="hi_geo.2 c.5", force="hi_geo:mean", c=0.5, w=W("hi_geo", .2), K=4),
    dict(label="hi_geoneg c.5", force="hi_geoneg:mean", c=0.5, w=W("hi_geoneg", -.2), K=4),
    dict(label="hi_acc 0.9cp", force="hi_acc:mean", c=("cp", .9), w=W("hi_acc"), K=4),
    dict(label="hi_alt 0.9cp", force="hi_alt:mean", c=("cp", .9), w=W("hi_alt"), K=4),
    dict(label="hi_geoneg 0.9cp", force="hi_geoneg:mean", c=("cp", .9), w=W("hi_geoneg", -.2), K=4),
    dict(label="t6g(-.01,0) c.5", scheme="t6g", ps=dict(eps=(-0.01, 0.0)), force="mean", c=0.5, w=(1.0,)),
    dict(label="t6g(-.03,.003) c.5", scheme="t6g", ps=dict(eps=(-0.03, 0.003)), force="mean", c=0.5, w=(1.0,)),
    dict(label="t6g(-1/12,1/90) c.5", scheme="t6g", ps=dict(eps=(-1 / 12, 1 / 90)), force="mean", c=0.5, w=(1.0,)),
    dict(label="t6g(-.03,.003) c1", scheme="t6g", ps=dict(eps=(-0.03, 0.003)), force="mean", c=1.0, w=(1.0,)),
]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    return [i4.eval_cfg(g, truth, meta, lam, u0, cfg, steps=500) for cfg in CONFIGS]


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t411_partition_hold.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["fail_frac", "dead_frac", "ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "ari_timeavg_2means", "ari_traj_corr_spectral", "edge_margin", "bridge_auc", "hold_edge_cut"]
    print(df.groupby("config", sort=False)[cols].mean().round(3).to_string())
    for v in ("ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "bridge_auc", "hold_edge_cut"):
        print(v)
        print(C.paired_vs_baseline(df, v, "config", "wave c.5", pair=("graph", "seed")).round(4).to_string(index=False))
