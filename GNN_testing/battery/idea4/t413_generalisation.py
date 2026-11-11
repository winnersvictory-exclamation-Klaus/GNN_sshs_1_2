"""T4.13 generalisation.  Zoo `wide` (15/seed) + `scale` (12/seed; n=48..384), dev seeds 0-1, top configs (selected on dev in T4.04/05/11) vs baselines.
R=8, T=500.  Reports ARI (edge_cut_lp, fft_lowmode), edge_margin, bridge_auc, fail_frac per graph kind and size."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C
W = i4.weights
CONFIGS = [
    dict(label="wave c.5", force="mean", c=0.5, w=(1.0,)),
    dict(label="damp.1 c1", scheme="damp", ps=dict(k=0.1), force="mean", c=1.0, w=(1.0,)),
    dict(label="diffusion", kind="diff", force="mean", eta=0.5),
    dict(label="hi_geoneg c.5", force="hi_geoneg:mean", c=0.5, w=W("hi_geoneg", -.2), K=4),
    dict(label="hi_acc c.5", force="hi_acc:mean", c=0.5, w=W("hi_acc"), K=4),
    dict(label="hi_alt c.5", force="hi_alt:mean", c=0.5, w=W("hi_alt"), K=4),
]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    out = []
    for cfg in CONFIGS:
        r = i4.eval_cfg(g, truth, meta, lam, u0, cfg, steps=500)
        r["kind"] = meta["kind"]
        out.append(r)
    return out


if __name__ == "__main__":
    specs = C.zoo("wide", seeds=[0, 1]) + C.zoo("scale", seeds=[0, 1])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t413_generalisation.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["fail_frac", "ari_edge_cut_lp", "ari_fft_lowmode", "edge_margin", "bridge_auc"]
    print(df.groupby("config", sort=False)[cols].mean().round(3).to_string())
    print(df.groupby(["kind", "config"], sort=False).ari_edge_cut_lp.mean().unstack(1).round(2).to_string())
    print(df.groupby(["n", "config"], sort=False).ari_edge_cut_lp.mean().unstack(1).round(2).to_string())
    for v in ("ari_edge_cut_lp", "ari_fft_lowmode", "bridge_auc"):
        print(v)
        print(C.paired_vs_baseline(df, v, "config", "wave c.5", pair=("graph", "seed")).round(4).to_string(index=False))
