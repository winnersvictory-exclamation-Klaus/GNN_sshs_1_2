"""T4.15 negative controls.  Zoo `neg` (er, shuffled, random_regular d=3,4,6,8, watts_strogatz; no true partition), dev seeds 0-3 (28 graphs), R=8, T=500.
False-partition rate = fraction of graphs where the edge_cut / edge_cut_lp labels have all parts >= 10% of nodes AND quasi-equilibrium (hold>0) is
claimed on at least one realisation (mean over realisations of hold >0); degenerate-cut rate = minfrac<0.1."""
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
    dict(label="t6g(-.03,.003) c.5", scheme="t6g", ps=dict(eps=(-0.03, 0.003)), force="mean", c=0.5, w=(1.0,)),
]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    out = []
    for cfg in CONFIGS:
        r = i4.eval_cfg(g, None, meta, lam, u0, cfg, steps=500)
        r["kind"] = meta["kind"]
        out.append(r)
    return out


if __name__ == "__main__":
    specs = C.zoo("neg", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t415_negative_controls.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for nm in ("edge_cut", "edge_cut_lp"):
        df[f"degen_{nm}"] = (df[f"minfrac_{nm}"] < 0.1).astype(float)
    print(df.groupby("config", sort=False).agg(n=("graph", "count"), fail=("fail_frac", "mean"), false_ec=("false_edge_cut", "mean"), false_elp=("false_edge_cut_lp", "mean"),
          degen_ec=("degen_edge_cut", "mean"), degen_elp=("degen_edge_cut_lp", "mean"), hold_ec=("hold_edge_cut", "mean")).round(3).to_string())
    print(df.groupby(["kind", "config"], sort=False).false_edge_cut.mean().unstack(1).round(2).to_string())
