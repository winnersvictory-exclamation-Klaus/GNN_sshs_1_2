"""T4.11b (extension of T4.11, dev-seed selection of the time-series configs).  The stable t6 region found in T4.07 (e4<0) acts as a high-mode-selective
dissipation (gamma ~ e4 s^2/2 - e6 s^3 per step).  Scan 9 (e4,e6) points x c in {0.5,0.7,1.0,1.2,1.4}, core zoo dev seeds 0-3 (48 graphs), R=8, T=500;
metrics: fail/dead, ARI (all readouts), edge_margin, bridge_auc, hold (edge_cut), plus Spearman of the per-edge score with the diffusion score
('is it diffusion in disguise'), and the labels' ARI agreement with diffusion.  Baselines: wave c.5, diffusion(eta .5), damp(.1) c1, viscous(.2) c1.
The scan is the dev-seed selection set for the top-3 configurations; nothing here is held-out."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
from scipy.stats import spearmanr
C = i4.C

PAIRS = [(-0.003, 0.0), (-0.01, 0.0), (-0.01, 0.001), (-0.03, 0.0), (-0.03, 0.003), (-0.03, -0.003), (-0.1, 0.0), (-0.1, 0.01), (-1 / 12, 1 / 90)]
CS = (0.5, 0.7, 1.0, 1.2, 1.4)
BASE = [dict(label="wave c.5", force="mean", c=0.5, w=(1.0,)),
        dict(label="diffusion", kind="diff", force="mean", eta=0.5),
        dict(label="damp.1 c1", scheme="damp", ps=dict(k=0.1), force="mean", c=1.0, w=(1.0,)),
        dict(label="visc.2 c1", scheme="viscous", ps=dict(nu=0.2), force="mean", c=1.0, w=(1.0,))]
CONFIGS = BASE + [dict(label=f"t6g({e4:.4g},{e6:.4g}) c{c}", scheme="t6g", ps=dict(eps=(e4, e6)), force="mean", c=c, w=(1.0,)) for (e4, e6) in PAIRS for c in CS]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    rows = []
    dscore = None
    dlab = None
    for cfg in CONFIGS:
        row = i4.eval_cfg(g, truth, meta, lam, u0, cfg, steps=500)
        runner = i4.make_cfg_runner(cfg, lam, 500)
        run = runner(g, u0)
        sc = C.edge_score(run.X, g, run.fail_step) if run.ok.any() else None
        if cfg["label"] == "diffusion":
            dscore = sc
            dlab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
        row["_score"] = sc
        rows.append(row)
    for r in rows:
        sc = r.pop("_score")
        if sc is not None and dscore is not None:
            r["spearman_diff"] = float(spearmanr(sc, dscore)[0])
        else:
            r["spearman_diff"] = np.nan
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t411b_time_scan.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    cols = ["fail_frac", "dead_frac", "ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "bridge_auc", "hold_edge_cut", "spearman_diff"]
    print(df.groupby("config", sort=False)[cols].mean().round(3).to_string())
