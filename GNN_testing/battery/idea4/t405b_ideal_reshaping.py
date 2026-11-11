"""T4.05b (extension of T4.05): the best case for spectral reshaping.  g(l)=l-l^2/2 (hi_geo eps=-.5, K=2) maximises the stability limit
(c_pred ~2.83 vs 1.63 for plain wave) at 2 aggregations/step.  Also K=2 (1,-.3).  Same protocol as T4.05 (equal aggregation budget, c=0.9*c_pred and
c=0.5, 48 core graphs dev seeds 0-3); baseline plain wave / diffusion / damp."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import t405_equal_cost as T
import numpy as np, pandas as pd
C = i4.C
T.CONFIGS = [
    ("wave", "mean", {}, (1.0,)),
    ("geoneg-.5_K2", "series:mean", dict(w=(1.0, -0.5)), (1.0, -0.5)),
    ("geoneg-.3_K2", "series:mean", dict(w=(1.0, -0.3)), (1.0, -0.3)),
]

if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(T.job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t405b_ideal_reshaping.csv", rows)
    df = pd.DataFrame(rows)
    d = df[df.config != "spectral"]
    pd.set_option("display.width", 250)
    for pol in ("0.9cp", "c0.5"):
        sub = d[d.policy == pol]
        print("policy", pol)
        print(sub.groupby("config", sort=False).agg(n=("graph", "count"), c=("c", "mean"), reach_fft=("agg90_fft_lowmode", lambda x: x.notna().mean()), reach_elp=("agg90_edge_cut_lp", lambda x: x.notna().mean()),
              fft_B160=("ari_fft_lowmode_B160", "mean"), fft_B640=("ari_fft_lowmode_B640", "mean"), fft_B1280=("ari_fft_lowmode_B1280", "mean"),
              elp_B160=("ari_edge_cut_lp_B160", "mean"), elp_B640=("ari_edge_cut_lp_B640", "mean"), elp_B1280=("ari_edge_cut_lp_B1280", "mean")).round(3).to_string())
        for v in ("ari_edge_cut_lp_B160", "ari_edge_cut_lp_B640", "ari_fft_lowmode_B640", "ari_fft_lowmode_B1280"):
            print(v); print(C.paired_vs_baseline(sub, v, "config", "wave", pair=("graph", "seed")).round(4).to_string(index=False))
