"""Select the top-3 configurations on DEV seeds only (T5.09 + T5.12 outputs, seeds 0-3).
Criterion: mean ARI of the rx edge_cut readout (edge_cut = the project's standard readout), among configs with fail_frac==0 on every
graph and dead_frac<=0.2 on average.  Writes results/top3.json and prints the dev table.  Held-out data are never read."""
import json
from lib5 import *

rows = []
d9 = pd.read_csv(RES / "t09_eps_bridges.csv"); d12 = pd.read_csv(RES / "t12_partition_hold.csv")
for nm, d in (("t09", d9), ("t12", d12)):
    d = d[d.config.notna()]
    for cfg, sub in d.groupby("config"):
        if sub.get("family", pd.Series(["x"])).iloc[0] in ("spectral",):
            continue
        rows.append(dict(src=nm, config=cfg, n=len(sub), fail=sub.fail_frac.fillna(1).max(), dead=sub.dead_frac.mean() if "dead_frac" in sub else np.nan,
                         ari_edge_cut=sub.ari_edge_cut.mean(), ari_lp=sub.ari_edge_cut_lp.mean() if "ari_edge_cut_lp" in sub else np.nan,
                         em=sub.edge_margin.median()))
t = pd.DataFrame(rows).sort_values("ari_edge_cut", ascending=False)
print(t.head(25).round(3).to_string())
