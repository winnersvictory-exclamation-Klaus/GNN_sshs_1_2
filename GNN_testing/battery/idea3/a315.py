import i3
from i3 import C, np
import pandas as pd
from scipy.stats import spearmanr
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t315_degree.csv")
df["pen"] = df.ari_edge_cut.fillna(0.0)
print("graphs", df.graph.nunique())
for sch in ("wave", "damp", "diff"):
    d = df[df.scheme == sch]
    print(f"== {sch}: fail_frac  random_regular d=3,4,6,8 | sbm_het sigma=0,0.3,0.6,1")
    rr = d[d.gkind == "random_regular"].pivot_table(index="force", columns="d", values="fail_frac")
    hh = d[d.gkind == "sbm_het"].pivot_table(index="force", columns="sigma", values="fail_frac")
    print(pd.concat([rr, hh], axis=1, keys=["rr_d", "het_sigma"]).round(2).to_string())
    print(f"== {sch}: penalised ARI on sbm_het (sigma)")
    print(d[d.gkind == "sbm_het"].pivot_table(index="force", columns="sigma", values="pen").round(2).to_string())
    print("degree-CV by sigma:", d[d.gkind == "sbm_het"].groupby("sigma").deg_cv.mean().round(3).to_dict())
    # relation fail vs low degree for rank forces
    for f in ("trim", "median", "symowa_med", "huber_local"):
        s = d[(d.force == f)]
        print(f"   {f}: Spearman(fail, deg_cv) over all graphs={spearmanr(s.fail_frac, s.deg_cv)[0]:.2f}; Spearman(fail, frac_deg<=3)={spearmanr(s.fail_frac, s.frac_deg_le3)[0]:.2f}")
