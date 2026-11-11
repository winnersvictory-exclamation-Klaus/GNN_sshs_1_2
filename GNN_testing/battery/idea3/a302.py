import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = pd.read_csv(i3.RES / "t302_property_table.csv")
print("rows", len(df), "errors:", df.get("error", pd.Series(dtype=object)).notna().sum())
cols = ["oddness", "homog_rel", "const_defect", "momentum_rel", "dsym_rel", "eig_sym_max", "lam_max", "c_crit_pred", "c_crit_eig", "jac_kink"]
for st in ("random", "mid", "near_consensus"):
    d = df[df.state == st]
    print(f"\n== state={st}: median over 4 graph kinds x 3 seeds")
    print(d.groupby("force")[cols].median().round(4).to_string())
print("\n== random state, by graph kind: dsym_rel / eig_sym_max / c_crit_eig (median over seeds)")
d = df[df.state == "random"]
print(d.pivot_table(index="force", columns="gkind", values=["dsym_rel", "eig_sym_max", "c_crit_eig"], aggfunc="median").round(3).to_string())
print("\nfraction of rows with kink>1e-3:", (df.jac_kink > 1e-3).mean().round(3), "by force:", df.groupby("force").jac_kink.apply(lambda x: (x > 1e-3).mean()).round(2).to_dict())
