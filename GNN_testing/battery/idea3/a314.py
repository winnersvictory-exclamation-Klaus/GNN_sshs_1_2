import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t314_confounders.csv")
print("graphs", df.graph.nunique())
df["pen"] = df.ari_edge_cut.fillna(0.0); df["pen_tm"] = df.ari_timeavg_2means.fillna(0.0)
for sch in ("damp", "diff"):
    d = df[df.scheme == sch]
    p = d.pivot_table(index="force", columns="cond", values="pen").round(3)
    print(f"== {sch}: penalised ARI(edge_cut) by condition"); print(p.to_string())
    p2 = d.pivot_table(index="force", columns="cond", values="pen_tm").round(3)
    print(f"== {sch}: penalised ARI(timeavg_2means)"); print(p2.to_string())
    print(f"== {sch}: fail_frac"); print(d.pivot_table(index="force", columns="cond", values="fail_frac").round(2).to_string())
    # degradation relative to clean, paired by graph; compare to mean's degradation
    rows = []
    for cond in ("stuck5", "noisy5", "bridge0.05", "bridge0.2"):
        for f in d.force.unique():
            a = d[(d.force == f) & (d.cond == cond)].set_index("graph").pen; b = d[(d.force == f) & (d.cond == "clean")].set_index("graph").pen
            m1 = d[(d.force == "mean") & (d.cond == cond)].set_index("graph").pen; m0 = d[(d.force == "mean") & (d.cond == "clean")].set_index("graph").pen
            deg_f = (a - b); deg_m = (m1 - m0)
            ci = C.boot_ci((deg_f - deg_m).dropna().values); w = C.paired_wilcoxon(deg_f.values, deg_m.reindex(deg_f.index).values)
            rows.append(dict(scheme=sch, cond=cond, force=f, d_force=deg_f.mean(), d_mean=deg_m.mean(), diff=ci[0], lo=ci[1], hi=ci[2], p=w["p"]))
    T = pd.DataFrame(rows); T["p_holm"] = C.holm(T.p.fillna(1).values)
    print(f"== {sch}: degradation (cond - clean) of force minus degradation of mean; positive = more robust than mean; Holm over {len(T)}")
    print(T[T.force != "mean"].sort_values("diff", ascending=False).head(10).round(3).to_string(index=False))
    T.to_csv(i3.RES / f"t314_summary_{sch}.csv", index=False)
