import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t307_param_sweeps.csv")
print("graphs", df.graph.nunique())
df["cs"] = df.scheme + np.where(df.c.notna(), "@" + df.c.astype(str), "")
df["pen"] = df.ari_edge_cut.fillna(0.0)
df["alive"] = (df.ac_final > 1e-6)
base = df[df.force == "mean"].groupby("cs").agg(fail=("fail_frac", "mean"), ari=("pen", "mean"), dead=("dead_frac", "mean"))
print("baseline mean force:"); print(base.round(3).to_string())
d = df[df.force != "mean"]
print("\n== fail_frac (mean over graphs) by force/param/value x scheme@c")
print(d.groupby(["force", "param", "value", "cs"])["fail_frac"].mean().unstack("cs").round(2).to_string())
print("\n== penalised ARI(edge_cut)")
print(d.groupby(["force", "param", "value", "cs"])["pen"].mean().unstack("cs").round(2).to_string())
print("\n== dead_frac")
print(d.groupby(["force", "param", "value", "cs"])["dead_frac"].mean().unstack("cs").round(2).to_string())
d2 = d.assign(em=d.edge_margin.where(d.alive))
print("\n== edge_margin median (alive only)")
print(d2.groupby(["force", "param", "value", "cs"])["em"].median().unstack("cs").round(2).to_string())
# best config per scheme by paired difference vs mean, count of configs tried
rows = []
for (f, p, v, cs), s in d.groupby(["force", "param", "value", "cs"]):
    b = df[(df.force == "mean") & (df.cs == cs)].set_index("graph").pen
    diff = (s.set_index("graph").pen - b).dropna()
    w = C.paired_wilcoxon(s.set_index("graph").pen.reindex(b.index).values, b.values)
    rows.append(dict(force=f, param=p, value=v, cs=cs, fail=s.fail_frac.mean(), ari=s.pen.mean(), d_ari=diff.mean(), p=w["p"]))
T = pd.DataFrame(rows); T["p_holm"] = C.holm(T.p.fillna(1).values)
print(f"\n== best 12 (config x scheme) by mean ARI gain vs mean force (n configs tested = {len(T)}), requiring fail<=0.05")
print(T[T.fail <= 0.05].sort_values("d_ari", ascending=False).head(12).round(3).to_string(index=False))
T.to_csv(i3.RES / "t307_summary.csv", index=False)
