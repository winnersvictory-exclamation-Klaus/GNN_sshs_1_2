import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
df = pd.read_csv(i3.RES / "t306_localisation.csv")
print("fail rates (graph-level mean) for all forces by gkind, c=0.5/1.0")
print(df[df.thr == 5.0].groupby(["gkind", "force", "c"]).fail.mean().unstack(["gkind", "c"]).round(2).to_string())
for thr in (2.0, 5.0):
    d = df[(df.thr == thr) & df.node.notna()].copy()
    d["at_end"] = (d.dist == 0).astype(float); d["near"] = (d.dist <= 1).astype(float)
    g = d.groupby(["gkind", "force", "c", "graph"]).agg(at_end=("at_end", "mean"), near=("near", "mean"), b0=("frac_nodes_d0", "first"), b1=("frac_nodes_le1", "first"), n=("dist", "size"), deg=("deg", "mean"), dist=("dist", "mean")).reset_index()
    g["lift0"] = g.at_end - g.b0; g["lift1"] = g.near - g.b1
    rows = []
    for (k, f, c), s in g.groupby(["gkind", "force", "c"]):
        if len(s) < 8 or f == "mean": continue
        m0 = C.boot_ci(s.lift0.values); m1 = C.boot_ci(s.lift1.values)
        w = C.paired_wilcoxon(s.near.values, s.b1.values)
        rows.append(dict(gkind=k, force=f, c=c, graphs=len(s), real=int(s.n.sum()), at_end=s.at_end.mean(), base0=s.b0.mean(), lift0=m0[0], lift0_lo=m0[1], lift0_hi=m0[2],
                         near=s.near.mean(), base1=s.b1.mean(), lift1=m1[0], lift1_lo=m1[1], lift1_hi=m1[2], p=w["p"], mean_dist=s["dist"].mean(), deg=s.deg.mean()))
    T = pd.DataFrame(rows)
    T["p_holm"] = C.holm(T.p.fillna(1).values) if len(T) else []
    print(f"\n== first node exceeding |u|>{thr}: share at a bridge endpoint (d=0) / within 1 hop vs uniform-node baseline (paired by graph, Holm over all rows)")
    print(T.round(3).to_string(index=False))
    T.to_csv(i3.RES / f"t306_summary_thr{thr:g}.csv", index=False)
