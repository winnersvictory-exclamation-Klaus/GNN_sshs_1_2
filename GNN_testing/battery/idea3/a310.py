import sys, i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 800)
fn = sys.argv[1] if len(sys.argv) > 1 else "t310_partition.csv"
df = pd.read_csv(i3.RES / fn)
RD = ["edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode"]
print("graphs", df.graph.nunique(), "rows", len(df))
sp = df[df.force == "spectral"]; print("spectral ARI", C.boot_ci(sp.ari_edge_cut.values))
d = df[df.force != "spectral"].copy()
d["alive"] = (d.ac_final > 1e-6) & (d.dead_frac < 0.5)
for r in RD: d[f"pen_{r}"] = d[f"ari_{r}"].fillna(0.0)
d["em_valid"] = d.edge_margin.where(d.alive)
d["best_pen"] = d[[f"pen_{r}" for r in RD]].max(axis=1)
rows = []
for (sch, f), s in d.groupby(["scheme", "force"]):
    row = dict(scheme=sch, force=f, n=len(s), fail=s.fail_frac.mean(), dead=s.dead_frac.mean())
    for r in RD:
        m, lo, hi = C.boot_ci(s[f"pen_{r}"].values); row[f"ARI_{r}"] = m
    m, lo, hi = C.boot_ci(s.pen_edge_cut.values); row["edge_cut_lo"], row["edge_cut_hi"] = lo, hi
    row["ARI_cond_edge_cut"] = s.ari_edge_cut.mean(); row["n_ari"] = s.ari_edge_cut.notna().sum()
    row["margin"] = s.em_valid.median(); row["bauc"] = s.bridge_auc.mean(); row["cond"] = s.cond_edge_cut.mean(); row["best_pen"] = s.best_pen.mean()
    rows.append(row)
T = pd.DataFrame(rows)
print(T.round(3).to_string())
T.to_csv(i3.RES / "t310_summary.csv", index=False)
print("\n== paired vs mean (same scheme), penalised ARI (failed -> 0), edge_cut and best-readout; Holm over forces")
for sch in ("wave", "damp", "diff"):
    for col in ("pen_edge_cut", "best_pen", "pen_fft_lowmode"):
        out = C.paired_vs_baseline(d[d.scheme == sch], col, "force", "mean", pair=("graph", "seed"))
        sig = out[(out.p_holm < 0.05)].sort_values("mean_diff", ascending=False)
        print(f"-- {sch} {col}: forces with Holm p<0.05 vs mean (mean_diff, CI):")
        print(sig[["config", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))
