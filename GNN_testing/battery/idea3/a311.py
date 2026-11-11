import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t311_hold.csv")
print("graphs", df.graph.nunique())
for col in ("holdpos_truth_rms", "holdpos_truth_ac", "holdpos_edge_rms", "holdpos_edge_nofix", "hold_truth_rms", "hold_edge_rms"):
    df[col] = df[col].fillna(0.0)   # failed-everywhere -> no quasi-equilibrium
t = df.groupby(["scheme", "force"]).agg(fail=("fail_frac", "mean"), hp_truth=("holdpos_truth_rms", "mean"), hp_truth_ac=("holdpos_truth_ac", "mean"), hp_edge=("holdpos_edge_rms", "mean"),
                                        hp_edge_nofix=("holdpos_edge_nofix", "mean"), hold_truth=("hold_truth_rms", "mean"), hold_edge=("hold_edge_rms", "mean"), ac_final=("ac_final", "median"))
print(t.round(3).to_string())
for sch in ("wave", "damp", "diff"):
    out = C.paired_vs_baseline(df[df.scheme == sch], "holdpos_truth_rms", "force", "mean", pair=("graph", "seed"))
    print(f"-- {sch}: holdpos_truth_rms paired vs mean, Holm p<.05:"); print(out[out.p_holm < 0.05][["config", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))
