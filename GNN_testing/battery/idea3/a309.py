import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t309_mechanism.csv")
print("graphs", df.graph.nunique())
live = df[df.rms_end > 1e-6]     # exclude windows where the state is already at round-off
print("rows total", len(df), "live", len(live))
g = live.groupby(["force", "dyn", "win"]).agg(auc=("auc", "mean"), prec=("prec_at_nb", "mean"), auc_absx=("auc_absx", "mean"), auc_deg=("auc_deg", "mean"),
        flag_b=("flag_bridge", "mean"), flag_i=("flag_intra", "mean"), rho=("rho_flag_absx", "mean"), tau=("tau", "mean"), n=("auc", "size")).round(3)
print(g.to_string())
# chance level of precision@nb : nb / m
print("\nAUC (mean over graphs) with bootstrap CI, per force/dyn, window 20-100 (post-transient) and 100-300")
for w in ("20-100", "100-300"):
    for (f, dn), s in live[live.win == w].groupby(["force", "dyn"]):
        a = s.groupby("graph").auc.mean().values; b = s.groupby("graph").auc_absx.mean().values; dg = s.groupby("graph").auc_deg.mean().values
        print(f"{w:8s} {f:13s} {dn:5s} flagAUC={C.boot_ci(a)[0]:.3f} [{C.boot_ci(a)[1]:.3f},{C.boot_ci(a)[2]:.3f}] |x|AUC={b.mean():.3f} degAUC={dg.mean():.3f} n={len(a)}")
