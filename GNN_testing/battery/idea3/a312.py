import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t312_stimulus.csv")
if "error" in df.columns: print("errors:", df.error.notna().sum())
print("graphs", df.graph.nunique())
d = df.copy()
t = d.groupby(["scheme", "force", "amp"]).agg(fail=("fail_frac", "mean"), far=("far_peak", "mean"), tr=("transmit_ratio", "mean"), arrival=("arrival", "mean"), snr=("snr", "mean"),
                                             ng=("noise_gain", "mean"), auc=("det_auc", "mean"), n_ok=("n_ok", "mean"))
print("== amp=1 summary"); print(t.xs(1.0, level="amp").round(3).to_string())
# sign asymmetry: far_peak at +amp vs -amp
p = d.pivot_table(index=["scheme", "force", "graph", "amp"], columns="sign", values="far_peak").reset_index()
p["asym"] = np.log((p[1] + 1e-12) / (p[-1] + 1e-12))
a = p.groupby(["scheme", "force", "amp"]).asym.agg(["mean", "std"]).round(3)
print("\n== sign asymmetry log(far_peak(+)/far_peak(-)) (0 = symmetric) amp=5 and 0.2"); print(a.xs(5.0, level="amp").to_string()); print(a.xs(0.2, level="amp").to_string())
# nonlinearity: far_peak per unit amplitude
s = d[d.sign == 1].groupby(["scheme", "force", "amp"]).far_peak.mean().unstack("amp")
s["gain_5_over_0.2"] = (s[5.0] / 5) / (s[0.2] / 0.2)
print("\n== per-unit amplitude response ratio far_peak(5)/5 over far_peak(0.2)/0.2 (1 = linear)"); print(s["gain_5_over_0.2"].unstack("scheme").round(3).to_string())
for sch in ("damp", "diff"):
    for col in ("far_peak", "transmit_ratio", "det_auc", "noise_gain"):
        sub = d[(d.scheme == sch) & (d.amp == 1.0) & (d.sign == 1)]
        out = C.paired_vs_baseline(sub, col, "force", "mean", pair=("graph",))
        print(f"-- {sch} amp=1 +: {col} vs mean, Holm p<.05"); print(out[out.p_holm < 0.05][["config", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))
