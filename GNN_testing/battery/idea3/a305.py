import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = pd.read_csv(i3.RES / "t305_energy.csv")
A = df[df.form == "A"]
print("n graphs", df.graph.nunique())
def ci(x):
    g = x.groupby(level=0) if False else x
    m, lo, hi = C.boot_ci(x.values); return f"{m:.4f} [{lo:.4f},{hi:.4f}]"
print("== Form A: energy E=1/2||v||^2_D + c^2 Phi  (median-over-realisations per graph; mean over graphs w/ boot CI)")
for (f, p, c), s in A.groupby(["force", "params", "c"]):
    print(f"{f:12s} {p:16s} c={c:<4} drift_ratio={ci(s.drift_ratio)} osc={ci(s.osc)} slope/1000={ci(s.slope_per1000)} fail={s.fail_frac.mean():.2f}")
B = df[df.form == "B"]
print("\n== Form B: quadratic wave energy (C.energy) -- ratio end/start (median over graphs), log-rate per step")
t = B.groupby(["force", "c"]).agg(ratio=("energy_ratio_end", "median"), rate=("log_energy_rate", "median"), fail=("fail_frac", "mean"), vlen=("mean_valid_len", "mean")).round(5)
print(t.unstack("c").to_string())
