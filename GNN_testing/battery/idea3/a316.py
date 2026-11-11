import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t316_amplitude.csv")
print("graphs", df.graph.nunique())
for col in ("fail_frac", "dev_mean_early", "dev_mean_mid", "dev_mean_late", "scale_inv_defect", "rms_late_over_scale", "rms_late_ref_over_scale"):
    print(f"== {col} (median over graphs)")
    print(df.groupby(["scheme", "force", "scale"])[col].median().unstack("scale").round(4).to_string())
