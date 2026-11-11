import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
df = pd.read_csv(i3.RES / "t308_discontinuity.csv")
A = df[df.part == "A"]
print("== A: twin divergence (perturbation 1e-10 rms; divergence = rms diff > 1e-6), steps<=500")
t = A.groupby(["scheme", "force"]).agg(frac_div=("div_step", lambda x: (x >= 0).mean()), med_div_step=("div_step", lambda x: np.median(x[x >= 0]) if (x >= 0).any() else np.nan),
                                       growth_med=("growth", "median"), growth_max=("growth", "max"), failed=("failed", "mean"), rel_end=("rel_d_end", "median")).round(3)
print(t.to_string())
B = df[df.part == "B"]
print("\n== B: max local Lipschitz ratio ||F(u+d)-F(u)||/||d|| over 64 pairs x graphs")
print(B.groupby(["force", "delta"]).lip_max.max().unstack().round(2).to_string())
Cc = df[df.part == "C"]
print("\n== C: neighbour-order permutation defect (max abs), by weighted? and init")
print(Cc.groupby(["force", "weighted", "init"]).perm_defect.max().unstack(["weighted", "init"]).to_string())
D = df[df.part == "D"]
print("\n== D: init effect (fail_frac)  uniform / gauss / rademacher")
print(D.groupby(["scheme", "force", "init"]).fail_frac.mean().unstack("init").round(2).to_string())
print(D.groupby(["scheme", "force", "init"]).ac_final.median().unstack("init").round(4).to_string())
