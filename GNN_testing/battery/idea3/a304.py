import i3
from i3 import C, np
import pandas as pd
from scipy.stats import spearmanr
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 800)
df = pd.read_csv(i3.RES / "t304_ccrit.csv")
print("n rows", len(df), "graphs", df.graph.nunique())
df["emp_state"] = np.where(df.c_crit_emp.isna(), "unstable_at_c_lo", np.where(np.isinf(df.c_crit_emp), "stable_to_c_hi", "finite"))
print(df.groupby(["force"]).emp_state.value_counts().unstack(fill_value=0).to_string())
print("\nmedian c_crit by force x gkind: emp | pred(sym, random state) | pred(sym, trajectory min) | eig(trajectory min)")
t = df.groupby(["force", "gkind"]).agg(emp=("c_crit_emp", "median"), pred_u0=("c_pred_u0", "median"), pred_traj=("c_pred_traj", "median"), eig_traj=("c_eig_traj", "median"),
                                       kink=("kink_traj", "max"), nonmono=("non_monotone", "mean"), dsym=("dsym_traj", "mean")).round(3)
print(t.to_string())
fin = df[np.isfinite(df.c_crit_emp) & (df.c_crit_emp > 0)].copy()
fin["lr_sym"] = np.log(fin.c_crit_emp / fin.c_pred_traj); fin["lr_eig"] = np.log(fin.c_crit_emp / fin.c_eig_traj.clip(1e-3))
print("\nlog(emp/pred): mean over rows with finite emp (negative => pred too optimistic)")
rows = []
for f, s in fin.groupby("force"):
    rows.append(dict(force=f, n=len(s), lr_sym=s.lr_sym.mean(), lr_sym_med=s.lr_sym.median(), emp_med=s.c_crit_emp.median(), pred_med=s.c_pred_traj.median(),
                     rho=spearmanr(s.c_crit_emp, s.c_pred_traj)[0] if len(s) > 3 and s.c_pred_traj.std() > 0 else np.nan,
                     eig_lt_emp=(s.c_eig_traj < 0.5 * s.c_crit_emp).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
m = df[df.force == "mean"]
print("\nmean force: emp vs pred ratio", (m.c_crit_emp / m.c_pred_u0).describe().round(4).to_string())
# prediction of the binary 'unstable at every c tested' from the linear criterion
df["lin_unstable"] = (df.eig_sym_max_traj.fillna(df.eig_sym_max_u0) > 1e-8)
df["eig_unstable"] = (df.c_eig_traj.fillna(df.c_eig_u0) < 0.02)
ct = pd.crosstab(df.emp_state == "unstable_at_c_lo", [df.lin_unstable, df.eig_unstable])
print("\ncrosstab: emp unstable_at_c_lo (rows) vs [anti-restoring sym eig>0, true-eig c_crit<0.02] (cols)"); print(ct.to_string())
print("\nforces with non-monotone c_crit:", df[df.non_monotone == True].force.value_counts().to_dict())
