import i3
from i3 import C, np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = pd.read_csv(i3.RES / "t303_predictors.csv")
print("rows", len(df), "graphs", df.graph.nunique())
df["fail_bin"] = (df.fail_frac >= 0.5).astype(int)
print("\n== fail_frac by force, c (mean over graphs, boot CI)")
for c in (0.5, 1.0):
    for f, sub in df[df.c == c].groupby("force"):
        m, lo, hi = C.boot_ci(sub.groupby("graph").fail_frac.mean().values)
        print(f"c={c} {f:14s} fail={m:.3f} [{lo:.3f},{hi:.3f}] ac_fail={sub.ac_fail_frac.mean():.2f} dc_only={sub.dc_only_frac.mean():.2f} growth_med={sub.growth_rate.median():.4f}")
P = ["oddness", "momentum_rel", "dsym_rel", "eig_sym_max", "lam_max", "homog_rel"]
rows = []
for tag in ("u0", "mid"):
    for c in (0.5, 1.0):
        d = df[df.c == c]
        u = d.groupby("force")[[f"{p}_{tag}" for p in P] + [f"stiff_{tag}", "fail_frac", "growth_rate"]].mean()
        for p in P + ["stiff"]:
            col = f"{p}_{tag}"
            r1 = spearmanr(u[col], u.fail_frac)[0]; r2 = spearmanr(u[col], u.growth_rate, nan_policy="omit")[0]
            rows.append(dict(state=tag, c=c, prop=p, rho_fail_forcelevel=r1, rho_growth_forcelevel=r2,
                             rho_fail_rowlevel=spearmanr(d[col], d.fail_frac)[0]))
print("\n== Spearman of property with outcome (force-level n=%d; row-level n=%d)" % (df.force.nunique(), len(df)//2))
print(pd.DataFrame(rows).round(3).to_string())

print("\n== leave-one-force-out logistic AUC for fail>=0.5 (rows: force x graph x c)")
def lofo(cols, d):
    X = np.log1p(np.abs(d[cols].fillna(0).values)); y = d.fail_bin.values; g = d.force.values
    pred = np.zeros(len(d))
    for f in np.unique(g):
        tr = g != f
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), y[tr])
        pred[~tr] = m.predict_proba(sc.transform(X[~tr]))[:, 1]
    return roc_auc_score(y, pred), pred
for tag in ("u0", "mid"):
    sets = {"oddness": [f"oddness_{tag}"], "momentum": [f"momentum_rel_{tag}"], "dsym": [f"dsym_rel_{tag}"], "anti_restoring": [f"eig_sym_max_{tag}"],
            "stiffness c/c_pred": [f"stiff_{tag}"], "c only": ["c"],
            "all5": [f"{p}_{tag}" for p in ("oddness", "momentum_rel", "dsym_rel", "eig_sym_max")] + [f"stiff_{tag}"]}
    for k, cols in sets.items():
        d = df.copy()
        a, _ = lofo(cols, d)
        # in-sample single-feature univariate AUC (no model)
        uni = roc_auc_score(d.fail_bin, np.abs(d[cols[0]].fillna(0))) if len(cols) == 1 else np.nan
        print(f"{tag:4s} {k:20s} LOFO AUC={a:.3f}  univariate AUC={uni:.3f}")
d = df[df.fail_bin.notna()]
print("\n== force-level table (mean over graphs) with u0 properties, c=0.5")
t = df[df.c == 0.5].groupby("force")[["fail_frac", "oddness_u0", "momentum_rel_u0", "dsym_rel_u0", "eig_sym_max_u0", "stiff_u0", "eig_sym_max_mid"]].mean().sort_values("fail_frac")
print(t.round(3).to_string())

print("\n== paired comparisons (by graph) ; Wilcoxon two-sided, Holm within family")
for c in (0.5, 1.0):
    for base, others in (("owa", ["odd_owa", "owa_mc"]), ("vw", ["odd_vw", "vw_mc"]), ("trim", ["trim_mc"])):
        sub = df[(df.c == c) & df.force.isin([base] + others)]
        out = C.paired_vs_baseline(sub, "fail_frac", "force", base, pair=("graph", "seed"))
        print(f"c={c} baseline={base}"); print(out.round(4).to_string())
for c in (0.5, 1.0):
    for f in ("owa", "owa_mc", "odd_owa"):
        s = df[(df.c == c) & (df.force == f)]
        print(f"c={c} {f}: fail_frac={s.fail_frac.mean():.3f}  mean_fail_step={s.mean_fail_step.mean():.1f}  ac_fail={s.ac_fail_frac.mean():.3f} dc_only={s.dc_only_frac.mean():.3f}")
