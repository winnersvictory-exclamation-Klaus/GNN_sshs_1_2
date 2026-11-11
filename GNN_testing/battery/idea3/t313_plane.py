"""T3.13 partition-transmission plane (analysis of T3.10 + T3.12 raw rows, same dev graphs): ARI vs transmit vs noise gain per force; Pareto vs mean."""
import i3
from i3 import C, np
import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 900)
P = pd.read_csv(i3.RES / "t310_partition.csv"); S = pd.read_csv(i3.RES / "t312_stimulus.csv")
P = P[P.force != "spectral"].copy(); P["pen"] = P.ari_edge_cut.fillna(0.0); P["pen_lp"] = P.ari_edge_cut_lp.fillna(0.0)
sg = S[(S.amp == 1.0) & (S.sign == 1)][["graph", "force", "scheme", "far_peak", "transmit_ratio", "noise_gain", "det_auc", "arrival", "snr", "fail_frac"]].rename(columns={"fail_frac": "stim_fail"})
sg5 = S[(S.amp == 5.0) & (S.sign == 1)][["graph", "force", "scheme", "far_peak"]].rename(columns={"far_peak": "far_peak5"})
M = P.merge(sg, on=["graph", "force", "scheme"], how="inner").merge(sg5, on=["graph", "force", "scheme"], how="left")
print("merged graphs", M.graph.nunique(), "rows", len(M))
# restrict to seeds present in both
rows = []
for (sch, f), s in M.groupby(["scheme", "force"]):
    b = M[(M.scheme == sch) & (M.force == "mean")].set_index("graph")
    s = s.set_index("graph")
    def pd_(col):
        a = s[col]; bb = b[col].reindex(a.index)
        w = C.paired_wilcoxon(a.values, bb.values); ci = C.boot_ci((a - bb).dropna().values)
        return ci[0], ci[1], ci[2], w["p"]
    r = dict(scheme=sch, force=f, n=len(s), fail=s.fail_frac.mean(), ari=s.pen.mean(), ari_lp=s.pen_lp.mean(), transmit=s.transmit_ratio.mean(), far_peak=s.far_peak.mean(),
             noise_gain=s.noise_gain.mean(), det_auc=s.det_auc.mean())
    for col in ("pen", "transmit_ratio", "noise_gain", "far_peak"):
        m, lo, hi, p = pd_(col); r[f"d_{col}"] = m; r[f"lo_{col}"] = lo; r[f"hi_{col}"] = hi; r[f"p_{col}"] = p
    rows.append(r)
T = pd.DataFrame(rows)
for col in ("pen", "transmit_ratio", "noise_gain", "far_peak"):
    for sch in T.scheme.unique():
        idx = T.scheme == sch; mask = idx & (T.force != "mean")
        T.loc[mask, f"ph_{col}"] = C.holm(T.loc[mask, f"p_{col}"].fillna(1).values)
def verdict(r):
    if r.force == "mean": return "baseline"
    better_ari = r.d_pen > 0 and r.ph_pen < 0.05
    worse_ari = r.d_pen < 0 and r.ph_pen < 0.05
    better_tr = r.d_transmit_ratio > 0 and r.ph_transmit_ratio < 0.05
    worse_tr = r.d_transmit_ratio < 0 and r.ph_transmit_ratio < 0.05
    if better_ari and not worse_tr: return "ARI up, transmit not worse"
    if better_ari and worse_tr: return "ARI up, transmit down (trade-off)"
    if worse_ari and not better_tr: return "ARI down"
    if better_tr and not worse_ari: return "transmit up"
    return "no sig. difference" if not (worse_ari or worse_tr) else "worse"
T["verdict"] = T.apply(verdict, axis=1)
cols = ["scheme", "force", "n", "fail", "ari", "ari_lp", "transmit", "far_peak", "noise_gain", "det_auc", "d_pen", "ph_pen", "d_transmit_ratio", "ph_transmit_ratio", "d_noise_gain", "ph_noise_gain", "verdict"]
print(T[cols].round(3).to_string(index=False))
T.to_csv(i3.RES / "t313_plane_summary.csv", index=False)
