import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
C = L.C
d = pd.read_csv(Path(__file__).parent / "results" / "t02_do_no_harm.csv")
pd.set_option("display.width", 250)
print("n graphs", d.graph.nunique())
def ci(x):
    e, lo, hi = C.boot_ci(np.asarray(x.dropna())) if x.notna().any() else (np.nan,) * 3
    return f"{e:.3f} [{lo:.3f},{hi:.3f}]"
for c in (0.5, 1.0):
    print(f"\n=== c={c} (mean force) : mean over 12 graphs [95% boot CI] ===")
    s = d[d.c == c]
    rows = []
    for cfg, sub in s.groupby("config", sort=False):
        rows.append(dict(config=cfg, fail=sub.fail_frac.mean(), dev50=ci(sub.dev50), dev500=ci(sub.dev500), dCentroid=ci(sub.d_centroid), dNyq=ci(sub.d_nyq),
                         E_end=ci(sub.E_end), dARI_edge=ci(sub.d_ari_edge_cut), dARI_fft=ci(sub.d_ari_fft_lowmode), dARI_avg=ci(sub.d_ari_timeavg_2means)))
    print(pd.DataFrame(rows).to_string(index=False))
    w = s.drop_duplicates("graph")
    print("wave: fail %.2f E_end %.3f ARI edge %.3f fft %.3f avg %.3f centroid %.3f" % (w.wave_fail_frac.mean(), w.wave_E_end.mean(), w.wave_ari_edge_cut.mean(), w.wave_ari_fft_lowmode.mean(), w.wave_ari_timeavg_2means.mean(), w.wave_centroid.mean()))
    # paired tests on ARI (edge_cut) vs wave, Holm
    ps, names = [], []
    for cfg, sub in s.groupby("config", sort=False):
        r = C.paired_wilcoxon(sub.ari_edge_cut.values, sub.wave_ari_edge_cut.values); ps.append(r["p"]); names.append((cfg, r["n"], r["mean_diff"]))
    hp = C.holm(ps)
    print("paired Wilcoxon dARI(edge_cut) vs wave, Holm-corrected (sig only):")
    for (cfg, n, md), p in zip(names, hp):
        if p < 0.05: print(f"  {cfg}: n={n} mean_diff={md:+.3f} p_holm={p:.4f}")
