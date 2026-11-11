"""T5.12 partition + hold on raw trajectories.  Configs (all stable): d4:mean eps in {-0.3,-0.1,0.1,0.3,1.0}x c in {0.5} (+c=1 for
eps<=0.3), t4neg eps in {-0.1,-0.05} c in {0.5,1.0} and wave+mean c=0.5/1, damp(.1) c=1, diffusion, spectral.
All five readouts, edge_margin/bridge_auc, hold (min_frac=0.1) on edge_cut / edge_cut_lp / fft_lowmode labels, ARI over time
(window 100) -> best-window ARI and its end time.  Core zoo, dev seeds 0-3 (REDUCED from 0-7), R=8, T=500.  Output results/t12_partition_hold.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 500


def configs():
    L = []
    for c in (0.5, 1.0):
        L.append((f"wave c={c:g}", C.make_runner("wave", "mean", c, steps=T), dict(family="wave", c=c, eps=0.0)))
        for e in (-0.3, -0.1, 0.1, 0.3, 1.0):
            if c == 1.0 and e > 0.3:
                continue
            L.append((f"d4 eps={e:g} c={c:g}", C.make_runner("wave", "d4:mean", c, pf=dict(eps=e), steps=T), dict(family="d4", c=c, eps=e)))
        for e in (-0.1, -0.05):
            L.append((f"t4neg eps={e:g} c={c:g}", C.make_runner("t4neg", "mean", c, ps=dict(eps=e), steps=T), dict(family="t4neg", c=c, eps=e)))
    L.append(("damp(k=.1) c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=T), dict(family="damp", c=1.0, eps=np.nan)))
    L.append(("diffusion", C.make_diff_runner("mean", 0.5, steps=T), dict(family="diff", c=np.nan, eps=np.nan)))
    return L


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    rows = []
    for label, runner, ex in configs():
        r = runner(g, u0)
        row = eval_run(r, g, truth, meta, label, extra=ex)
        if r.ok.any():
            ao = C.ari_over_time(r.X, g, truth, window=100, stride=25, fail_step=r.fail_step)
            a = np.asarray(ao["ari"], float)
            if np.isfinite(a).any():
                k = int(np.nanargmax(a))
                row["best_window_ari"] = float(a[k]); row["best_window_end"] = float(np.asarray(ao["t_end"])[k])
                row["last_window_ari"] = float(a[-1])
        rows.append(row)
    rows.append(dict(spectral_row(g, truth, meta), family="spectral"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t12_small.csv" if small else "t12_partition_hold.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "edge_margin", "hold_edge_cut", "hold_fft_lowmode", "best_window_ari"]].mean().round(3).to_string())
