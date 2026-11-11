"""T5.12b (my addition, motivated by T5.12 dev result: t4neg c=1 gives raw-wave ARI/hold comparable to diffusion).
Fine map of the time term with eps<0: t4neg eps in {-0.2,-0.15,-0.1,-0.075,-0.05,-0.025} x c in {0.75,1,1.2,1.4}, plus controls:
viscous (idea-1 scheme; damping ~ lambda, vs t4neg ~ lambda^2) nu in {0.05,0.1,0.2,0.4,0.8} at c=1 and c=0.5, damp k in {0.02,0.1} c=1,
wave c=1, diffusion, spectral.  Core zoo, dev seeds 0-3, R=8, T=500.  All readouts, edge_margin, hold, alive (ac_final, dead).
Output results/t12b_t4neg_map.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 500
EPS = [-0.2, -0.15, -0.1, -0.075, -0.05, -0.025]
CS = [0.75, 1.0, 1.2, 1.4]


def configs():
    L = []
    for e in EPS:
        for c in CS:
            L.append((f"t4neg eps={e:g} c={c:g}", C.make_runner("t4neg", "mean", c, ps=dict(eps=e), steps=T), dict(family="t4neg", c=c, eps=e)))
    for nu in (0.05, 0.1, 0.2, 0.4, 0.8):
        for c in (0.5, 1.0):
            L.append((f"viscous nu={nu:g} c={c:g}", C.make_runner("viscous", "mean", c, ps=dict(nu=nu), steps=T), dict(family="viscous", c=c, eps=np.nan, nu=nu)))
    for k in (0.02, 0.1):
        L.append((f"damp k={k:g} c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=k), steps=T), dict(family="damp", c=1.0, eps=np.nan, k=k)))
    L.append(("wave c=1", C.make_runner("wave", "mean", 1.0, steps=T), dict(family="wave", c=1.0, eps=0.0)))
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
                k = int(np.nanargmax(a)); row["best_window_ari"] = float(a[k]); row["best_window_end"] = float(np.asarray(ao["t_end"])[k])
            row["rms_end_over_start"] = float(np.nanmedian(C.rms_final(r.X, r.fail_step)))
        rows.append(row)
    rows.append(dict(spectral_row(g, truth, meta), family="spectral"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t12b_small.csv" if small else "t12b_t4neg_map.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "dead_frac", "ac_final", "ari_edge_cut", "ari_edge_cut_lp", "edge_margin", "hold_edge_cut", "hold_fft_lowmode"]].mean().round(3).to_string())
