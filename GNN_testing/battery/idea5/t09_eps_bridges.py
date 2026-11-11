"""T5.09 eps x bridge count: wave + d4:mean, eps in {-0.4,-0.2,-0.1,0,0.1,0.3,0.6}, c in {0.5,1.0}, bridges in {1,2,4,8},
cycle/clique/sbm 24+24, dev seeds 0-3 (REDUCED from 0-7), R=8, T=1500.  Standard metrics at T=500 (truncated), ARI-vs-T (prefix
T in 100,200,300,500,1000,1500) for edge_cut/edge_cut_lp/fft_lowmode and steps-to-partition (first T with ARI>=0.9).
Baselines: damp(.1) c=1, diffusion(0.5), spectral.  Output results/t09_eps_bridges.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.3, 0.6]
CS = [0.5, 1.0]
TS = [100, 200, 300, 500, 1000, 1500]
TMAX = 1500


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    rows = []
    cfgs = [(f"d4 eps={e:g} c={c:g}", C.make_runner("wave", "d4:mean", c, pf=dict(eps=e), steps=TMAX), dict(eps=e, c=c, family="d4"))
            for e in EPS for c in CS]
    cfgs += [("damp(k=.1)+mean c=1", C.make_runner("damp", "mean", 1.0, ps=dict(k=0.1), steps=TMAX), dict(eps=np.nan, c=1.0, family="damp")),
             ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=TMAX), dict(eps=np.nan, c=np.nan, family="diff"))]
    for label, runner, ex in cfgs:
        r = runner(g, u0)
        if r.ok.any():
            row = eval_run(truncate(r, 500), g, truth, meta, label, readouts=("edge_cut", "edge_cut_lp", "fft_lowmode"), extra=ex)
            row.update(ari_vs_T(r, g, truth, TS))
            row["fail_frac_T1500"] = float(r.fail.mean())
        else:
            row = dict(graph=meta["id"], seed=meta["seed"], kind=meta["kind"], n=meta["n"], bridges=meta["n_bridges"], config=label,
                       fail_frac=1.0, fail_frac_T1500=1.0, **ex)
        rows.append(row)
    rows.append(dict(spectral_row(g, truth, meta), family="spectral"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t09_small.csv" if small else "t09_eps_bridges.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac_T1500", "ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "stp_edge_cut_lp"]].mean().round(3).to_string())
