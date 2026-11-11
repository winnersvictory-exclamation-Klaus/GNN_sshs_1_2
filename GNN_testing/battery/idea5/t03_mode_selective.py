"""T5.03 mode-selective stiffness: wave + d4:mean over eps, three c-modes (c=0.5, c=1, c=0.8*c_crit(eps)).
Metrics: high-mode energy fraction (displacement and potential=edge-difference weighting, late half), ARI of all
readouts, edge_margin, bridge_auc, raw hold (min_frac fix) on edge_cut / edge_cut_lp / fft_lowmode labels.
Zoo: core (cycle/clique/sbm x bridges 1,2,4,8, 24+24), dev seeds 0-3 (REDUCED from 0-7: CPU budget), R=8, T=500.
Output results/t03_mode_selective.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS = [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0]
T = 500


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for e in EPS:
        gv = spatial_g(lam[1:], e)
        ccrit = 2 / np.sqrt(gv.max()) if gv.min() > 0 else 0.0
        for cm, c in (("c0.5", 0.5), ("c1", 1.0), ("rel0.8", 0.8 * ccrit)):
            if c <= 0:
                continue
            runner = C.make_runner("wave", "d4:mean", c, pf=dict(eps=e), steps=T)
            r = runner(g, u0)
            rows.append(eval_run(r, g, truth, meta, f"d4 eps={e:g} {cm}",
                                 extra=dict(eps=e, cmode=cm, c=c, c_crit_pred=ccrit)))
    for label, runner in baselines(T)[2:]:
        r = runner(g, u0)
        rows.append(eval_run(r, g, truth, meta, label, extra=dict(eps=np.nan, cmode="base", c=getattr(runner, "c", np.nan))))
    rows.append(dict(spectral_row(g, truth, meta), eps=np.nan, cmode="base"))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t03_small.csv" if small else "t03_mode_selective.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac", "ari_edge_cut", "edge_margin", "high_pot_late", "high_disp_late", "hold_edge_cut"]].mean().round(3).to_string())
