"""T5.07 pure biharmonic control (plate:<base> force = -F o F, i.e. -L^2 u for mean). Wave scheme.
Theory: g(lam)=lam^2, stable iff c^2 lam_max^2 < 4 (c < 2/lam_max^2... = 2/lam_max for g=lam^2 -> c<2/lam_max).
c in {0.25,0.5,0.75,1.0,1.4}, plus the d4 mixture eps in {0.3,1,3,10} at matched c for comparison.
Core zoo (cycle/clique/sbm x bridges 1,2,4,8), dev seeds 0-3 at T=500 and 0-1 at T=2000 (REDUCED), R=8 (frequencies are ~lam smaller).
Output results/t07_plate.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

CS = [0.25, 0.5, 0.75, 1.0, 1.4]
CFG = [("plate", "plate:mean", {}), ("plain", "d4:mean", dict(eps=0.0)), ("d4 eps=0.3", "d4:mean", dict(eps=0.3)),
       ("d4 eps=1", "d4:mean", dict(eps=1.0)), ("d4 eps=3", "d4:mean", dict(eps=3.0)), ("d4 eps=10", "d4:mean", dict(eps=10.0))]


def job(args):
    spec, T = args
    g, truth, meta, u0 = graph_job_setup(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for name, force, pf in CFG:
        for c in CS:
            r = C.make_runner("wave", force, c, pf=pf, steps=T)(g, u0)
            gv = (lam[1:] ** 2) if name == "plate" else spatial_g(lam[1:], pf["eps"])
            rows.append(eval_run(r, g, truth, meta, f"{name} c={c:g} T={T}",
                                 extra=dict(form=name, c=c, T=T, pred_stable=bool((c * c * gv < 4).all() and (gv > 0).all()),
                                            c_crit_pred=2 / np.sqrt(gv.max()),
                                            w_block=float(np.arccos(np.clip(1 - c * c * gv[0] / 2, -1, 1))))))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:1] if small else C.DEV_SEEDS)
    specs4 = [s for s in specs if s['seed'] in (0, 1, 2, 3)]; specs2 = [s for s in specs if s['seed'] in (0, 1)]
    jobs = [(s, 500) for s in specs4] + ([] if small else [(s, 2000) for s in specs2])
    out = C.pmap(job, jobs, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t07_small.csv" if small else "t07_plate.csv"), df)
    print(df.groupby(["form", "c", "T"])[["fail_frac", "pred_stable", "ari_edge_cut", "ari_edge_cut_lp", "ari_fft_lowmode", "edge_margin", "hold_edge_cut"]].mean().round(3).to_string())
