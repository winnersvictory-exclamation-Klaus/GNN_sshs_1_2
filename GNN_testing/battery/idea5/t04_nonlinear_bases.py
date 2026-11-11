"""T5.04 nonlinear bases: F - eps F o F for base in mean/trim/owa/vw/huber (d4:<base> with pf eps), wave scheme.
eps in {-0.3,-0.1,0,0.1,0.3}, c in {0.25,0.5,1.0}; cycle/clique/sbm, bridges=2, dev seeds 0-3 (12 graphs; REDUCED from 0-7), R=8, T=500.
Also force_props of the composed force on a random state (+ at eps=0 as reference).
Output results/t04_nonlinear_bases.csv (fail map) and results/t04_props.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

EPS = [-0.3, -0.1, 0.0, 0.1, 0.3]
CS = [0.25, 0.5, 1.0]
T = 500


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    rows, prows = [], []
    for b in BASES:
        force = f"d4:{b}"
        for e in EPS:
            for k, u in enumerate(u0[:2]):
                try:
                    pr = C.force_props(g, force, u, pf=dict(eps=e))
                except Exception as ex:
                    pr = dict(error=str(ex))
                prows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], base=b, eps=e, state=k, **pr))
            for c in CS:
                r = C.make_runner("wave", force, c, pf=dict(eps=e), steps=T)(g, u0)
                s = C.traj_summary(r)
                fs = r.fail_step
                rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], base=b, eps=e, c=c,
                                 first_fail=float(fs[fs >= 0].min()) if (fs >= 0).any() else np.nan, **s))
    return rows, prows


if __name__ == "__main__":
    small = "--small" in sys.argv
    seeds = C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4]
    specs = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, specs, procs=2)
    df = pd.DataFrame([r for o in out for r in o[0]]); dp = pd.DataFrame([r for o in out for r in o[1]])
    sfx = "_small" if small else ""
    C.write_csv(RES / f"t04_nonlinear_bases{sfx}.csv", df); C.write_csv(RES / f"t04_props{sfx}.csv", dp)
    print(df.pivot_table(index=["base", "eps"], columns="c", values="fail_frac", aggfunc="mean").round(2))
