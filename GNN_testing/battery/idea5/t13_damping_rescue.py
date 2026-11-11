"""T5.13 damping rescue.
(a) time term: t4 (eps in {0.05,0.1,0.5}) + explicit velocity damping (scheme t4damp: -k(u_t-u_{t-1})), c in {0.25,0.5,1.0},
    k grid; minimal k with fail_frac=0 at T=2000 vs the root-analysis prediction (damped quartic, graph spectrum).
(b) space: d4:<base> eps in {-0.3, 0, +0.3}, base in mean/trim/owa/vw/huber, wave+damp (k grid) and viscous (nu grid), c=1, T=1000:
    minimal k / nu for fail_frac=0; compare eps=0 reference.
(c) anti-restoring: d4:mean eps in {-0.7,-1.0} (g<0 for lam>1/|eps|) + damp/viscous: theory says no rescue (P(1)=-c^2 mu<0).
Graphs: cycle/clique/sbm bridges 2, dev seeds 0-1 (6 graphs; REDUCED), R=8.  Output results/t13_damping_rescue.csv (rows: graph x part x cfg x parameter value)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

KS = [0.0] + list(np.round(np.logspace(-3, np.log10(0.5), 10), 5))
NUS = [0.0] + list(np.round(np.logspace(-2, np.log10(1.0), 9), 5))


def damped_t4_rho(e, s, k):
    lead = 1 + e
    return float(np.max(np.abs(np.roots([lead, s - 2 - 4 * e + k * lead, 1 + 6 * e - k * lead, -4 * e, e]))))


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    base = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"])
    # (a)
    for e in (0.05, 0.1, 0.5):
        for c in (0.25, 0.5, 1.0):
            for k in KS:
                ff = C.fail_frac(g, "t4damp", "mean", u0, c, 2000, ps=dict(eps=e, k=k))
                pr = max(damped_t4_rho(e, c * c * l, k) for l in lam[1:])
                rows.append(dict(base, part="a_time", cfg=f"t4damp eps={e} c={c}", eps=e, c=c, k=k, nu=0.0, fail_frac=ff, pred_rho=pr, pred_stable=pr <= 1 + 1e-9))
    # (b)
    for b in BASES:
        for e in (-0.3, 0.0, 0.3):
            for sch, par, grid in (("damp", "k", KS), ("viscous", "nu", NUS)):
                for v in grid:
                    ff = C.fail_frac(g, sch, f"d4:{b}", u0, 1.0, 1000, ps={par: v}, pf=dict(eps=e))
                    rows.append(dict(base, part="b_space", cfg=f"{sch}+d4:{b} eps={e}", base_force=b, eps=e, c=1.0, k=v if par == "k" else 0.0,
                                     nu=v if par == "nu" else 0.0, scheme=sch, fail_frac=ff))
    # (c)
    for e in (-0.7, -1.0):
        for sch, par, grid in (("damp", "k", KS), ("viscous", "nu", NUS)):
            for c in (0.25, 0.5, 1.0):
                for v in grid:
                    ff = C.fail_frac(g, sch, "d4:mean", u0, c, 1000, ps={par: v}, pf=dict(eps=e))
                    rows.append(dict(base, part="c_anti", cfg=f"{sch}+d4:mean eps={e} c={c}", eps=e, c=c, k=v if par == "k" else 0.0,
                                     nu=v if par == "nu" else 0.0, scheme=sch, fail_frac=ff))
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in (C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:2])]
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t13_small.csv" if small else "t13_damping_rescue.csv"), df)
    a = df[df.part == "a_time"]
    print(a.pivot_table(index=["eps", "c"], columns="k", values="fail_frac", aggfunc="mean").round(2).to_string())
