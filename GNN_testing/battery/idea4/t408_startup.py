"""T4.08 start-up sensitivity (mean base, time-series t6).  History init u(-k)=u0 (as in wt.time_even_step) vs (a) plain-wave start-up for m=6 / 20 steps
(`t6su`), (b) symmetric start-up u(-j)=u(+j) (`t6su_sym`).  Grid: (e4,e6) in E4 x E6, c in {0.1,0.3,0.5,1.0}, graphs cycle/clique/sbm blocks 24+24
bridges 2, seeds 0-1; T=1000, R=8.  Prediction from roots: rho_graph (max |z| over spectrum).  A failure is 'start-up induced' if baseline
fails and a proper start-up does not; 'root induced' if rho>1 (it must then fail for every start-up, up to growth rate x T)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

E4 = [-0.1, -0.03, -0.01, -1 / 12, 0.0, 0.003, 0.01, 0.1]
E6 = [-0.01, 0.0, 1 / 90, 0.01]
CS = (0.1, 0.3, 0.5, 1.0)
STARTS = [("hist=u0", "t6g", lambda e: dict(eps=e)),
          ("wave m=6", "t6su", lambda e: dict(e4=e[0], e6=e[1], m=6)),
          ("wave m=20", "t6su", lambda e: dict(e4=e[0], e6=e[1], m=20)),
          ("sym", "t6su_sym", lambda e: dict(e4=e[0], e6=e[1], force="mean"))]
T = 1000


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    rows = []
    for e4 in E4:
        for e6 in E6:
            for c in CS:
                rho = float(i4.time_rho((e4, e6), c * c * lam[1:]).max())
                for label, sch, mk in STARTS:
                    run = C.run(g, sch, "mean", u0, c=c, steps=T, ps=mk((e4, e6)))
                    m = C.traj_summary(run)
                    rows.append(dict(graph=meta["id"], seed=spec["seed"], e4=e4, e6=e6, c=c, start=label, rho_pred=rho,
                                     grow_T=np.log(max(rho, 1e-300)) * T, fail_frac=m["fail_frac"], mean_fail_step=m["mean_fail_step"],
                                     ac_final=m["ac_final"], growth_rate=m["growth_rate"]))
    return rows


if __name__ == "__main__":
    specs = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=2, seed=s) for b in ("cycle", "clique", "sbm") for s in (0, 1)]
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t408_startup.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    piv = df.pivot_table(index=["e4", "e6", "c"], columns="start", values="fail_frac", aggfunc="mean")
    piv["rho_pred"] = df.groupby(["e4", "e6", "c"]).rho_pred.max()
    print(piv.round(2).to_string())
