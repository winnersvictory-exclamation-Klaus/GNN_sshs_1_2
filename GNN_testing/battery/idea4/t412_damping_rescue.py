"""T4.12 damping rescue of unstable series.  Unstable cases: (i) linear series beyond c_pred (hi_acc:mean c=1.5 > 1.28, hi_geo(.5):mean c=1.0 > .707),
(ii) series over rank/clip bases (hi_acc:owa, hi_geo:owa, hi_alt:vw, hi_geo:trim, hi_acc:huber) at c in {0.5,1.0}.
Rescues: damp k in {.05,.1,.2,.4}, viscous nu in {.05,.1,.2,.4}, damp_visc(.05,.1).  Core zoo seed 0 (12 graphs), T=1000, R=4.
Also records ac_final (dead?) and the mean-base theory: damp stable iff c^2 g < 4-2k (damping LOWERS the top limit), viscous raises it."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

CASES = [("hi_acc:mean", {}, 1.5), ("hi_geo:mean", dict(eps=0.5), 1.0),
         ("hi_acc:owa", {}, 0.5), ("hi_acc:owa", {}, 1.0), ("hi_geo:owa", {}, 0.5), ("hi_alt:vw", {}, 0.5), ("hi_alt:vw", {}, 1.0),
         ("hi_geo:trim", {}, 0.5), ("hi_acc:huber", {}, 0.5), ("hi_acc:huber", {}, 1.0), ("hi_acc:trim", {}, 0.5), ("hi_acc:vw", {}, 0.5)]
RESC = [("none", "wave", {})] + [(f"damp{k}", "damp", dict(k=k)) for k in (0.05, 0.1, 0.2, 0.4)] + \
       [(f"visc{v}", "viscous", dict(nu=v)) for v in (0.05, 0.1, 0.2, 0.4)] + [("dv.05_.1", "damp_visc", dict(k=0.05, nu=0.1))]
T = 1000


def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = C.init("uniform", g, 4, np.random.default_rng(spec["seed"] * 1000 + 17))
    rows = []
    for fname, pf, c in CASES:
        for label, sch, ps in RESC:
            run = C.run(g, sch, fname, u0, c=c, steps=T, ps=ps, pf=pf)
            m = C.traj_summary(run)
            rows.append(dict(graph=meta["id"], force=fname + (f"[eps={pf['eps']}]" if pf else ""), c=c, rescue=label, fail_frac=m["fail_frac"],
                             dead_frac=m["dead_frac"], ac_final=m["ac_final"], mean_fail_step=m["mean_fail_step"]))
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=[0])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t412_damping_rescue.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["force", "c"], columns="rescue", values="fail_frac", aggfunc="mean", sort=False).round(2).to_string())
    ok = df[df.fail_frac == 0]
    print("rescued cases (fail_frac==0): alive vs dead")
    print(ok.groupby(["force", "c", "rescue"], sort=False).agg(n=("graph", "count"), dead=("dead_frac", "mean"), ac_final=("ac_final", "median")).round(3).to_string())
