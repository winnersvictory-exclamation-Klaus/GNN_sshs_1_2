"""T4.14 long horizon (T=5000) on marginal settings.  Space: c = f * c_pred, f in {.9,.99,.999,1.001,1.01,1.05}, forces mean / hi_acc / hi_alt / hi_geoneg.
Time: t6g at stable-boundary (e4=-.03,e6=.003: c=sqrt(s*/lam_max)*f) and weakly unstable small positive e4 (0.001, 0.003; e6=0) at c in {.3,.5,1.0}.
Graphs: cycle/clique/sbm 24+24 bridges 2, dev seeds 0-1 (6 graphs), R=8.  fail_frac(T) at T in {500,1000,2000,5000}, predicted growth rate."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C
W = i4.weights
TCK = (500, 1000, 2000, 5000)
FR = (0.9, 0.99, 0.999, 1.001, 1.01, 1.05)
SPACE = [("mean", "mean", {}, (1.0,)), ("hi_acc", "hi_acc:mean", {}, W("hi_acc")), ("hi_alt", "hi_alt:mean", {}, W("hi_alt")),
         ("hi_geoneg", "hi_geoneg:mean", {}, W("hi_geoneg", -.2))]
TIME = [((-0.03, 0.003), None), ((0.001, 0.0), (0.3, 0.5, 1.0)), ((0.003, 0.0), (0.3, 0.5, 1.0))]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    rows = []

    def rec(label, sch, fname, pf, ps, c, rate_pred, f):
        run = C.run(g, sch, fname, u0, c=c, steps=5000, ps=ps, pf=pf)
        m = C.traj_summary(run)
        row = dict(graph=meta["id"], seed=spec["seed"], config=label, c=c, f=f, rate_pred=rate_pred, ac_final=m["ac_final"], growth_rate=m["growth_rate"],
                   dead_frac=m["dead_frac"])
        for T in TCK:
            row[f"ff_T{T}"] = float(((run.fail_step >= 0) & (run.fail_step <= T)).mean())
        rows.append(row)

    for label, fname, pf, w in SPACE:
        cp = i4.c_crit_pred(lam, w)
        gv = i4.g_of(lam[1:], w)
        for f in FR:
            c = f * cp
            rate = float(np.log(i4.modal_rho_wave(c * c * gv)).max())
            rec(label, "wave", fname, pf, {}, c, rate, f)
    for eps, cs in TIME:
        if cs is None:
            rho_s = np.linspace(0.01, 4.2, 840)
            rr = i4.time_rho(eps, rho_s)
            bad = np.where(rr > 1 + 1e-9)[0]
            s_star = rho_s[bad[0] - 1] if len(bad) else 4.2
            cs_ = [np.sqrt(s_star / lam[-1]) * f for f in (0.9, 0.99, 1.01, 1.05)]
            fs = (0.9, 0.99, 1.01, 1.05)
        else:
            cs_, fs = cs, [None] * len(cs)
        for c, f in zip(cs_, fs):
            rate = float(np.log(i4.time_rho(eps, c * c * lam[1:])).max())
            rec(f"t6g{eps}", "t6g", "mean", {}, dict(eps=eps), c, rate, f)
    return rows


if __name__ == "__main__":
    specs = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=2, seed=s) for b in ("cycle", "clique", "sbm") for s in (0, 1)]
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t414_long_horizon.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["config", "f"], dropna=False).agg(rate=("rate_pred", "mean"), ff500=("ff_T500", "mean"), ff1000=("ff_T1000", "mean"), ff2000=("ff_T2000", "mean"),
          ff5000=("ff_T5000", "mean"), dead=("dead_frac", "mean")).round(3).to_string())
