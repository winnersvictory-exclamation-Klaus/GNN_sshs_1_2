"""T5.15 long horizon: T=5000, mean base, R=8, cycle/clique/sbm bridges=2, dev seeds 0-7 (24 graphs).
Configs: wave plain; d4 eps in {-0.5,-0.3,0.3,1.0} at c=0.5 and at 0.95*c_crit_pred(eps) (marginal);
t4neg eps in {-0.1,-1/12,-0.05} at c=0.5,1.0 and at 0.97*c_crit_time(eps) (from s_crit=4(1+4eps)); t4 eps in {0.01,0.05} at c=0.1,0.25.
fail_frac at T in {500,1000,2000,5000} (from fail_step), growth rate, ac_final, dead at T=5000.  Output results/t15_long_horizon.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *

T = 5000
TS = [500, 1000, 2000, 5000]


def job(spec):
    g, truth, meta, u0 = graph_job_setup(spec)
    lam, Phi, d = C.spectrum(g)
    lm = lam[-1]
    cfgs = []
    for c in (0.5, 1.0):
        cfgs.append((f"wave c={c:g}", "wave", "d4:mean", c, {}, dict(eps=0.0), "plain"))
    for e in (-0.5, -0.3, 0.3, 1.0):
        gv = spatial_g(lam[1:], e)
        cc = 2 / np.sqrt(gv.max()) if gv.min() > 0 else 0.0
        cfgs.append((f"d4 eps={e:g} c=0.5", "wave", "d4:mean", 0.5, {}, dict(eps=e), "d4"))
        if cc > 0:
            cfgs.append((f"d4 eps={e:g} c=0.95ccrit", "wave", "d4:mean", 0.95 * cc, {}, dict(eps=e), "d4_marginal"))
            cfgs.append((f"d4 eps={e:g} c=1.02ccrit", "wave", "d4:mean", 1.02 * cc, {}, dict(eps=e), "d4_just_over"))
    for e in (-0.1, -1 / 12, -0.05):
        cc = np.sqrt(4 * (1 + 4 * e) / lm)
        for c in (0.5, 1.0):
            cfgs.append((f"t4neg eps={e:.3g} c={c:g}", "t4neg", "mean", c, dict(eps=e), {}, "t4neg"))
        cfgs.append((f"t4neg eps={e:.3g} c=0.97ccrit", "t4neg", "mean", 0.97 * cc, dict(eps=e), {}, "t4neg_marginal"))
        cfgs.append((f"t4neg eps={e:.3g} c=1.03ccrit", "t4neg", "mean", 1.03 * cc, dict(eps=e), {}, "t4neg_just_over"))
    for e in (0.01, 0.05):
        for c in (0.1, 0.25):
            cfgs.append((f"t4 eps={e:g} c={c:g}", "t4", "mean", c, dict(eps=e), {}, "t4pos"))
    rows = []
    for label, sch, force, c, ps, pf, fam in cfgs:
        r = C.run(g, sch, force, u0, c, T, ps=ps, pf=pf)
        fs = r.fail_step
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], config=label, family=fam, c=c, **C.traj_summary(r))
        for t in TS:
            row[f"fail_frac_T{t}"] = float(((fs >= 0) & (fs <= t)).mean())
        # late-time slope of log AC rms over steps 2500-5000 for runs still alive
        if r.ok.any():
            ac = C.ac_rms(r.X, g)[2500:, r.ok]
            tt = np.arange(ac.shape[0])
            with np.errstate(all="ignore"):
                sl = np.polyfit(tt, np.log(np.maximum(ac.mean(1), 1e-300)), 1)[0]
            row["late_slope_logac"] = float(sl)
        rows.append(row)
    return rows


if __name__ == "__main__":
    small = "--small" in sys.argv
    sp = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in (C.DEV_SEEDS[:1] if small else C.DEV_SEEDS)]
    out = C.pmap(job, sp, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / ("t15_small.csv" if small else "t15_long_horizon.csv"), df)
    print(df.groupby("config", sort=False)[["fail_frac_T500", "fail_frac_T1000", "fail_frac_T2000", "fail_frac_T5000", "ac_final", "late_slope_logac"]].mean().round(4).to_string())
