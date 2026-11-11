"""T3.04 Jacobian-predicted vs empirical c_crit (wave). force x graph (cycle/clique/sbm, bridges=2) x seeds."""
import i3
from i3 import C, np
import pandas as pd

FORCES = i3.FORCES19 + ["owa_mc", "vw_mc", "trim_mc", "mean"]
STEPS = 2000

def job(args):
    spec, f = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 304)
    u0 = C.init("uniform", g, 8, rng)
    ce = C.c_crit_empirical(g, "wave", f, u0=u0, steps=STEPS, R=8, seed=spec["seed"])
    row = dict(graph=meta["id"], gkind=spec["block"], seed=spec["seed"], force=f, c_crit_emp=ce["c_crit"],
               status=ce["status"], non_monotone=ce["non_monotone"])
    # random-state linear prediction
    pr = C.force_props_rows(g, f, u0[:4], None)
    row["lam_max_u0"] = float(np.nanmean([p["lam_max"] for p in pr]))
    row["eig_sym_max_u0"] = float(np.nanmax([p["eig_sym_max"] for p in pr]))
    row["c_pred_u0"] = float(np.nanmin([p["c_crit_pred"] for p in pr]))
    row["c_eig_u0"] = float(np.nanmin([p["c_crit_eig"] for p in pr]))
    row["kink_u0"] = float(np.nanmax([p["jac_kink"] for p in pr]))
    row["dsym_u0"] = float(np.nanmean([p["dsym_rel"] for p in pr]))
    # trajectory-sampled prediction at a sub-critical c
    cc = ce["c_crit"]
    cs = 0.5 * cc if np.isfinite(cc) and cc > 0 else (0.5 if not np.isfinite(cc) else np.nan)
    if np.isfinite(cs):
        cs = min(cs, 1.0)
        run = C.run(g, "wave", f, u0, cs, STEPS)
        row["c_sample"] = cs; row["sample_fail_frac"] = float(run.fail.mean())
        ts = [t for t in (0, 10, 50, 200, 500, 1000, 2000)]
        lam, eigs, cp, ce2, kk, ds = [], [], [], [], [], []
        for t in ts:
            for r in range(4):
                if run.fail[r] and t >= run.fail_step[r]:
                    continue
                p = C.force_props(g, f, u=run.X[t, r])
                lam.append(p["lam_max"]); eigs.append(p["eig_sym_max"]); cp.append(p["c_crit_pred"]); ce2.append(p["c_crit_eig"]); kk.append(p["jac_kink"]); ds.append(p["dsym_rel"])
        if lam:
            row.update(lam_max_traj=float(np.nanmax(lam)), eig_sym_max_traj=float(np.nanmax(eigs)), c_pred_traj=float(np.nanmin(cp)),
                       c_eig_traj=float(np.nanmin(ce2)), kink_traj=float(np.nanmax(kk)), dsym_traj=float(np.nanmean(ds)), n_samples=len(lam))
    return row

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = (0,) if quick else (0, 1, 2, 3)
    specs = [dict(kind="two_block", block=b, bridges=2, sizes=(24, 24), wb=1.0, seed=s) for b in ("cycle", "clique", "sbm") for s in seeds]
    fs = FORCES[:3] if quick else FORCES
    rows = C.pmap(job, [(s, f) for s in specs for f in fs], procs=2)
    C.write_csv(i3.RES / ("t304_ccrit_quick.csv" if quick else "t304_ccrit.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["force", "gkind"])[["c_crit_emp", "c_pred_u0", "c_pred_traj", "c_eig_traj"]].median().round(3).to_string())
