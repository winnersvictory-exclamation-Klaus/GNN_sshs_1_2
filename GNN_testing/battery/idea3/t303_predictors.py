"""T3.03 which force property predicts wave blow-up?  (force x core graph x dev seed x c)."""
import i3
from i3 import C, np
import pandas as pd

FORCES = i3.FORCES19 + ["owa_mc", "vw_mc", "trim_mc", "mean"]
CS = (0.5, 1.0)
PROPS = ["oddness", "momentum_rel", "dsym_rel", "eig_sym_max", "lam_max", "homog_rel"]

def job(args):
    spec, R, steps = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 303)
    u0 = C.init("uniform", g, R, rng)
    mid = C.run(g, "damp", "mean", u0, 0.5, 100, ps=dict(k=0.1)).X[-1][:4]   # generic mid-trajectory states
    rows = []
    for f in FORCES:
        P = {}
        for tag, U in (("u0", u0[:4]), ("mid", mid)):
            pr = C.force_props_rows(g, f, U, None)
            for k in PROPS:
                P[f"{k}_{tag}"] = float(np.nanmean([p[k] for p in pr]))
            P[f"kink_{tag}"] = float(np.nanmean([p["jac_kink"] for p in pr]))
        for c in CS:
            run = C.run(g, "wave", f, u0, c, steps)
            row = dict(graph=meta["id"], seed=spec["seed"], gkind=spec["block"], bridges=meta["n_bridges"], force=f, c=c)
            row.update(P)
            row["stiff_u0"] = c * np.sqrt(max(P["lam_max_u0"], 0)) / 2
            row["stiff_mid"] = c * np.sqrt(max(P["lam_max_mid"], 0)) / 2
            s = C.traj_summary(run)
            row.update(fail_frac=s["fail_frac"], growth_rate=s["growth_rate"], mean_fail_step=s["mean_fail_step"], dead_frac=s["dead_frac"])
            fm = C.fail_modes(g, "wave", f, u0, c, steps)
            row["ac_fail_frac"] = float(np.mean(fm["ac_fail"])); row["dc_only_frac"] = float(np.mean(fm["dc_only"]))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:1] if quick else C.DEV_SEEDS)
    out = C.pmap(job, [(s, 8, 500) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t303_predictors_quick.csv" if quick else "t303_predictors.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["force", "c"]).fail_frac.mean().unstack().round(2).to_string())
