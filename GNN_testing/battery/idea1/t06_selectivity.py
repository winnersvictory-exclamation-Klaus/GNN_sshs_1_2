"""T1.06 mode selectivity: damp vs viscous at matched total energy loss (mean force, c=0.5)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

T = 1000; R = 8; C0 = 0.5; TM = 100       # AC energy matched at step TM
KS = [0.003, 0.01, 0.02, 0.04]

def ac_energy_ratio(g, sch, ps, u0, steps=TM):
    run = erun(g, sch, "mean", u0, C0, steps, ps)
    a = C.ac_rms(run.X)
    return float(np.median(a[-1] ** 2 / a[0] ** 2))

def match_nu(g, u0, k):
    target = ac_energy_ratio(g, "damp", dict(k=k), u0)
    lo, hi = 1e-5, 0.6
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        e = ac_energy_ratio(g, "viscous", dict(nu=mid), u0)
        if e > target: lo = mid
        else: hi = mid
    nu = np.sqrt(lo * hi)
    return nu, target, ac_energy_ratio(g, "viscous", dict(nu=nu), u0)

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    cfgs = [("wave", {}, "wave", np.nan, np.nan, np.nan)]
    for k in KS:
        nu, tgt, got = match_nu(g, u0, k)
        cfgs.append(("damp", dict(k=k), f"damp k={k}", k, tgt, np.nan))
        cfgs.append(("viscous", dict(nu=nu), f"viscous (match k={k})", k, got, nu))
    cfgs.append(("diffusion", None, "diffusion", np.nan, np.nan, np.nan))
    for sch, ps, label, k, e, nu_ in cfgs:
        if sch == "diffusion":
            run = C.run_diff(g, "mean", u0, 0.5, T)
        else:
            run = erun(g, sch, "mean", u0, C0, T, ps)
        me = C.mode_energy(run.X, g, run.fail_step)
        # fractions at several times
        row = dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], config=label, family=sch, match_k=k, energy_ratio_TM=e, nu=nu_)
        for t in (0, 50, 100, 200, 300, 500, 1000):
            row[f"low_t{t}"] = float(np.nanmedian(me["low"][t])); row[f"high_t{t}"] = float(np.nanmedian(me["high"][t]))
        a = C.ac_rms(run.X)
        row["ac_ratio_T"] = float(np.median(a[-1] / a[0]))
        row["ac_ratio_300"] = float(np.median(a[min(300, T)] / a[0]))
        ao = C.ari_over_time(run.X, g, truth, window=100, stride=50, q=2, fail_step=run.fail_step)
        ar = np.asarray(ao["ari"], float)
        ts = np.asarray(ao["t_start"]); te = np.asarray(ao["t_end"])
        if np.isfinite(ar).any():
            i = int(np.nanargmax(ar)); row["best_ari"] = float(ar[i]); row["best_t_end"] = int(te[i])
        else:
            row["best_ari"] = np.nan; row["best_t_end"] = -1
        for j, (s_, e_) in enumerate(zip(ts, te)):
            if int(e_) in (100, 300, 500, 1000): row[f"ari_end{int(e_)}"] = float(ar[j])
        # amplitude check: bridge-edge margin only meaningful if not decayed to roundoff
        row["alive_end"] = bool(row["ac_ratio_T"] > 1e-6)
        if row["alive_end"]:
            lab = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))
            sc = C.edge_score(run.X, g, run.fail_step)
            row["edge_margin"] = float(C.edge_margin(sc, g, truth)) if sc is not None else np.nan
            row["bridge_auc"] = float(C.bridge_auc(sc, meta["bridge_mask"])) if sc is not None else np.nan
        rows.append(row)
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3])
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0)
        import pandas as pd; print(pd.DataFrame(r).drop(columns=["graph"]).round(3).T.to_string()); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t06_selectivity.csv", rows)
    print(len(rows), time.time() - t0)
