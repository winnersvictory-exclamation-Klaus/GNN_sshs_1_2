"""T1.08 raw-wave quasi-equilibrium: hold on raw damped waveform vs undamped vs diffusion (mean force)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

T = 500; R = 8
KS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6]
NUS = [0.05, 0.2, 0.4]
CS = [0.5, 1.0]

def evaluate(run, g, truth, meta):
    out = dict(fail_frac=float(run.fail.mean()))
    s = C.traj_summary(run); out.update(ac_final=s["ac_final"], dead_frac=s["dead_frac"])
    if not run.ok.any():
        return out
    ev = C.evaluate_partitions(run.X, g, truth, run.fail_step)
    out.update({k: ev[k] for k in ("ari_edge_cut", "ari_edge_cut_lp", "edge_margin", "bridge_auc", "minfrac_edge_cut")})
    for rd in ("edge_cut", "edge_cut_lp"):
        labs = C.partitions(run.X, g, 2, run.fail_step, which=(rd,))[rd]
        if labs is None: continue
        for tag, mf in (("fix", 0.1), ("rx", 0.0)):
            h = C.hold(run.X, g, labs, min_frac=mf, fail_step=run.fail_step)
            hv = np.asarray(h["hold"], float)
            out[f"hold_{rd}_{tag}"] = float(np.nanmean(hv)); out[f"holdfrac_{rd}_{tag}"] = float(np.nanmean(hv > 0))
        h = C.hold(run.X, g, labs, min_frac=0.1, fail_step=run.fail_step, norm="ac")
        out[f"holdfrac_{rd}_ac"] = float(np.nanmean(np.asarray(h["hold"], float) > 0))
    return out

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    cfgs = [("diffusion", None, "diffusion", None)]
    for c in CS:
        for k in KS:
            sch, ps = scheme_for(k, 0)
            cfgs.append((sch, ps, f"damp k={k}" if k else "wave", c))
        for nu in NUS:
            cfgs.append(("viscous", dict(nu=nu), f"viscous nu={nu}", c))
    for sch, ps, label, c in cfgs:
        run = C.run_diff(g, "mean", u0, 0.5, T) if sch == "diffusion" else erun(g, sch, "mean", u0, c, T, ps)
        row = dict(graph=meta["id"], kind=spec["kind"], bridges=spec["bridges"], seed=spec["seed"], config=label, c=c if c else np.nan)
        row.update(evaluate(run, g, truth, meta))
        rows.append(row)
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3], bridges=2) + core_specs([0, 1, 2, 3], bridges=4)
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0)
        import pandas as pd; print(pd.DataFrame(r).drop(columns=["graph"]).round(3).to_string()); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t08_hold.csv", rows)
    print(len(rows), time.time() - t0)
