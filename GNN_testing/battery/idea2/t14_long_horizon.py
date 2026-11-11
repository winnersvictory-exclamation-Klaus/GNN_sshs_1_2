"""T2.14 long horizon, T=5000.  Configs = T2.07 minimal-rescue parameters (x1 = marginal by construction, and x2) per (kind, force, scheme) for
forces trim/owa/vw/median (damp = idea-1 control).  The rescue was established at T=2000 on seeds 0-2; here seeds 0-3 (seed 3 unseen) at
T=5000.  Records fail_frac as a function of horizon (500/1000/2000/3000/5000), first failure step, growth_rate, late amplitude, dead."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np, pandas as pd
C = L.C
T = 5000
RES = Path(__file__).parent / "results"
HZ = (500, 1000, 2000, 3000, 5000)
L.PARAM["damp"] = ("k", +1)

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    d = pd.read_csv(RES / "t07_minimal_rescue.csv")
    d = d[(d.type == "rescue") & (d.kind == spec["kind"]) & d.force.isin(["trim", "owa", "vw", "median"]) & d.factor.isin([1, 2])]
    cf = d.groupby(["force", "scheme", "factor"]).param.first().reset_index()
    rows = []
    for _, r in cf.iterrows():
        ps = {L.PARAM[r.scheme][0]: float(r.param)}
        run = C.run(g, r.scheme, r.force, u0, 1.0, T, ps)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], force=r.force, scheme=r.scheme, factor=int(r.factor), param=float(r.param))
        for h in HZ:
            row[f"fail_{h}"] = float(((run.fail_step >= 0) & (run.fail_step <= h)).mean())
        row["first_fail"] = float(run.fail_step[run.fail].min()) if run.fail.any() else np.nan
        row.update(C.traj_summary(run))
        row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
        ac = C.ac_rms(run.X, g)
        row["late_amp"] = L.nanmed(ac[-500:].mean(0)[run.ok] / ac[0][run.ok]) if run.ok.any() else np.nan
        rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1, 2, 3))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(RES / "t14_long_horizon.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
