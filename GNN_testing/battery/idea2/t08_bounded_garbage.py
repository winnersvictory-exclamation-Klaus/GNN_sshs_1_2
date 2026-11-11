"""T2.08 bounded-but-garbage?  Configs = the minimal-rescue (x1) and 2x-strength parameters found by T2.07 (per graph kind, force, scheme),
c=1, T=1000, graphs = kind x bridges 2 x seeds 0-2, R=8.  For non-failed realisations: Lyapunov rate (twin 1e-8, T=500), temporal
Nyquist fraction (checkerboard in time), high-mode energy fraction (top quartile of L_rw modes) in the last 200 steps, low-mode fraction,
late amplitude vs initial (limit-cycle level) and its coefficient of variation.  Reference: wave+mean c=1 and c=0.5, damp(.1)+mean."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np, pandas as pd
C = L.C
T = 1000
RES = Path(__file__).parent / "results"
L.PARAM["damp"] = ("k", +1)

def load_cfgs():
    d = pd.read_csv(RES / "t07_minimal_rescue.csv")
    d = d[d.type == "rescue"]
    cf = d.groupby(["kind", "force", "scheme", "factor"]).param.first().reset_index()
    return cf[cf.factor.isin([1, 2])]

def metrics(g, run, sch, f, u0, c, ps):
    ok = run.ok
    row = dict(fail_frac=float(run.fail.mean()))
    if not ok.any():
        return row
    X = run.X
    ts = C.temporal_spectrum(X, run.fail_step, start_frac=0.5)
    me = C.mode_energy(X, g, run.fail_step)
    ac = C.ac_rms(X, g)
    late = ac[-200:]
    row.update(nyq=L.nanmed(ts["nyquist"]), centroid=L.nanmed(ts["centroid"]),
               high_frac=L.nanmed(np.nanmean(me["high"][-200:], 0)[ok]), low_frac=L.nanmed(np.nanmean(me["low"][-200:], 0)[ok]),
               late_amp=L.nanmed(late.mean(0) / ac[0]), amp_cv=L.nanmed(late.std(0) / np.maximum(late.mean(0), 1e-300)),
               dead_frac=float(C.dead(X, run.fail_step).mean()))
    ly = C.lyapunov(g, sch, f, u0, c, 500, ps)
    row["lyap"] = L.nanmed(ly["rate"])
    row["lyap_chaotic_frac"] = float(np.nanmean(ly["rate"] > 0.01)) if np.isfinite(ly["rate"]).any() else np.nan
    return row

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    cf = load_cfgs()
    cf = cf[cf.kind == spec["kind"]]
    rows = []
    refs = [("wave", "mean", {}, 1.0, "ref"), ("wave", "mean", {}, 0.5, "ref"), ("damp", "mean", {"k": 0.1}, 1.0, "ref")]
    for sch, f, ps, c, tag in refs:
        run = C.run(g, sch, f, u0, c, T, ps)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=sch, force=f, c=c, factor=0, param=ps.get("k", np.nan), tag=tag)
        row.update(metrics(g, run, sch, f, u0, c, ps)); rows.append(row)
    for _, r in cf.iterrows():
        pn = L.PARAM[r.scheme][0]
        ps = {pn: float(r.param)}
        run = C.run(g, r.scheme, r.force, u0, 1.0, T, ps)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=r.scheme, force=r.force, c=1.0, factor=int(r.factor), param=float(r.param), tag="rescue")
        row.update(metrics(g, run, r.scheme, r.force, u0, 1.0, ps)); rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1, 2))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(RES / "t08_bounded_garbage.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
