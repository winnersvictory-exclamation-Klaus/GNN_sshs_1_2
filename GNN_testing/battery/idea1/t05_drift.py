"""T1.05 drift vs oscillation: fail vs fail_ac, dc_drift; recenter control schemes."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

@wt.scheme("recenter")
def _(ctx, cur, prev, F, Fp):
    """wave step then subtract plain node-mean."""
    nxt = 2 * cur - prev + ctx.c ** 2 * F
    return nxt - nxt.mean(1, keepdims=True)

@wt.scheme("recenter_dw")
def _(ctx, cur, prev, F, Fp):
    """wave step then subtract degree-weighted node-mean."""
    if "d" not in ctx.state:
        ctx.state["d"] = C.wdeg(ctx.g)
    d = ctx.state["d"]
    nxt = 2 * cur - prev + ctx.c ** 2 * F
    return nxt - (nxt * d).sum(1, keepdims=True) / d.sum()

@wt.scheme("damp_recenter", k=0.1)
def _(ctx, cur, prev, F, Fp):
    """damp step then subtract plain node-mean."""
    nxt = cur + (1 - ctx.ps["k"]) * (cur - prev) + ctx.c ** 2 * F
    return nxt - nxt.mean(1, keepdims=True)

FORCES = ["mean", "owa", "vw", "trim", "median", "symowa_ext"]
CONFIGS = [("wave", {}), ("recenter", {}), ("recenter_dw", {}), ("damp", dict(k=0.1)), ("damp", dict(k=0.3)),
           ("damp_recenter", dict(k=0.1))]
CS = [0.5, 1.0]
T = 1000; R = 8

def job(args):
    spec = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for f in FORCES:
        for sch, ps in CONFIGS:
            for c in CS:
                run = erun(g, sch, f, u0, c, T, ps)
                s = C.traj_summary(run)
                # AC-only failure: rerun with big threshold (recentering schemes: dc is zero anyway)
                big = erun(g, sch, f, u0, c, 1200, ps, fail_amp=1e6)
                X = big.X
                with np.errstate(all="ignore"):
                    raw = (np.abs(X).max(2) >= 20); acm = (np.abs(X - X.mean(2, keepdims=True)).max(2) >= 20)
                raw_f = raw.any(0); ac_f = acm.any(0)
                dcm = X.mean(2)                      # (t,R) node-mean trajectory (valid until big failure)
                ok6 = ~(np.abs(X) >= 1e5).any((0, 2)) # exclude realisations that went astronomically big
                dc_t = np.nanmax(np.abs(dcm), 0) if len(dcm) else np.nan
                rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=f, scheme=sch,
                                 ps=str(ps), c=c, fail_frac_T1000=s["fail_frac"], fail1200_raw=float(raw_f.mean()),
                                 fail1200_ac=float(ac_f.mean()), dc_only1200=float((raw_f & ~ac_f).mean()),
                                 dc_max_abs1200=float(np.nanmedian(dc_t)), mean_fail_step=s["mean_fail_step"],
                                 dc_dw=s["dc_dw"], dc_plain=s["dc_plain"], dead_frac=s["dead_frac"], growth_rate=s["growth_rate"]))
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2, 3])
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0)
        import pandas as pd; print(pd.DataFrame(r).drop(columns=["graph"]).round(3).to_string()); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t05_drift.csv", rows)
    print(len(rows), time.time() - t0)
