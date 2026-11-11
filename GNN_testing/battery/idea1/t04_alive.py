"""T1.04 alive vs stable: at x_min, 2x, 4x (from T1.03) lifetime / fail / dead."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *
import pandas as pd

T = 2000; R = 8
MULTS = [1, 2, 4]

def lifetime(X, fs, thr=0.05):
    a = C.ac_rms(X)               # (T+1,R)
    out = np.full(a.shape[1], np.nan)
    for r in range(a.shape[1]):
        if fs[r] >= 0: continue
        below = np.flatnonzero(a[:, r] < thr * a[0, r])
        out[r] = below[0] if len(below) else X.shape[0] - 1
    return out

def job(args):
    spec, grp = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for _, r in grp.iterrows():
        for m in MULTS:
            x = r.x_min * m
            ps = dict(k=min(x, 1.0)) if r.scheme == "damp" else dict(nu=x)
            run = erun(g, r.scheme, r.force, u0, r.c, T, ps)
            s = C.traj_summary(run)
            life = lifetime(run.X, run.fail_step)
            X500 = run.X[:501]
            a500 = C.ac_rms(X500)
            fs = run.fail_step
            alive500 = [(a500[-1, i] / a500[0, i]) if fs[i] < 0 else np.nan for i in range(run.R)]
            rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=r.force, scheme=r.scheme, c=r.c,
                             mult=m, x=x, x_min=r.x_min, fail_frac=s["fail_frac"], dead_frac=s["dead_frac"],
                             life_med=float(np.nanmedian(life)) if np.isfinite(life).any() else np.nan,
                             ac_ratio_T500=float(np.nanmedian(alive500)) if np.isfinite(alive500).any() else np.nan,
                             ac_final=s["ac_final"], growth_rate=s["growth_rate"]))
    return rows

if __name__ == "__main__":
    t3 = pd.read_csv(OUT / "t03_rescue.csv")
    t3 = t3[t3.status.isin(["ok", "stable_at_lowest"])]
    specs = {C.make_graph(s)[2]["id"]: s for s in core_specs([0, 1, 2])}
    jobs = [(specs[gid], grp) for gid, grp in t3.groupby("graph")]
    if len(sys.argv) > 1:
        jobs = [(jobs[0][0], jobs[0][1].head(4))]
        t0 = time.time(); r = job(jobs[0]); print(pd.DataFrame(r).round(3).to_string(), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, jobs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t04_alive.csv", rows)
    print(len(rows), time.time() - t0)
