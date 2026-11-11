"""T2.13 impulse initial condition.  u0 = unit impulse at a random node of block 0 (RMS(u0)=0.144, extremely non-uniform |v|).
Per-node stabilisers (vel_shrink, drag, limiter) vs per-realisation (cfl, energy, saturate), relative (rms) and fixed-scale (abs) versions,
forces mean/trim/owa/vw at c=1, 'mid'/'strong' parameters; same configs on a uniform IC for contrast.  cycle/clique/sbm bridges 2 seeds 0-1; R=8; T=500."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 500
FORCES = ["mean", "trim", "owa", "vw"]

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    n = g.n
    rng = np.random.default_rng(spec["seed"] + 77)
    nodes = rng.choice(24, R)
    imp = np.zeros((R, n)); imp[np.arange(R), nodes] = 1.0
    uni = L.make_u0(spec, g, R)
    far = np.arange(n) >= 24
    rows = []
    cfgs = [("wave", {}, "-", "wave"), ("damp", {"k": 0.1}, "-", "damp")]
    for b in L.ORIG:
        for mode in ("", "_abs"):
            if b == "cfl" and mode:
                continue
            for lvl, i in (("mid", 2), ("strong", 4)):
                cfgs.append((b + mode, {L.PARAM[b][0]: L.GRID6[b][i]}, lvl, b + mode))
    for f in FORCES:
        for sch, ps, lvl, nm in cfgs:
            for ic, u0 in (("impulse", imp), ("uniform", uni)):
                run, act = L.instr_run(g, sch, f, u0, 1.0, T, ps)
                row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], force=f, scheme=sch, level=lvl, ic=ic,
                           per_node=sch.split("_")[0] in ("vel", "drag", "limiter"), fail_frac=float(run.fail.mean()),
                           ac_final=L.nanmed(C.ac_final(run.X, run.fail_step)), lifetime=L.nanmed(L.lifetime(run.X, g, run.fail_step)))
                if run.ok.any():
                    row["act_first20"] = L.nanmean(act[:20][:, run.ok]); row["act_mean"] = L.nanmean(act[:, run.ok])
                    row["act_late"] = L.nanmean(act[T // 2:][:, run.ok])
                    if ic == "impulse":
                        ok = run.ok
                        fp = np.abs(run.X[:, ok][:, :, far]).max((0, 2))     # far-side peak per realisation
                        row["far_peak"] = float(fp.mean())
                        thr = 3e-3
                        arr = [(np.abs(run.X[:, r][:, far]).max(1) > thr).argmax() if (np.abs(run.X[:, r][:, far]).max(1) > thr).any() else np.nan for r in np.flatnonzero(ok)]
                        row["arrival"] = L.nanmean(arr)
                rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t13_impulse.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
