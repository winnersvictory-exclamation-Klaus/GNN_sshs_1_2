"""T1.14 negative controls (zoo neg, no planted structure): false-partition and degenerate-cut rates."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfgs import *

T = 500; R = 8

def job(args):
    spec, cfgs = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for cfg in cfgs:
        run = run_cfg(g, u0, cfg, T)
        row = dict(graph=meta["id"], kind=meta["kind"], seed=spec["seed"], config=cfg["label"])
        s = C.traj_summary(run)
        row.update(fail_frac=s["fail_frac"], dead_frac=s["dead_frac"], ac_final=s["ac_final"])
        if run.ok.any():
            # per-realisation labels (pooled cut is what the repo does; here per run to get a rate)
            fl, dg, hd, fp = [], [], [], []
            for r in np.flatnonzero(run.ok):
                Xr = run.X[:, r:r + 1]; fs = np.array([-1])
                lab = C.partitions(Xr, g, 2, fs, which=("edge_cut",))["edge_cut"]
                if lab is None: continue
                mf = float(C.partition_metrics(g, lab, None)["minfrac"])
                h = C.hold(Xr, g, lab, min_frac=0.1, fail_step=fs)
                hv = float(np.asarray(h["hold"], float)[0])
                dg.append(mf < 0.1); fp.append((mf >= 0.1) and hv > 0); hd.append(hv)
            row.update(n_cuts=len(dg), degenerate_rate=float(np.mean(dg)) if dg else np.nan,
                       false_partition_rate=float(np.mean(fp)) if fp else np.nan, hold_mean=float(np.mean(hd)) if hd else np.nan)
        rows.append(row)
    return rows

if __name__ == "__main__":
    cfgs = load_cfgs() + BASELINES
    specs = C.zoo("neg", seeds=(0, 1, 2, 3))
    if len(sys.argv) > 1:
        t0 = time.time(); r = job((specs[0], cfgs)); print(len(r), time.time() - t0); sys.exit()
    rs = C.pmap(job, [(s, cfgs) for s in specs], procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t14_negctrl.csv", rows)
    print(len(rows))
