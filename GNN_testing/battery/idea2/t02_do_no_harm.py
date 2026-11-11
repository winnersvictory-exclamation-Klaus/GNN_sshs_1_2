"""T2.02 do-no-harm: mean force, c in {0.5,1.0}, each scheme at weak/mid/strong parameter vs plain wave.
Graphs: cycle/clique/sbm, bridges 2, 24+24, seeds 0-3 (12), R=8, T=500."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
STEPS = 500
READ = ("edge_cut", "fft_lowmode", "timeavg_2means")

def feats(run, g, truth):
    X, fs = run.X, run.fail_step
    d = {}
    ts = C.temporal_spectrum(X, fs)
    d["centroid"] = L.nanmed(ts["centroid"]); d["nyq"] = L.nanmed(ts["nyquist"])
    E = C.energy(X, g, run.c, fs)
    d["E_end"] = L.nanmed(C.energy_drift(E)["ratio_end"])
    labs = C.partitions(X, g, 2, fs, which=READ) if run.ok.any() else {}
    for r in READ:
        lab = labs.get(r)
        d["ari_" + r] = C.partition_metrics(g, lab, truth)["ari"] if lab is not None else np.nan
    d["ac_final"] = L.nanmed(C.ac_final(X, fs))
    d["fail_frac"] = float(run.fail.mean())
    return d

def reldev(X, Xw, g, n):
    A = X[:n] - X[:n].mean(2, keepdims=True); B = Xw[:n] - Xw[:n].mean(2, keepdims=True)
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    rows = []
    for c in (0.5, 1.0):
        w = C.run(g, "wave", "mean", u0, c, STEPS)
        fw = feats(w, g, truth)
        cfgs = [("damp", {"k": 0.1}, "damp k=0.1")]
        for b in L.ORIG:
            for i in (0, 2, 4):
                cfgs.append((b, {L.PARAM[b][0]: L.GRID6[b][i]}, ["weak", "mid", "strong"][i // 2]))
        for sch, ps, lvl in cfgs:
            r = C.run(g, sch, "mean", u0, c, STEPS, ps)
            f = feats(r, g, truth)
            row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], c=c, scheme=sch, level=lvl,
                       param=list(ps.values())[0], config=L.cfg_label(sch, ps))
            row["dev50"] = reldev(r.X, w.X, g, 51) if not r.fail.any() else np.nan
            row["dev500"] = reldev(r.X, w.X, g, STEPS + 1) if not r.fail.any() else np.nan
            for k, v in f.items():
                row[k] = v; row["wave_" + k] = fw[k]; row["d_" + k] = v - fw[k]
            rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1, 2, 3))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t02_do_no_harm.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
