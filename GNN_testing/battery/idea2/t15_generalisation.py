"""T2.15 generalisation: dev-selected configs (results/selected.json: per-scheme best from T2.10 on dev seeds) + baselines on zoo `wide`
(seeds 0,1) and `scale` (seed 0; n=48,96,192 -- n=384 DROPPED for cost).  c as in the config, T=500, R=8 (R=4 for n>=192)."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
from sklearn.metrics import adjusted_rand_score
C = L.C; rx = L.rx
T = 500
RES = Path(__file__).parent / "results"

def runners():
    sel = json.load(open(RES / "selected.json"))
    cf = [("wave+mean", C.make_runner("wave", "mean", 1.0, steps=T)), ("damp0.1+mean", C.make_runner("damp", "mean", 1.0, ps={"k": 0.1}, steps=T)),
          ("diffusion+mean", C.make_diff_runner("mean", 0.5, steps=T))]
    for s in sel:
        cf.append((s["name"], C.make_runner(s["scheme"], s["force"], s["c"], ps=s["ps"], steps=T)))
    return cf

def job(args):
    spec, group = args
    g, truth, meta = C.make_graph(spec)
    R = 8 if g.n < 192 else 4
    u0 = L.make_u0(spec, g, R)
    rows = []
    for name, runner in runners():
        run = runner(g, u0)
        row = dict(group=group, graph=meta["id"], seed=spec["seed"], kind=spec["kind"], n=meta["n"], bridges=meta["n_bridges"], config=name)
        row.update(C.traj_summary(run)); row["lifetime"] = L.nanmed(L.lifetime(run.X, g, run.fail_step))
        if run.ok.any() and truth is not None:
            row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))
        rows.append(row)
    if truth is not None:
        rows.append(dict(group=group, graph=meta["id"], seed=spec["seed"], kind=spec["kind"], n=meta["n"], bridges=meta["n_bridges"], config="spectral",
                         ari_edge_cut=adjusted_rand_score(truth, rx.spectral_labels(g))))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = [(s, "wide") for s in C.zoo("wide", seeds=(0, 1))] + [(s, "scale") for s in C.zoo("scale", seeds=(0,)) if C.normalize_spec(s).get("n", 0) < 384 and sum(C.normalize_spec(s).get("sizes", (0,))) < 384]
    print(len(specs), "graphs", flush=True)
    out = C.pmap(job, specs, procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(RES / "t15_generalisation.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
