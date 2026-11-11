"""T1.13 generalisation of the dev-selected top configs: zoo wide + scale, dev seeds 0-1 (+ baselines)."""
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
        row = dict(graph=meta["id"], kind=meta["kind"], n=meta["n"], seed=spec["seed"], bridges=meta["n_bridges"], config=cfg["label"],
                   sizes=str(spec.get("sizes")), wb=spec.get("wb"), params=str(meta["params"]))
        row.update(metrics(g, truth, meta, run))
        sp = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",)) if False else None
        rows.append(row)
    if truth is not None:
        rows.append(dict(graph=meta["id"], kind=meta["kind"], n=meta["n"], seed=spec["seed"], bridges=meta["n_bridges"], config="spectral",
                         ari_edge_cut=adjusted_rand_score(truth, rx.spectral_labels(g)), fail_frac=0.0))
    return rows

if __name__ == "__main__":
    cfgs = load_cfgs() + BASELINES
    specs = []
    for name in ("wide", "scale"):
        specs += C.zoo(name, seeds=(0, 1))
    # extra: bridges 1-8, unequal sizes
    for b in (1, 2, 4, 8):
        for sz in ((16, 32), (8, 40)):
            for k in ("cycle", "clique", "sbm"):
                for s in (0, 1):
                    specs.append(dict(kind=k, sizes=sz, bridges=b, wb=1.0, seed=s))
    print(len(specs), "graphs", len(cfgs), "configs")
    if len(sys.argv) > 1:
        t0 = time.time(); r = job((specs[0], cfgs)); print(len(r), time.time() - t0); sys.exit()
    t0 = time.time()
    rs = C.pmap(job, [(s, cfgs) for s in specs], procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t13_generalise.csv", rows)
    print(len(rows), time.time() - t0)
