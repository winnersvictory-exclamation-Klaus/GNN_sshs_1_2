"""T2.16 negative controls: zoo `neg` (er, shuffled, random_regular d=3,4,6,8, watts_strogatz; no true blocks), dev seeds 0-3 (28 graphs),
dev-selected configs + baselines.  False-partition rate = fraction of ok realisations whose edge_cut(q=2) labels have minfrac>=0.1 AND
hold>0 (min_frac-fixed hold); degenerate-cut rate = minfrac<0.1.  T=500, R=8."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
from t15_generalisation import runners
C = L.C
RES = Path(__file__).parent / "results"

def job(spec):
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, 8)
    rows = []
    for name, runner in runners():
        run = runner(g, u0)
        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], config=name, fail_frac=float(run.fail.mean()),
                   dead_frac=float(C.dead(run.X, run.fail_step).mean()) if run.ok.any() else np.nan)
        for rd in ("edge_cut", "edge_cut_lp"):
            if not run.ok.any():
                continue
            lab = C.partitions(run.X, g, 2, run.fail_step, which=(rd,))[rd]
            if lab is None:
                continue
            pm = C.partition_metrics(g, lab)
            h0 = C.hold(run.X, g, lab, min_frac=0.0, fail_step=run.fail_step)     # rx criterion (no min_frac)
            h1 = C.hold(run.X, g, lab, min_frac=0.1, fail_step=run.fail_step)
            ok = run.ok
            row[f"minfrac_{rd}"] = pm["minfrac"]
            row[f"degenerate_{rd}"] = float(pm["minfrac"] < 0.1)
            row[f"false_part_{rd}"] = float(np.nanmean((h1["hold"][ok] > 0)))
            row[f"false_part_rx_{rd}"] = float(np.nanmean((h0["hold"][ok] > 0)))
            row[f"cond_{rd}"] = pm["cond"]
        rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = C.zoo("neg", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2, progress=True)
    rows = [r for rs in out for r in rs]
    C.write_csv(RES / "t16_negative_controls.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
