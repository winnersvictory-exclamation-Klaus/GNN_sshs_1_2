"""T2.03 activation audit with scheme-specific instrumentation (own 10%-modification criterion, see i2lib._core).
Forces mean, trim, owa, vw, huber at c=1 (+mean at c=0.5); each scheme at weak/mid/strong parameter;
graphs cycle/clique/sbm bridges 2 seeds 0-1 (6), R=8, T=1000; activation binned in 10 time bins."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
STEPS = 1000
NB = 10

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    rows = []
    cases = [("mean", 0.5), ("mean", 1.0), ("trim", 1.0), ("owa", 1.0), ("vw", 1.0), ("huber", 1.0)]
    for f, c in cases:
        for b in L.ORIG:
            for i, lvl in ((0, "weak"), (2, "mid"), (4, "strong")):
                ps = {L.PARAM[b][0]: L.GRID6[b][i]}
                run, act = L.instr_run(g, b, f, u0, c, STEPS, ps)
                ac = C.ac_rms(run.X, g)            # (T+1,R)
                row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], force=f, c=c, scheme=b, level=lvl,
                           param=ps[L.PARAM[b][0]], fail_frac=float(run.fail.mean()), ac_final=L.nanmed(C.ac_final(run.X, run.fail_step)))
                ok = run.ok
                if ok.any():
                    a = act[:, ok]
                    for k in range(NB):
                        sl = slice(k * STEPS // NB, (k + 1) * STEPS // NB)
                        row[f"act_b{k}"] = float(np.nanmean(a[sl]))
                        row[f"amp_b{k}"] = float(np.nanmean(ac[sl][:, ok] / ac[0][ok]))
                    row["act_mean"] = float(np.nanmean(a)); row["act_late"] = float(np.nanmean(a[STEPS // 2:]))
                    row["act_first"] = float(np.nanmean(a[:50]))
                    # permanent: still >5% active in the last quarter while the state is still non-trivial (AC-RMS>1e-3 of start)
                    live = (ac[STEPS * 3 // 4:STEPS][:, ok] / ac[0][ok]) > 1e-3
                    row["live_late_frac"] = float(live.mean())
                    row["act_late_given_live"] = float(np.nanmean(np.where(live, a[STEPS * 3 // 4:STEPS], np.nan))) if live.any() else np.nan
                rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t03_activation.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
