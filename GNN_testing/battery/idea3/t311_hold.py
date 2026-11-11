"""T3.11 quasi-equilibrium (hold, min_frac=0.1 fix) on raw trajectories: every force x {wave c=.5, damp c=1, diffusion}."""
import i3
from i3 import C, np
import pandas as pd
from t310_partition import FORCES, runner

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 311))
    rows = []
    for f in FORCES:
        for sk in ("wave", "damp", "diff"):
            run = runner(sk, f)(g, u0)
            row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force=f, scheme=sk, fail_frac=float(run.fail.mean()))
            if run.ok.any():
                labs = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))["edge_cut"]
                for tag, lab in (("edge", labs), ("truth", truth)):
                    if lab is None: continue
                    for norm in ("rms", "ac"):
                        h = C.hold(run.X, g, lab, fail_step=run.fail_step, norm=norm)
                        hh = h["hold"][run.ok]
                        row[f"hold_{tag}_{norm}"] = float(np.nanmean(hh)); row[f"holdpos_{tag}_{norm}"] = float(np.nanmean(hh > 0))
                    row[f"rejected_{tag}"] = bool(h["rejected"]) if np.ndim(h["rejected"]) == 0 else float(np.mean(h["rejected"]))
                # un-fixed rx criterion for reference (min_frac=0), edge_cut labels
                if labs is not None:
                    h0 = C.hold(run.X, g, labs, min_frac=0.0, fail_step=run.fail_step)
                    row["holdpos_edge_nofix"] = float(np.nanmean(h0["hold"][run.ok] > 0))
                row["ac_final"] = float(np.nanmedian(C.ac_final(run.X, run.fail_step)[run.ok]))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else C.DEV_SEEDS[:4])
    if quick: specs = specs[:2]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t311_hold_quick.csv" if quick else "t311_hold.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["force", "scheme"])[["fail_frac", "hold_truth_rms", "holdpos_truth_rms", "holdpos_edge_rms", "holdpos_edge_nofix"]].mean().round(2).to_string())
