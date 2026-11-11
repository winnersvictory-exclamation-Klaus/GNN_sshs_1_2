"""T3.14 confounders: 5% outlier nodes (stuck / noisy) and bridge-node noise; ARI/edge_margin degradation per force."""
import i3
from i3 import C, np
import pandas as pd
from t310_partition import FORCES, runner

CONDS = [("clean", None), ("stuck5", ("stuck", 0.05)), ("noisy5", ("noisy", 0.05)), ("bridge0.05", ("bridge", 0.05)), ("bridge0.2", ("bridge", 0.2))]
SCH = ("damp", "diff")
FOR = ["mean", "trim", "median", "huber", "tanh", "amp", "gauss", "absvw", "symowa_ext", "symowa_med", "odd_owa", "odd_vw", "huber_local", "huber_global",
       "amp_gated", "blend_agree", "huber_anneal", "edge_memory", "owa", "vw"]

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    mask = meta["bridge_mask"]
    u0 = C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 314))
    rows = []
    for f in FOR:
        for sk in SCH:
            base = runner(sk, f)
            for cn, cd in CONDS:
                if cd is None: rn = base
                elif cd[0] == "bridge": rn = C.bridge_noise(base, mask, sigma=cd[1], seed=spec["seed"])
                else: rn = C.with_outliers(base, frac=cd[1], mode=cd[0], seed=spec["seed"])
                run = rn(g, u0)
                row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force=f, scheme=sk, cond=cn)
                row.update(C.traj_summary(run))
                if run.ok.any():
                    row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut", "timeavg_2means", "fft_lowmode")))
                rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else (0, 1))
    if quick: specs = specs[:1]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t314_confounders_quick.csv" if quick else "t314_confounders.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["force", "scheme", "cond"])[["fail_frac", "ari_edge_cut", "ari_timeavg_2means"]].mean().round(2).unstack("cond").to_string())
