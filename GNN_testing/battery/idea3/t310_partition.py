"""T3.10 partition quality: every force x {wave c=.5, damp(.1) c=1, diffusion eta=.5} on zoo core; all readouts."""
import i3
from i3 import C, np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

FORCES = ["mean"] + i3.FORCES19
SCH = [("wave", dict(scheme="wave", c=0.5)), ("damp", dict(scheme="damp", c=1.0, ps=dict(k=0.1))), ("diff", None)]
STEPS = 500

def runner(sk, f):
    if sk == "diff": return C.make_diff_runner(f, eta=0.5, steps=STEPS)
    kw = SCH[[s[0] for s in SCH].index(sk)][1]
    return C.make_runner(kw["scheme"], f, c=kw["c"], ps=kw.get("ps"), steps=STEPS)

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 310))
    rows = []
    sp = C.wt.rx.spectral_labels(g)
    rows.append(dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force="spectral", scheme="spectral",
                     ari_edge_cut=adjusted_rand_score(truth, sp), n_ok=1, fail_frac=0.0))
    for f in FORCES:
        for sk, _ in SCH:
            run = runner(sk, f)(g, u0)
            row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force=f, scheme=sk)
            row.update(C.traj_summary(run))
            if run.ok.any():
                row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else C.DEV_SEEDS[:4])
    if quick: specs = specs[:2]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t310_partition_quick.csv" if quick else "t310_partition.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["force", "scheme"])[["fail_frac", "ari_edge_cut", "ari_fft_lowmode", "edge_margin"]].mean().round(2).unstack("scheme").to_string())
