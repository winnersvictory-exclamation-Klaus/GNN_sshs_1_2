"""T3.07 parameter sweeps of each force's own parameter x scheme x c: fail, dead, ARI, edge_margin."""
import i3
from i3 import C, np
import pandas as pd

SERIES = [  # (force, param, values, fixed other params)
    ("huber", "tau", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], {}),
    ("tanh", "tau", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], {}),
    ("amp", "tau", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], {}),
    ("amp", "beta", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0], {}),
    ("gauss", "sigma", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], {}),
    ("absvw", "beta", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], {}),
    ("symowa_ext", "gamma", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], {}),
    ("symowa_med", "gamma", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], {}),
    ("huber_local", "kappa", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], {}),
    ("huber_global", "kappa", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], {}),
    ("amp_gated", "kappa", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], {}),
    ("blend_agree", "tau", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], {}),
    ("huber_anneal", "Ta", [10.0, 30.0, 100.0, 300.0, 1000.0, 1e9], {}),
    ("edge_memory", "rho", [0.0, 0.005, 0.02, 0.05, 0.2, 0.5], {}),
    ("edge_memory", "sigma", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], {}),
]
SCHEMES = [("wave", 0.5), ("wave", 1.0), ("damp", 0.5), ("damp", 1.0), ("diff", None)]
STEPS = 500

def make(scheme, c, f, pf):
    if scheme == "diff": return C.make_diff_runner(f, eta=0.5, pf=pf, steps=STEPS)
    return C.make_runner(scheme, f, c=c, ps=dict(k=0.1) if scheme == "damp" else None, pf=pf, steps=STEPS)

def configs():
    cfgs = [(f"mean", "-", np.nan, "mean", None)]
    for f, p, vals, fixed in SERIES:
        for v in vals:
            cfgs.append((f, p, v, f, dict(fixed, **{p: v})))
    return cfgs

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 307))
    rows = []
    for name, p, v, f, pf in configs():
        for sch, c in SCHEMES:
            run = make(sch, c, f, pf)(g, u0)
            row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], force=name, param=p, value=v,
                       scheme=sch, c=c if c else np.nan)
            row.update(C.traj_summary(run))
            if run.ok.any():
                row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut", "fft_lowmode")))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else (0, 1))
    if quick: specs = specs[:2]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t307_param_sweeps_quick.csv" if quick else "t307_param_sweeps.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["force", "param", "value", "scheme", "c"], dropna=False)[["fail_frac", "ari_edge_cut"]].mean().round(2).unstack(["scheme", "c"]).to_string())
