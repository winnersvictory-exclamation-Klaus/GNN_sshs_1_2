"""T3.15 degree effects: sbm_het sigma sweep and random_regular d in {3,4,6,8}: fail and ARI vs degree heterogeneity / low degree."""
import i3
from i3 import C, np
import pandas as pd
from t310_partition import FORCES, runner

def specs(seeds):
    out = [dict(kind="sbm_het", sizes=(24, 24), bridges=2, wb=1.0, sigma=s, seed=sd) for s in (0.0, 0.3, 0.6, 1.0) for sd in seeds]
    out += [dict(kind="random_regular", n=48, d=d, seed=sd) for d in (3, 4, 6, 8) for sd in seeds]
    return out

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    dg = C.wdeg(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 315))
    rows = []
    for f in FORCES:
        for sk in ("wave", "damp", "diff"):
            run = runner(sk, f)(g, u0)
            row = dict(graph=meta["id"], gkind=spec["kind"], sigma=spec.get("sigma", np.nan), d=spec.get("d", np.nan), seed=spec["seed"], n=g.n,
                       deg_cv=float(dg.std() / dg.mean()), deg_min=int(g.deg.min()), deg_max=int(g.deg.max()), frac_deg_le3=float((g.deg <= 3).mean()),
                       force=f, scheme=sk)
            row.update(C.traj_summary(run))
            if truth is not None and run.ok.any():
                row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut", "fft_lowmode")))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    sp = specs((0,) if quick else (0, 1, 2, 3))
    if quick: sp = sp[:1] + sp[-1:]
    out = C.pmap(job, [(s,) for s in sp], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t315_degree_quick.csv" if quick else "t315_degree.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df[df.scheme == "damp"].groupby(["force", "gkind", "sigma", "d"], dropna=False)[["fail_frac", "ari_edge_cut"]].mean().round(2).unstack(["gkind", "sigma", "d"]).to_string())
