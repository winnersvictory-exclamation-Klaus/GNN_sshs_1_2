"""T3.16 amplitude regime: fixed-threshold forces under u0 scaling and under decay; deviation from the mean-force trajectory."""
import i3
from i3 import C, np
import pandas as pd

FORCES = ["huber", "tanh", "amp", "gauss", "absvw", "edge_memory", "blend_agree", "huber_anneal", "amp_gated",   # fixed-threshold / scale-aware
          "trim", "symowa_med", "huber_local", "huber_global", "odd_owa", "odd_vw"]                                 # (near-)homogeneous controls
SCALES = (0.01, 0.1, 1.0, 10.0)
SCH = [("damp", dict(scheme="damp", c=1.0, ps=dict(k=0.1))), ("wave", dict(scheme="wave", c=0.5))]
STEPS = 1000

def rel(a, b, sl):
    A = a[sl]; B = b[sl]
    return float(np.sqrt(((A - B) ** 2).sum()) / max(np.sqrt((B ** 2).sum()), 1e-300))

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    u1 = C.init("uniform", g, 4, np.random.default_rng(spec["seed"] * 1000 + 316))
    rows = []
    for sk, kw in SCH:
        for s in SCALES:
            fa = 20.0 * s
            ref = C.run(g, kw["scheme"], "mean", u1 * s, kw["c"], STEPS, ps=kw.get("ps"), fail_amp=fa)
            for f in FORCES:
                run = C.run(g, kw["scheme"], f, u1 * s, kw["c"], STEPS, ps=kw.get("ps"), fail_amp=fa)
                one = C.run(g, kw["scheme"], f, u1, kw["c"], STEPS, ps=kw.get("ps"))
                row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], scheme=sk, scale=s, force=f,
                           fail_frac=float(run.fail.mean()))
                ok = run.ok & ref.ok & one.ok
                if ok.any():
                    X, Xm, X1 = run.X[:, ok], ref.X[:, ok], one.X[:, ok]
                    row["dev_mean_early"] = rel(X, Xm, slice(0, 101)); row["dev_mean_mid"] = rel(X, Xm, slice(300, 501)); row["dev_mean_late"] = rel(X, Xm, slice(-100, None))
                    row["scale_inv_defect"] = rel(X / s, X1, slice(None))
                    ac = X - X.mean(2, keepdims=True)
                    row["rms_late_over_scale"] = float(np.sqrt((ac[-100:] ** 2).mean()) / s)
                    row["rms_late_ref_over_scale"] = float(np.sqrt(((Xm - Xm.mean(2, keepdims=True))[-100:] ** 2).mean()) / s)
                rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = (0,) if quick else (0, 1, 2, 3)
    sp = [dict(kind="two_block", block=b, bridges=2, sizes=(24, 24), wb=1.0, seed=s) for b in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, [(s,) for s in sp], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t316_amplitude_quick.csv" if quick else "t316_amplitude.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["scheme", "force", "scale"])[["fail_frac", "dev_mean_early", "dev_mean_late", "scale_inv_defect"]].median().round(4).unstack("scale").to_string())
