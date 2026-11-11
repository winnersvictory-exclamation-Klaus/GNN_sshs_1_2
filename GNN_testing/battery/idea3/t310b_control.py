"""T3.10 follow-up control: is edge_memory's raw-wave gain just a smaller effective c?  wave+mean over a c grid vs edge_memory (c=.5)
with c_eff = c*sqrt(mean late edge weight) reconstructed from the trajectory."""
import i3
from i3 import C, np
import pandas as pd

CS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
STEPS = 500

def weights_from_traj(g, X, rho, sigma):
    x = X[:, :, g.ej] - X[:, :, g.ei]             # (T+1,R,m) ; w symmetric in x -> sign irrelevant
    w = np.ones(x.shape[1:]); W = np.empty_like(x[:-1])
    for t in range(x.shape[0] - 1):
        W[t] = w
        w = (1 - rho) * w + rho * np.exp(-(x[t] / sigma) ** 2)
    return W                                       # weight USED at step t

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    mask = meta["bridge_mask"]
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 3100))
    rows = []
    def add(label, run, **kw):
        row = dict(graph=meta["id"], gkind=spec["block"], bridges=meta["n_bridges"], seed=spec["seed"], config=label, **kw)
        row.update(C.traj_summary(run))
        if run.ok.any(): row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut", "edge_cut_lp", "fft_lowmode")))
        rows.append(row)
    for c in CS:
        add(f"wave mean c={c}", C.run(g, "wave", "mean", u0, c, STEPS), c=c)
    em = C.run(g, "wave", "edge_memory", u0, 0.5, STEPS)
    W = weights_from_traj(g, em.X, 0.05, 0.5)
    wl = W[-250:].mean((0, 1))                                   # (m,) late-window mean weight per edge
    wd = C.wdeg(g)
    ceff = 0.5 * np.sqrt(wl.mean())
    add("wave edge_memory c=0.5", em, c=0.5, w_late_mean=float(wl.mean()), w_late_bridge=float(wl[mask].mean()), w_late_intra=float(wl[~mask].mean()), c_eff=float(ceff))
    add("wave mean c=c_eff", C.run(g, "wave", "mean", u0, float(ceff), STEPS), c=float(ceff), c_eff=float(ceff))
    for rho in (0.005, 0.2):
        add(f"wave edge_memory rho={rho}", C.run(g, "wave", "edge_memory", u0, 0.5, STEPS, pf=dict(rho=rho)), c=0.5)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    specs = C.zoo("core", seeds=(0,) if quick else C.DEV_SEEDS[:4])
    if quick: specs = specs[:2]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t310b_control_quick.csv" if quick else "t310b_control.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    df["pen"] = df.ari_edge_cut.fillna(0)
    print(df.groupby("config", sort=False)[["pen", "ari_edge_cut_lp", "ari_fft_lowmode", "edge_margin", "bridge_auc", "c_eff", "w_late_mean", "w_late_bridge", "w_late_intra"]].mean().round(3).to_string())
