"""T6.04b: is the 'bimodality' of invq late states a single-node spike?  variance share of the top node + node degree."""
import i6lib as L
from i6lib import C, np


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    deg = g.A.sum(1)
    rows = []
    for tr, gam in [("mean", None), ("tanh_g", None), ("invq_g", 0.3), ("invq_g", 1.0), ("invq_g", 4.0), ("invq_l", 1.0), ("invq_l", 4.0)]:
        f = "mean" if tr == "mean" else L.fname(tr)
        pf = None if gam is None else {"gamma": gam}
        run = C.run_diff(g, f, u0, eta=0.25, steps=500, pf=pf)
        if not run.ok.any():
            continue
        X = run.X[:, run.ok]
        for t in (50, 150, 500):
            x = X[t] - X[t].mean(1, keepdims=True)
            sd = np.sqrt((x ** 2).sum(1))
            al = sd > 1e-9
            if not al.any():
                continue
            share = (x[al] ** 2).max(1) / (x[al] ** 2).sum(1)
            top = np.abs(x[al]).argmax(1)
            rows.append(L.base_row(meta, gkind=spec.get("block", spec["kind"]), neg=truth is None, transform=tr, gamma=gam if gam else np.nan, t=t,
                                   alive=float(al.mean()), top_share=float(np.median(share)), top_share_mean=float(share.mean()),
                                   top_deg_rel=float(np.median(deg[top] / deg.mean())),
                                   n2sig=float(np.median((np.abs(x[al]) > 2 * x[al].std(1, keepdims=True)).sum(1)))))
    return rows


if __name__ == "__main__":
    specs = C.zoo("neg", seeds=range(4)) + L.core_specs(range(2), bridges=(2, 4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t604b_spike.csv", rows)
    print(len(rows), "rows")
