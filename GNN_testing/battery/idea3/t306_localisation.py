"""T3.06 instability localisation: first node exceeding |u|>THR, its hop distance to the nearest bridge endpoint."""
import i3
from i3 import C, np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

FORCES = ["trim", "owa", "vw", "median", "symowa_ext", "symowa_med", "odd_owa", "odd_vw", "owa_mc", "trim_mc", "gauss", "absvw", "mean"]
CS = (0.5, 1.0)
THRS = (2.0, 5.0)
STEPS = 500

def hop_dist(g, mask):
    n = g.n
    A = csr_matrix((np.ones(len(g.ei)), (g.ei, g.ej)), shape=(n, n)); A = A + A.T
    D = shortest_path(A, unweighted=True)
    ends = np.unique(np.concatenate([g.ei[mask], g.ej[mask]]))
    return D[:, ends].min(1), ends

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    dist, ends = hop_dist(g, meta["bridge_mask"])
    rng = np.random.default_rng(spec["seed"] * 1000 + 306)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for f in FORCES:
        for c in CS:
            run = C.run(g, "wave", f, u0, c, STEPS)
            X = np.abs(run.X)                                  # (T+1,R,n); zeros after failure
            for r in range(run.R):
                for thr in THRS:
                    hit = np.where((X[:, r, :] > thr).any(1))[0]
                    row = dict(graph=meta["id"], gkind=spec["block"], bridges=spec["bridges"], seed=spec["seed"], force=f, c=c, real=r, thr=thr,
                               fail=bool(run.fail[r]), fail_step=int(run.fail_step[r]), n=g.n, n_endpoints=len(ends),
                               frac_nodes_d0=float((dist == 0).mean()), frac_nodes_le1=float((dist <= 1).mean()))
                    if len(hit):
                        t = hit[0]; node = int(np.argmax(X[t, r]))
                        row.update(first_step=int(t), node=node, dist=int(dist[node]), deg=int(g.deg[node]), amp=float(X[t, r, node]),
                                   same_block_as_endpoint=True)
                    rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = C.DEV_SEEDS[:1] if quick else C.DEV_SEEDS
    specs = [dict(kind="two_block", block=b, bridges=k, sizes=(24, 24), wb=1.0, seed=s) for b in ("cycle", "clique", "sbm") for k in (1, 2, 4) for s in seeds]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t306_localisation_quick.csv" if quick else "t306_localisation.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    d = df[(df.thr == 5.0) & df["node"].notna()]
    d = d.assign(at_end=(d.dist == 0), near=(d.dist <= 1))
    print(d.groupby(["gkind", "force", "c"]).agg(n=("dist", "size"), at_end=("at_end", "mean"), near=("near", "mean"), base0=("frac_nodes_d0", "mean"), base1=("frac_nodes_le1", "mean"), deg=("deg", "mean")).round(2).to_string())
