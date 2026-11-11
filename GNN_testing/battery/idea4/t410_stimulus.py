"""T4.10 receptive field / stimulus (mean base).  (a) exact light cone: noiseless impulse at the near node farthest from the far block; first step
with |u_far|>1e-12 vs BFS hop distance (K-term series should reach K hops/step).  (b) C.stimulus (amps .2/1/5, signs +-, sigma .05, R=8, T=200,
det_trials=32) for plain wave and series, policies c=0.5 and c=0.9*c_pred.  Arrival is reported in steps and in aggregation units (steps*K).
Graphs: core zoo dev seeds 0-3 (48), n0=24."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
from collections import deque
C = i4.C

G = i4.weights
CONFIGS = [("wave", "mean", {}, (1.0,)), ("hi_acc4", "hi_acc:mean", {}, G("hi_acc")), ("hi_alt4", "hi_alt:mean", {}, G("hi_alt")),
           ("hi_geo.2_K4", "hi_geo:mean", {}, G("hi_geo", 0.2)), ("hi_geoneg-.2_K4", "hi_geoneg:mean", {}, G("hi_geoneg", -0.2))]


def bfs_far_dist(g, src, far):
    nb = [[] for _ in range(g.n)]
    for i, j in zip(g.ei, g.ej):
        nb[i].append(j); nb[j].append(i)
    dist = {src: 0}
    q = deque([src])
    while q:
        a = q.popleft()
        for b in nb[a]:
            if b not in dist:
                dist[b] = dist[a] + 1
                q.append(b)
    return min(dist[i] for i in range(g.n) if far[i]), dist


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    n0 = 24
    far = np.arange(g.n) >= n0
    # source = near node with maximal hop distance to the far block
    dists = [bfs_far_dist(g, s, far)[0] for s in range(n0)]
    src = int(np.argmax(dists)); hop = int(max(dists))
    rows = []
    for label, fname, pf, w in CONFIGS:
        K = len(w)
        cp = i4.c_crit_pred(lam, w)
        for pol, c in (("0.9cp", 0.9 * cp), ("c0.5", 0.5)):
            u0 = np.zeros((1, g.n)); u0[0, src] = 1.0
            run = C.run(g, "wave", fname, u0, c=c, steps=40, pf=pf)
            resp = np.abs(run.X[:, 0, :][:, far]).max(1)
            first = int(np.argmax(resp > 1e-12)) if (resp > 1e-12).any() else -1
            row = dict(graph=meta["id"], seed=spec["seed"], config=label, policy=pol, c=c, K=K, bridges=meta["n_bridges"], block=meta["params"].get("block"),
                       hop_dist=hop, first_step=first, expected_step=int(np.ceil(hop / K)))
            runner = C.make_runner("wave", fname, c=c, pf=pf, steps=200)
            st = C.stimulus(g, n0, runner, R=8)
            df = pd.DataFrame(st)
            for amp in (0.2, 1, 5):
                sub = df[df.amp == amp]
                for key in ("far_peak", "transmit_ratio", "arrival", "snr", "noise_gain", "det_auc", "fail_frac"):
                    row[f"{key}_a{amp}"] = float(sub[key].mean())
            row["arrival_agg_a1"] = row["arrival_a1"] * K
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t410_stimulus.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print((df.first_step == df.expected_step).groupby(df.config).mean())
    print(df.groupby(["policy", "config"], sort=False)[["arrival_a1", "arrival_agg_a1", "far_peak_a1", "transmit_ratio_a1", "snr_a1", "noise_gain_a1", "det_auc_a1"]].mean().round(3).to_string())
