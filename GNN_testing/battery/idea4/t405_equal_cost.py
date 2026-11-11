"""T4.05 equal-aggregation-cost comparison (mean base).  A K-term series costs K aggregations per step.  Budgets B in {160,320,640,1280}
aggregations; series with K terms is run for B/K steps; plain wave (K=1), diffusion (eta=.5) and damp(k=.1) c=1 get B steps.
Policies: c=0.9*c_pred per config, and c=0.5.  Graphs: core zoo, dev seeds 0-3 (48), R=8.  ARI of 4 readouts at the end of the budget,
steps-to-ARI>=0.9 in aggregation units.  Paired (graph) tests against plain wave with the same policy."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

BUDGETS = [160, 320, 640, 1280]
FINE = [20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 320, 400, 480, 640, 800, 960, 1280]   # aggregation units checked for steps-to-90
G = i4.weights
CONFIGS = [  # label, force, pf, weights(K)
    ("wave", "mean", {}, (1.0,)),
    ("hi_acc4", "hi_acc:mean", {}, G("hi_acc")),
    ("hi_alt4", "hi_alt:mean", {}, G("hi_alt")),
    ("hi_geo.2_K4", "hi_geo:mean", {}, G("hi_geo", 0.2)),
    ("hi_geoneg-.2_K4", "hi_geoneg:mean", {}, G("hi_geoneg", -0.2)),
    ("geoneg-.2_K2", "series:mean", dict(w=(1.0, -0.2)), (1.0, -0.2)),
    ("geoneg-.5_K3", "series:mean", dict(w=(1.0, -0.5, 0.25)), (1.0, -0.5, 0.25)),
]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []

    def record(label, pol, c, K, run):
        Ts_agg = sorted(set(FINE + BUDGETS))
        steps_grid = [T // K for T in Ts_agg]
        ar = i4.prefix_ari(run, g, truth, steps_grid)
        row = dict(graph=meta["id"], seed=spec["seed"], config=label, policy=pol, c=c, K=K, bridges=meta["n_bridges"],
                   block=meta["params"].get("block"), fail_frac=float(run.fail.mean()))
        for rd, a in ar.items():
            f, s = i4.first_reach(Ts_agg, a)
            row[f"agg90_{rd}"] = f
            row[f"agg90s_{rd}"] = s
            for B in BUDGETS:
                row[f"ari_{rd}_B{B}"] = a[Ts_agg.index(B)]
        rows.append(row)

    for label, fname, pf, w in CONFIGS:
        K = len(w)
        cp = i4.c_crit_pred(lam, w)
        for pol, c in (("0.9cp", 0.9 * cp), ("c0.5", 0.5)):
            run = C.run(g, "wave", fname, u0, c=c, steps=BUDGETS[-1] // K, pf=pf)
            record(label, pol, c, K, run)
    for pol in ("0.9cp", "c0.5"):    # baselines (same rows for both policies, labelled for pairing)
        record("diffusion", pol, 0.5, 1, C.run_diff(g, "mean", u0, 0.5, BUDGETS[-1]))
        record("damp0.1_c1", pol, 1.0, 1, C.run(g, "damp", "mean", u0, c=1.0, steps=BUDGETS[-1], ps=dict(k=0.1)))
    # direct spectral
    lab = i4.rx.spectral_labels(g)
    m = C.partition_metrics(g, lab, truth)
    rows.append(dict(graph=meta["id"], seed=spec["seed"], config="spectral", policy="0.9cp", ari_direct=m["ari"], bridges=meta["n_bridges"], block=meta["params"].get("block")))
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t405_equal_cost.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    d = df[df.config != "spectral"]
    for pol in ("0.9cp", "c0.5"):
        sub = d[d.policy == pol]
        print("policy", pol)
        print(sub.groupby("config", sort=False).agg(n=("graph", "count"), reach_fft=("agg90_fft_lowmode", lambda x: x.notna().mean()),
              reach_elp=("agg90_edge_cut_lp", lambda x: x.notna().mean()),
              fft_B160=("ari_fft_lowmode_B160", "mean"), fft_B640=("ari_fft_lowmode_B640", "mean"), fft_B1280=("ari_fft_lowmode_B1280", "mean"),
              elp_B160=("ari_edge_cut_lp_B160", "mean"), elp_B640=("ari_edge_cut_lp_B640", "mean"), elp_B1280=("ari_edge_cut_lp_B1280", "mean")).round(3).to_string())
    print("spectral direct ARI mean", df[df.config == "spectral"].ari_direct.mean())
