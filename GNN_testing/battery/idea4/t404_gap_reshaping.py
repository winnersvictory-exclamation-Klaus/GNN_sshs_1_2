"""T4.04 spectral-gap reshaping + steps-to-partition (mean base).  Configs: plain wave, hi_acc, hi_alt, hi_geo(.2), hi_geoneg(-.2), each at
c = 0.9*c_pred (max-speed policy) and c = 0.5 (fixed).  Graphs: core zoo, dev seeds 0-3 (48 graphs), R=8 uniform u0, T=1600.
Predictors: g(l3)/g(l2), omega_2, omega_3, kappa_g=g(lmax)/g(l2) (steps ~ 1/omega_2 ~ sqrt(kappa_g)/ (0.9*2)...).
Measured: ARI of fft_lowmode / edge_cut_lp / edge_cut / timeavg_2means on prefixes T in TS; steps_to_ari90 (first T with ARI>=0.9)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

TS = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]
CONFIGS = [("wave", "mean", {}, (1.0,)), ("hi_acc", "hi_acc:mean", {}, i4.weights("hi_acc")),
           ("hi_alt", "hi_alt:mean", {}, i4.weights("hi_alt")),
           ("hi_geo.2", "hi_geo:mean", {}, i4.weights("hi_geo", 0.2)),
           ("hi_geoneg-.2", "hi_geoneg:mean", {}, i4.weights("hi_geoneg", -0.2))]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for label, fname, pf, w in CONFIGS:
        cp = i4.c_crit_pred(lam, w)
        for pol, c in (("0.9cp", 0.9 * cp), ("c0.5", 0.5)):
            g2, g3, gm = (float(i4.g_of(lam[i], w)) for i in (1, 2, len(lam) - 1))
            row = dict(graph=meta["id"], seed=spec["seed"], config=label, policy=pol, c=c, bridges=meta["n_bridges"],
                       block=meta["params"].get("block"), g2=g2, g3=g3, gap_ratio=g3 / g2, kappa_g=gm / g2,
                       om2=np.arccos(1 - c * c * g2 / 2), om3=np.arccos(1 - c * c * g3 / 2),
                       om_sep=(np.arccos(1 - c * c * g3 / 2) - np.arccos(1 - c * c * g2 / 2)) / np.arccos(1 - c * c * g2 / 2))
            run = C.run(g, "wave", fname, u0, c=c, steps=TS[-1], pf=pf)
            row["fail_frac"] = float(run.fail.mean())
            ar = i4.prefix_ari(run, g, truth, TS)
            for rd, a in ar.items():
                f, s = i4.first_reach(TS, a)
                row[f"t90_{rd}"] = f
                row[f"t90s_{rd}"] = s
                row[f"ari_end_{rd}"] = a[-1]
                row[f"ari_best_{rd}"] = np.nanmax(a) if np.isfinite(a).any() else np.nan
                for T, v in zip(TS, a):
                    row[f"ari_{rd}_T{T}"] = v
            rows.append(row)
    # direct spectral reference
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:4])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t404_gap_reshaping.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["policy", "config"], sort=False).agg(n=("graph", "count"), gap=("gap_ratio", "mean"), kappa=("kappa_g", "mean"),
          om2=("om2", "mean"), reach_fft=("t90_fft_lowmode", lambda x: x.notna().mean()),
          reach_elp=("t90_edge_cut_lp", lambda x: x.notna().mean()), reach_ec=("t90_edge_cut", lambda x: x.notna().mean()),
          med_t90_fft=("t90_fft_lowmode", "median"), med_t90_elp=("t90_edge_cut_lp", "median"),
          ari_end_fft=("ari_end_fft_lowmode", "mean"), ari_end_elp=("ari_end_edge_cut_lp", "mean")).round(3).to_string())
