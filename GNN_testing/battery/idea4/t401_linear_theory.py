"""T4.01 linear theory (space), mean base: predicted stability 0<c^2 g(lam)<4, predicted vs empirical c_crit.
Grid: core zoo, dev seeds 0-1 (24 graphs) x 7 series forces; empirical c_crit by scan+bisection (T=2000, R=8),
plus stability classification at 0.98*c_pred and 1.02*c_pred (T=2000)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

CONFIGS = [  # label, force name, pf, weights
    ("mean", "mean", {}, (1.0,)),
    ("hi_acc", "hi_acc:mean", {}, i4.weights("hi_acc")),
    ("hi_alt", "hi_alt:mean", {}, i4.weights("hi_alt")),
    ("hi_geo.2", "hi_geo:mean", {}, i4.weights("hi_geo", 0.2)),
    ("hi_geoneg-.2", "hi_geoneg:mean", {}, i4.weights("hi_geoneg", -0.2)),
    ("hi_geo.5", "hi_geo:mean", dict(eps=0.5), i4.weights("hi_geo", 0.5)),
    ("hi_geo-.5", "hi_geo:mean", dict(eps=-0.5), i4.weights("hi_geo", -0.5)),
]


def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for label, fname, pf, w in CONFIGS:
        gv = i4.g_of(lam[1:], w)
        cp = i4.c_crit_pred(lam, w)
        row = dict(graph=meta["id"], seed=spec["seed"], config=label, n=meta["n"], lam2=lam[1], lam3=lam[2], lam_max=lam[-1],
                   g_min=gv.min(), g_max=gv.max(), n_neg=int((gv < -1e-12).sum()), c_pred=cp)
        if cp > 0:
            ce = C.c_crit_empirical(g, "wave", fname, u0=u0, pf=pf, steps=2000, rtol=0.005, R=8, c_hi=min(12.0, max(3.0, 1.5 * cp)))
            row.update(c_emp=ce["c_crit"], status=ce["status"], non_monotone=ce["non_monotone"])
            row["ff_below"] = C.fail_frac(g, "wave", fname, u0, 0.98 * cp, steps=2000, pf=pf)
            row["ff_above"] = C.fail_frac(g, "wave", fname, u0, 1.02 * cp, steps=2000, pf=pf)
            # modal growth rate prediction above threshold
            row["rho_above"] = float(i4.modal_rho_wave((1.02 * cp) ** 2 * gv).max())
        rows.append(row)
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:2])
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t401_linear_theory.csv", rows)
    df = pd.DataFrame(rows)
    df["rel_err"] = (df.c_emp - df.c_pred) / df.c_pred
    print(df.groupby("config", sort=False).agg(c_pred=("c_pred", "mean"), c_emp=("c_emp", "mean"), rel_err_mean=("rel_err", "mean"),
          rel_err_absmax=("rel_err", lambda x: np.nanmax(abs(x))), ff_below=("ff_below", "max"), ff_above=("ff_above", "min"),
          g_min=("g_min", "min"), n_neg=("n_neg", "max"), n=("graph", "count")).round(4))
