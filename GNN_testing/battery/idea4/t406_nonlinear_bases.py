"""T4.06 nonlinear bases.  Part 1: force_props of the pure terms G_k=(-1)^{k+1}F^{o k} (k=1..4) and of the composed series (hi_acc, hi_alt, hi_geo,
hi_geoneg) over bases mean/trim/owa/vw/huber, on core zoo seed 0 (12 graphs) x 3 random states: is G_k restoring (eig_sym_max<=1e-8)?
Part 2: fail map, wave at c in {0.25,0.5,1.0}, T=500, R=8, core zoo seeds 0-1 (24 graphs), every series x base, plus plain base forces."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

BASES = ("mean", "trim", "owa", "vw", "huber")
SERIES = ("hi_acc", "hi_alt", "hi_geo", "hi_geoneg")
CS = (0.25, 0.5, 1.0)
KEYS = ("oddness", "momentum_rel", "dsym_rel", "lam_max", "eig_sym_max", "jac_kink", "rho_J")


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 3)
    rows = []
    if spec["seed"] == 0:
        U = C.init("uniform", g, 3, rng)
        for b in BASES:
            forces = [(f"gk{k}", f"gk:{b}", dict(k=k)) for k in (1, 2, 3, 4)] + [(s, f"{s}:{b}", {}) for s in SERIES]
            for label, fname, pf in forces:
                for si, u in enumerate(U):
                    p = C.force_props(g, fname, u, pf)
                    rows.append(dict(part=1, graph=meta["id"], base=b, term=label, state=si, **{k: p[k] for k in KEYS}))
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    for b in BASES:
        forces = [("plain", b, {})] + [(s, f"{s}:{b}", {}) for s in SERIES]
        for label, fname, pf in forces:
            for c in CS:
                run = C.run(g, "wave", fname, u0, c=c, steps=500, pf=pf)
                m = C.traj_summary(run)
                rows.append(dict(part=2, graph=meta["id"], seed=spec["seed"], base=b, term=label, c=c, fail_frac=m["fail_frac"],
                                 mean_fail_step=m["mean_fail_step"], growth_rate=m["growth_rate"], dead_frac=m["dead_frac"]))
    return rows


if __name__ == "__main__":
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:2])
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t406_nonlinear_bases.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    p1 = df[df.part == 1]
    print(p1.groupby(["base", "term"]).agg(n=("graph", "count"), eig_sym_max=("eig_sym_max", "max"), restoring_frac=("eig_sym_max", lambda x: (x <= 1e-8).mean()),
          lam_max=("lam_max", "median"), momentum_rel=("momentum_rel", "median"), oddness=("oddness", "median"), kink=("jac_kink", "median")).round(4).to_string())
    p2 = df[df.part == 2]
    print(p2.pivot_table(index=["base", "term"], columns="c", values="fail_frac", aggfunc="mean").round(3).to_string())
