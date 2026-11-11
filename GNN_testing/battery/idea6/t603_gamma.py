"""T6.03 gamma sweep: quad/invq x scope x c, mean base, wave scheme."""
import i6lib as L
from i6lib import C, np

GAMMAS = np.logspace(-2, 1, 9)
CS = [0.25, 0.5, 1.0, 1.4]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for c in CS:
        run = C.run(g, "wave", "mean", u0, c=c, steps=500)
        rows.append(L.full_row(run, g, truth, meta, spec["block"], transform="mean", gamma=np.nan, scope="-", c=c))
        for mode in ("quad", "invq"):
            for scope in ("g", "l"):
                tr = f"{mode}_{scope}"
                for gam in GAMMAS:
                    run = C.run(g, "wave", L.fname(tr), u0, c=c, steps=500, pf={"gamma": float(gam)})
                    rows.append(L.full_row(run, g, truth, meta, spec["block"], transform=tr, gamma=float(gam), scope=scope, c=c))
    return rows


if __name__ == "__main__":
    specs = L.core_specs(C.DEV_SEEDS)
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t603_gamma.csv", rows)
    print(len(rows), "rows")
