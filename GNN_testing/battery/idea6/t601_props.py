"""T6.01 property table + numerical safety of spread forces (all 50 transform x base forces)."""
import i6lib as L
from i6lib import C, np
import warnings

GRAPHS = [dict(kind="two_block", block="cycle", sizes=(24, 24), bridges=2, wb=1.0, seed=0),
          dict(kind="two_block", block="clique", sizes=(24, 24), bridges=2, wb=1.0, seed=0),
          dict(kind="two_block", block="sbm", sizes=(24, 24), bridges=2, wb=1.0, seed=0),
          dict(kind="grid_neck", a=5, w=2, seed=0)]


def states(g, truth, rng):
    n = g.n
    u = rng.uniform(-1, 1, n)
    imp = np.zeros(n); imp[3] = 1.0
    bl = np.where(truth == 0, 1.0, -1.0) if truth is not None else np.where(np.arange(n) < n // 2, 1.0, -1.0)
    return {"uniform": u, "impulse": imp, "near_consensus": 1.0 + 1e-6 * rng.normal(size=n),
            "const": np.full(n, 0.7), "block_const": bl, "tiny_1e-13": 1e-13 * u, "tiny_1e-9": 1e-9 * u,
            "huge_1e6": 1e6 * u}


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] + 5)
    rows = []
    for stname, u in states(g, truth, rng).items():
        for tr in L.TRANSFORMS:
            for base in L.BASES:
                for gam in ([0.1, 1.0, 10.0] if L.MODE[tr] in L.HAS_GAMMA else [None]):
                    pf = None if gam is None else {"gamma": gam}
                    f = L.fname(tr, base)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with np.errstate(all="ignore"):
                            Fu = C.eval_force(g, f, u[None], pf)[0]
                            try:
                                pr = C.force_props(g, f, u, pf)
                            except Exception as e:  # noqa
                                pr = {"err": str(e)[:60]}
                    fin = bool(np.isfinite(Fu).all())
                    rows.append(dict(graph=meta["id"], gkind=spec.get("block", spec["kind"]), state=stname, transform=tr,
                                     mode=L.MODE[tr], scope=L.SCOPE[tr], base=base, gamma=gam, finite=fin,
                                     F_rms=float(np.sqrt(np.mean(Fu ** 2))) if fin else np.nan,
                                     F_max=float(np.abs(Fu).max()) if fin else np.nan, **pr))
    return rows


if __name__ == "__main__":
    out = C.pmap(job, GRAPHS, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t601_props.csv", rows)
    print(len(rows), "rows")
