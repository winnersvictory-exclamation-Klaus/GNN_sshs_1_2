"""T6.13 symmetry: u0 vs -u0, and node relabelling equivariance, for all 50 spread forces."""
import i6lib as L
from i6lib import C, np
import rank_experiment as rx

STEPS = 100


def relabel(g, perm):
    edges = [(int(perm[i]), int(perm[j]), w) for i, j, w in g.edges]
    return rx.G(g.n, edges)


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    rng = np.random.default_rng(spec["seed"] + 99)
    perm = rng.permutation(g.n)
    g2 = relabel(g, perm)
    for ik in ("uniform", "rademacher"):
        u0 = C.init(ik, g, 4, np.random.default_rng(spec["seed"] * 7 + 1))
        u0p = np.empty_like(u0); u0p[:, perm] = u0
        for b in ["mean"] + [x for x in L.BASES if x != "mean"]:
            for tr in ["identity"] + L.TRANSFORMS:
                f = b if tr == "identity" else L.fname(tr, b)
                pf = None if tr == "identity" else L.pf_for(tr, 1.0)
                kw = dict(c=0.25, steps=STEPS, pf=pf, fail_amp=1e9)
                r1 = C.run(g, "wave", f, u0, **kw)
                r2 = C.run(g, "wave", f, -u0, **kw)
                r3 = C.run(g2, "wave", f, u0p, **kw)
                row = L.base_row(meta, block=spec["block"], init=ik, base=b, transform=tr)
                for t in (10, 50, 100):
                    a = r1.X[t]
                    nrm = np.sqrt((a ** 2).sum(1)) + 1e-300
                    row[f"neg_err_t{t}"] = float(np.median(np.sqrt(((a + r2.X[t]) ** 2).sum(1)) / nrm))
                    pa = r3.X[t][:, perm]
                    row[f"perm_err_t{t}"] = float(np.median(np.sqrt(((a - pa) ** 2).sum(1)) / nrm))
                row["amp_end"] = float(np.median(np.abs(r1.X[-1]).max(1)))
                rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs([0, 1], bridges=(2,))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t613_symmetry.csv", rows)
    print(len(rows), "rows")
