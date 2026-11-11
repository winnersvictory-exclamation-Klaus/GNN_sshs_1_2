"""T6.05 block_offset initial conditions: lifetime of inter-block contrast vs plain mean."""
import i6lib as L
from i6lib import C, np

DELTAS = [0.0, 0.25, 0.5, 1.0, 2.0]
TRS = L.TRANSFORMS
# (label, kind, scheme/force, par, pf)
BASE_CFG = [("wave+mean c=.25", "wave", "mean", 0.25, None, None), ("wave+mean c=.5", "wave", "mean", 0.5, None, None),
            ("damp.1+mean c=.5", "damp", "mean", 0.5, dict(k=0.1), None), ("diff+mean eta=.25", "diff", "mean", 0.25, None, None),
            ("diff+mean eta=.5", "diff", "mean", 0.5, None, None)]


def configs():
    cf = list(BASE_CFG)
    for tr in TRS:
        pf = L.pf_for(tr, 1.0)
        cf.append((f"wave+{tr}:mean c=.25", "wave", L.fname(tr), 0.25, None, pf))
        cf.append((f"diff+{tr}:mean eta=.25", "diff", L.fname(tr), 0.25, None, pf))
    for tr in ("quad_g", "quad_l"):
        cf.append((f"wave+{tr}(g=.1):mean c=.25", "wave", L.fname(tr), 0.25, None, {"gamma": 0.1}))
        cf.append((f"diff+{tr}(g=.1):mean eta=.25", "diff", L.fname(tr), 0.25, None, {"gamma": 0.1}))
    return cf


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for delta in DELTAS:
        rng = np.random.default_rng(spec["seed"] * 1000 + 17)
        u0 = C.init("block_offset", g, 8, rng, truth=truth, delta=delta)
        for label, kind, f, par, ps, pf in configs():
            if kind == "diff":
                run = C.run_diff(g, f, u0, eta=par, steps=500, pf=pf)
            else:
                run = C.run(g, kind, f, u0, c=par, steps=500, ps=ps, pf=pf)
            row = L.base_row(meta, block=spec["block"], delta=delta, config=label, force=f, kind_dyn=kind, par=par,
                             gamma=(pf or {}).get("gamma", np.nan))
            row.update(C.traj_summary(run))
            row.update(L.contrast_metrics(run, g, truth))
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t605_blockoffset.csv", rows)
    print(len(rows), "rows")
