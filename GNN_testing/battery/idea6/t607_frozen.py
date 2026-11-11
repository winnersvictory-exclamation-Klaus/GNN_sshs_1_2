"""T6.07 adaptive vs frozen scale s."""
import i6lib as L
from i6lib import C, np

CS = [0.25, 0.5, 1.0, 1.4]
SCALES = [0.3, 1.0, 3.0]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for sc in SCALES:
        rng = np.random.default_rng(spec["seed"] * 1000 + 17)
        u0 = C.init("uniform", g, 8, rng, scale=sc)
        for c in CS:
            for tr in L.TRANSFORMS:
                pf = L.pf_for(tr, 1.0)
                for variant, f in (("adaptive", L.fname(tr)), ("frozen", f"fz_{tr}:mean")):
                    run = C.run(g, "wave", f, u0, c=c, steps=500, pf=pf)
                    row = L.base_row(meta, block=spec["block"], u_scale=sc, c=c, transform=tr, variant=variant)
                    row.update(C.traj_summary(run))
                    if run.ok.any() and truth is not None:
                        ev = C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut",))
                        row["ari_edge_cut"] = ev.get("ari_edge_cut", np.nan)
                    rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t607_frozen.csv", rows)
    print(len(rows), "rows")
