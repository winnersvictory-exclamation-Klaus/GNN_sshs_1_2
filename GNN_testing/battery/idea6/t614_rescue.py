"""T6.14 rescue: minimal damp k; limiter / saturate / composites."""
import i6lib as L
from i6lib import C, np

CS = [0.5, 1.0]
KS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
OTHER = [("limiter", dict(vmax=0.1)), ("limiter", dict(vmax=0.3)), ("limiter", dict(vmax=1.0)),
         ("saturate", dict(kappa=0.5)), ("saturate", dict(kappa=1.0)), ("saturate", dict(kappa=2.0)),
         ("limdamp", dict(vmax=0.3, k=0.1)), ("satdamp", dict(kappa=1.0, k=0.1))]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []

    def rec(f, pf, sch, ps, c, label, par):
        run = C.run(g, sch, f, u0, c=c, steps=1000, ps=ps, pf=pf)
        row = L.base_row(meta, block=spec["block"], transform=f.split(":")[0], base=f.split(":")[-1], c=c, scheme=sch, par=par, label=label)
        row.update(C.traj_summary(run))
        if run.ok.any() and truth is not None:
            ev = C.evaluate_partitions(run.X, g, truth, run.fail_step, which=("edge_cut",))
            row["ari_edge_cut"] = ev.get("ari_edge_cut", np.nan)
            row["edge_margin"] = ev.get("edge_margin", np.nan)
        rows.append(row)

    for c in CS:
        for f, tr in [("mean", "mean")] + [(L.fname(t), t) for t in L.TRANSFORMS]:
            pf = None if tr == "mean" else L.pf_for(tr, 1.0)
            for k in KS:
                rec(f, pf, "damp", dict(k=k), c, f"damp k={k}", k)
            for sch, ps in OTHER:
                rec(f, pf, sch, ps, c, f"{sch} {ps}", np.nan)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t614_rescue.csv", rows)
    print(len(rows), "rows")
