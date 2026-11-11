"""T6.17 negative controls + matching core graphs: false partitions (balanced cut & hold>0)."""
import i6lib as L
from i6lib import C, np

READS = ("edge_cut", "edge_cut_lp")


def cfgs():
    out = [("wave", dict(), 0.5, "mean", None, "base"), ("damp", dict(k=0.1), 0.5, "mean", None, "base"), ("diff", None, 0.5, "mean", None, "base")]
    for tr in L.TRANSFORMS:
        out.append(("damp", dict(k=0.1), 0.25, L.fname(tr), L.pf_for(tr, 1.0), tr))
        out.append(("diff", None, 0.25, L.fname(tr), L.pf_for(tr, 1.0), tr))
    out.append(("damp", dict(k=0.1), 0.25, "mean", None, "mean"))
    out.append(("diff", None, 0.25, "mean", None, "mean"))
    return out


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for sch, ps, par, f, pf, tr in cfgs():
        run = C.run_diff(g, f, u0, eta=par, steps=500, pf=pf) if sch == "diff" else C.run(g, sch, f, u0, c=par, steps=500, ps=ps, pf=pf)
        row = L.base_row(meta, gkind=spec.get("block", spec["kind"]), neg=truth is None, scheme=sch, par=par, transform=tr)
        row.update(C.traj_summary(run))
        if run.ok.any():
            for rd in READS:
                lab = C.partitions(run.X, g, 2, run.fail_step, which=(rd,)).get(rd)
                hh = C.hold(run.X, g, lab, fail_step=run.fail_step)
                h2 = C.hold(run.X, g, lab, fail_step=run.fail_step, norm="ac")
                pm = C.partition_metrics(g, lab, truth) if lab is not None else {}
                row[f"minfrac_{rd}"] = pm.get("minfrac", np.nan)
                row[f"cond_{rd}"] = pm.get("cond", np.nan)
                row[f"ari_{rd}"] = pm.get("ari", np.nan) if truth is not None else np.nan
                row[f"holdpos_{rd}"] = float(np.nanmean(hh["hold_pos"]))
                row[f"holdpos_ac_{rd}"] = float(np.nanmean(h2["hold_pos"]))
                row[f"degenerate_{rd}"] = float(lab is None or pm.get("minfrac", 0) < 0.1)
        rows.append(row)
    return rows


if __name__ == "__main__":
    specs = C.zoo("neg", seeds=C.DEV_SEEDS) + L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t617_neg.csv", rows)
    print(len(rows), "rows")
