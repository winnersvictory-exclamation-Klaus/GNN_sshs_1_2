"""T6.15 generalisation: zoo wide (+scale) with baselines, selected configs and all transforms (diffusion eta=.25, damp k=.1 c=.25)."""
import json
import i6lib as L
from i6lib import C, np
from rank_experiment import spectral_labels
from sklearn.metrics import adjusted_rand_score

SEL = json.load(open(L.HERE / "selected.json"))


def all_cfgs():
    cf = list(L.BASELINES) + SEL
    for tr in L.TRANSFORMS:
        pf = L.pf_for(tr, 1.0)
        cf.append(dict(label=f"diff+{tr} eta=.25", kind="diff", force=L.fname(tr), pf=pf, par=0.25))
        cf.append(dict(label=f"damp.1+{tr} c=.25", kind="damp", force=L.fname(tr), pf=pf, ps=dict(k=0.1), par=0.25))
    return cf


def job(spec):
    g, truth, meta = C.make_graph(spec)
    q = meta["q"] or 2
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    # spectral baseline (q=2 only meaningful when q==2)
    if q == 2:
        rows.append(L.base_row(meta, gkind=spec["kind"], label="SPECTRAL", ari_edge_cut=adjusted_rand_score(truth, spectral_labels(g))))
    for cfg in all_cfgs():
        run = L.build_runner(cfg)(g, u0)
        row = L.base_row(meta, gkind=spec["kind"], label=cfg["label"], q=q)
        row.update(C.traj_summary(run))
        if run.ok.any() and truth is not None:
            ev = C.evaluate_partitions(run.X, g, truth, run.fail_step, q=q, which=("edge_cut", "edge_cut_lp"))
            row.update(ev)
            lab = C.partitions(run.X, g, q, run.fail_step, which=("edge_cut",)).get("edge_cut")
            if lab is not None:
                row["hold_rec"] = float(np.nanmean(C.hold(run.X, g, lab, fail_step=run.fail_step, norm="ac")["hold"]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    specs = C.zoo("wide", seeds=range(4)) + C.zoo("scale", seeds=range(2))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t615_generalise.csv", rows)
    print(len(rows), "rows")
