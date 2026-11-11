"""T6.11 partition + hold under wave (c=.25) and damp(k=.1) (c=.25): all readouts, edge_margin, hold."""
import i6lib as L
from i6lib import C, np

READS = C.PARTITION_NAMES if hasattr(C, "PARTITION_NAMES") else ("edge_cut", "edge_cut_lp", "timeavg_2means", "traj_corr_spectral", "fft_lowmode")


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    def add(run, **kw):
        rows.append(L.full_row(run, g, truth, meta, spec["block"], readouts=READS, **kw))
    for (sch, ps, c) in [("wave", None, 0.5), ("damp", dict(k=0.1), 0.5)]:
        add(C.run(g, sch, "mean", u0, c=c, steps=500, ps=ps), scheme=sch, transform="baseline", base="mean", c=c)
    for (sch, ps) in [("wave", None), ("damp", dict(k=0.1))]:
        c = 0.25
        for tr in L.TRANSFORMS:
            for b in (L.BASES if sch == "damp" else ["mean"]):
                run = C.run(g, sch, L.fname(tr, b), u0, c=c, steps=500, ps=ps, pf=L.pf_for(tr, 1.0))
                add(run, scheme=sch, transform=tr, base=b, c=c)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(C.DEV_SEEDS)
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t611_partition.csv", rows)
    print(len(rows), "rows")
