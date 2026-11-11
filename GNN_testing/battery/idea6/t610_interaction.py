"""T6.10 base x transform interaction: 5 bases x 10 transforms x c (wave)."""
import i6lib as L
from i6lib import C, np

CS = [0.25, 0.5, 1.0]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for c in CS:
        for b in L.BASES:
            run = C.run(g, "wave", b, u0, c=c, steps=500)
            row = L.base_row(meta, block=spec["block"], c=c, base=b, transform="identity")
            row.update(C.traj_summary(run)); rows.append(row)
            for tr in L.TRANSFORMS:
                run = C.run(g, "wave", L.fname(tr, b), u0, c=c, steps=500, pf=L.pf_for(tr, 1.0))
                row = L.base_row(meta, block=spec["block"], c=c, base=b, transform=tr)
                row.update(C.traj_summary(run))
                if run.ok.any():
                    row["tail_amp_ratio"] = row["rms_final"]
                rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t610_interaction.csv", rows)
    print(len(rows), "rows")
