"""T6.06 global vs local scope: DC drift, different block means, fail maps."""
import i6lib as L
from i6lib import C, np

CS = [0.25, 0.5, 1.0, 1.4]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for delta, dc in [(0.0, 0.0), (1.0, 0.0), (0.0, 3.0)]:
        rng = np.random.default_rng(spec["seed"] * 1000 + 17)
        u0 = C.init("block_offset", g, 8, rng, truth=truth, delta=delta, dc=dc)
        for c in CS:
            for tr in ["mean"] + L.TRANSFORMS:
                f = "mean" if tr == "mean" else L.fname(tr)
                # fail_amp relative to DC offset so a pure shift is not counted as failure
                run = C.run(g, "wave", f, u0, c=c, steps=500, pf=L.pf_for(tr, 1.0) if tr != "mean" else None, fail_amp=20 + abs(dc))
                row = L.base_row(meta, block=spec["block"], delta=delta, dc=dc, c=c, transform=tr,
                                 scope="-" if tr == "mean" else L.SCOPE[tr], mode="mean" if tr == "mean" else L.MODE[tr])
                row.update(C.traj_summary(run))
                row.update({k: v for k, v in L.contrast_metrics(run, g, truth).items() if k in ("env_ret_end", "rel_contrast_end", "hold_true_ac", "ac_rms_end")})
                rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t606_scope.csv", rows)
    print(len(rows), "rows")
