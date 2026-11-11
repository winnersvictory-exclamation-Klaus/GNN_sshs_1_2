"""T6.08 mode transfer: high-mode energy fraction and temporal Nyquist fraction over time."""
import i6lib as L
from i6lib import C, np

TIMES = [0, 25, 50, 100, 200, 350, 500]
INITS = ["uniform", "low_mode"]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for ik in INITS:
        rng = np.random.default_rng(spec["seed"] * 1000 + 17)
        u0 = C.init(ik, g, 8, rng)
        cfgs = [("mean", None, 0.25), ("mean", None, 0.5)]
        for tr in L.TRANSFORMS:
            cfgs.append((tr, 1.0, 0.25))
        for (tr, gam, c) in cfgs:
            f = "mean" if tr == "mean" else L.fname(tr)
            run = C.run(g, "wave", f, u0, c=c, steps=500, pf=L.pf_for(tr, 1.0) if tr != "mean" else None)
            row = L.base_row(meta, block=spec["block"], init=ik, transform=tr, c=c, fail_frac=float(1 - run.ok.mean()))
            if run.ok.any():
                me = C.mode_energy(run.X, g, run.fail_step)
                for t in TIMES:
                    row[f"high_t{t}"] = float(np.nanmedian(me["high"][t]))
                    row[f"low_t{t}"] = float(np.nanmedian(me["low"][t]))
                ts = C.temporal_spectrum(run.X, run.fail_step, start_frac=0.5)
                row["nyq"] = float(np.nanmedian(ts["nyquist"]))
                row["centroid"] = float(np.nanmedian(ts["centroid"]))
                row["high_late"] = float(np.nanmedian(np.nanmean(me["high"][-100:], axis=0)))
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t608_modes.csv", rows)
    print(len(rows), "rows")
