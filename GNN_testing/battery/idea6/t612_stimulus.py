"""T6.12 stimulus: far_peak vs amplitude, noise gain, detection AUC, sign symmetry."""
import i6lib as L
from i6lib import C, np

AMPS = (0.2, 1.0, 5.0, 20.0)
FA = 400.0     # fail threshold raised so amp=20 is not an instant failure


def runners():
    R = [("wave+mean c=.25", C.make_runner("wave", "mean", c=0.25, fail_amp=FA)),
         ("wave+mean c=.5", C.make_runner("wave", "mean", c=0.5, fail_amp=FA)),
         ("damp.1+mean c=.5", C.make_runner("damp", "mean", c=0.5, ps=dict(k=0.1), fail_amp=FA)),
         ("diff+mean eta=.25", C.make_diff_runner("mean", eta=0.25, fail_amp=FA))]
    for tr in L.TRANSFORMS:
        pf = L.pf_for(tr, 1.0)
        R.append((f"wave+{tr} c=.25", C.make_runner("wave", L.fname(tr), c=0.25, pf=pf, fail_amp=FA)))
        R.append((f"diff+{tr} eta=.25", C.make_diff_runner(L.fname(tr), eta=0.25, pf=pf, fail_amp=FA)))
    for tr in ("quad_g", "quad_l"):
        R.append((f"wave+{tr}(g=.1) c=.25", C.make_runner("wave", L.fname(tr), c=0.25, pf={"gamma": 0.1}, fail_amp=FA)))
        R.append((f"diff+{tr}(g=.1) eta=.25", C.make_diff_runner(L.fname(tr), eta=0.25, pf={"gamma": 0.1}, fail_amp=FA)))
    return R


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rows = []
    for label, rn in runners():
        for r in C.stimulus(g, 24, rn, amps=AMPS, steps=200, seed=spec["seed"]):
            row = L.base_row(meta, block=spec["block"], config=label)
            row.update(r)
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(C.DEV_SEEDS, bridges=(2, 4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t612_stimulus.csv", rows)
    print(len(rows), "rows")
