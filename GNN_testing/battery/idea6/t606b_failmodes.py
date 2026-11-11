"""T6.06b: is local-scope failure drift (DC) or oscillation?  C.fail_modes on local/global transforms."""
import i6lib as L
from i6lib import C, np


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for tr in ["quad_l", "sq_l", "invq_l", "sqrt_l", "tanh_l", "quad_g", "sq_g", "invq_g", "tanh_g"]:
        for c in (0.25, 0.5, 1.0):
            fm = C.fail_modes(g, "wave", L.fname(tr), u0, c, 500, None, L.pf_for(tr, 1.0))
            rows.append(L.base_row(meta, block=spec["block"], transform=tr, c=c, raw_fail=float(np.mean(fm["raw_fail"])),
                                   ac_fail=float(np.mean(fm["ac_fail"])), dc_only=float(np.mean(fm["dc_only"]))))
    return rows


if __name__ == "__main__":
    out = C.pmap(job, L.core_specs(range(4)), procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t606b_failmodes.csv", rows)
    print(len(rows), "rows")
