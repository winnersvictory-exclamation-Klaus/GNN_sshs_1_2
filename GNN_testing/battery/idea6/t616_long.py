"""T6.16 long horizon: fail_frac(T) for T in {500,2000,5000} near the empirical threshold."""
import i6lib as L
from i6lib import C, np

T = 5000
FACS = [0.5, 0.8, 0.9, 0.97, 1.0, 1.03, 1.1]


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    rows = []
    for tr in L.TRANSFORMS:
        f = L.fname(tr); pf = L.pf_for(tr, 1.0)
        emp = C.c_crit_empirical(g, "wave", f, u0=u0, pf=pf, steps=500)
        c500 = emp["c_crit"]
        cl = [("fac", fac, fac * c500) for fac in FACS] if np.isfinite(c500) and c500 > 0 else []
        cl += [("fixed", c, c) for c in (0.25, 0.5, 1.0)]
        for kind, par, c in cl:
            run = C.run(g, "wave", f, u0, c=c, steps=T, pf=pf)
            fs = run.fail_step
            row = L.base_row(meta, block=spec["block"], transform=tr, c500=c500, c_status=emp["status"], kind_c=kind, par=par, c=c)
            for h in (500, 2000, 5000):
                row[f"fail_T{h}"] = float(((fs >= 0) & (fs < h)).mean())
            row["median_fail_step"] = float(np.median(fs[fs >= 0])) if (fs >= 0).any() else np.nan
            row["growth_rate"] = float(np.nanmedian(C.traj_metrics(run)["growth_rate"])) if run.ok.any() else np.nan
            row["dead_frac"] = float(C.traj_metrics(run)["dead"].mean()) if run.ok.any() else np.nan
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4), bridges=(2,))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t616_long.csv", rows)
    print(len(rows), "rows")
