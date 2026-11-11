"""T6.02b: why does linear prediction at u0 fail?  State-resolved Jacobian stiffness along runs just above c_emp."""
import i6lib as L
from i6lib import C, np
import warnings

GR = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=2, wb=1.0, seed=0) for b in ("cycle", "clique", "sbm")]
VARS = [("tanh_g", None), ("quad_g", 1.0), ("sq_g", None), ("invq_g", 1.0), ("sqrt_g", None)]


def stiff(ev):
    """ev = eig(-J).  returns (c_crit_state, max_imag_rel, min_real)."""
    sc = np.abs(ev).max()
    mr = ev.real.min()
    im = np.abs(ev.imag).max() / sc
    pos = ev.real.max()
    if mr < -1e-6 * sc or im > 1e-6:
        return 0.0, im, mr
    return 2 / np.sqrt(max(pos, 1e-12)), im, mr


def job(args):
    spec, (tr, gam) = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 31 + 7)
    u0 = C.init("uniform", g, 4, rng)
    f = L.fname(tr); pf = None if gam is None else {"gamma": gam}
    emp = C.c_crit_empirical(g, "wave", f, u0=u0, pf=pf, steps=2000)
    ce = emp["c_crit"]
    rows = []
    for fac in (0.9, 1.15):
        c = fac * ce
        run = C.run(g, "wave", f, u0, c=c, steps=1500, pf=pf)
        r = 0
        fs = run.fail_step[r]
        last = (fs if fs >= 0 else 1500) - 1
        ts = sorted(set(list(range(0, min(last, 100), 10)) + list(range(0, last, max(1, last // 25))) + [last]))
        lam, Phi, d = C.spectrum(g)
        for t in ts:
            u = run.X[t, r]
            dd = u - u.mean(); s = np.sqrt(np.mean(dd ** 2))
            if s < 1e-9:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                J = C.jacobian(g, f, u, pf)
            ev = np.linalg.eigvals(-J)
            cst, im, mr = stiff(ev)
            a = Phi.T @ (d * dd)
            hi = (a[-int(0.25 * (g.n - 1)):] ** 2).sum() / (a[1:] ** 2).sum()
            x = dd / s
            rows.append(L.base_row(meta, block=spec["block"], transform=tr, gamma=gam if gam else np.nan, c_emp=ce, fac=fac, c=c,
                                   fail_step=int(fs), t=t, c_state=cst, imag=im, min_real=mr, max_real=ev.real.max(),
                                   xmax=np.abs(x).max(), kurt=float(np.mean(x ** 4)), high_frac=float(hi), amp=float(np.abs(u).max())))
    return rows


if __name__ == "__main__":
    jobs = [(s, v) for s in GR for v in VARS]
    out = C.pmap(job, jobs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t602b_trajectory.csv", rows)
    print(len(rows), "rows")
