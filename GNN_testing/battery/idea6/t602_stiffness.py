"""T6.02 predicted stiffness vs measured c_crit (global scope, mean base)."""
import i6lib as L
from i6lib import C, np, wt
import warnings

GRAPHS = L.core_specs([0, 1], bridges=(2,))
DISTS = ["uniform", "gauss", "student_t3", "rademacher", "impulse"]
VARIANTS = [("quad_g", 0.1), ("quad_g", 1.0), ("quad_g", 10.0), ("sq_g", None), ("sqrt_g", None),
            ("invq_g", 1.0), ("tanh_g", None)]


def dphi(d, s, mode, gam):
    x = d / s
    if mode == "quad":
        return 1 + 3 * gam * x ** 2
    if mode == "sq":
        return 2 * np.abs(x)
    if mode == "sqrt":
        return 0.5 / np.sqrt(np.maximum(np.abs(x), 1e-12))
    if mode == "invq":
        return (1 - gam * x ** 2) / (1 + gam * x ** 2) ** 2
    if mode == "tanh":
        return 1 - np.tanh(x) ** 2


def analytic(g, u, mode, gam):
    """J_a = -L_rw diag(phi'_d) (I-P)  (fixed-s partial derivative); returns eig(-J_a)."""
    n = g.n
    d = u - u.mean(); s = np.sqrt(np.mean(d ** 2)) + 1e-12
    ph = dphi(d, s, mode, gam)
    Lr = C.L_rw(g)
    M = Lr @ np.diag(ph) @ (np.eye(n) - np.ones((n, n)) / n)
    ev = np.linalg.eigvals(M)
    return ph, ev


def cc_from_ev(ev):
    """wave stable iff all eig(-J) real in [0, 4/c^2) (complex -> unstable for all c). returns (c_crit, max_imag_frac)"""
    ev = np.asarray(ev)
    im = np.abs(ev.imag).max() / max(np.abs(ev).max(), 1e-12)
    if (ev.real < -1e-9).any():
        return 0.0, im
    return 2 / np.sqrt(max(ev.real.max(), 1e-12)), im


def job(args):
    spec, dist, (tr, gam) = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 31 + 7)
    u0 = C.init(dist, g, 8, rng)
    mode = L.MODE[tr]
    f = L.fname(tr, "mean")
    pf = None if gam is None else {"gamma": gam}
    lam_mean = C.spectrum(g)[0][-1]
    c_mean = 2 / np.sqrt(lam_mean)
    # predictions at u0
    pa, cc_a, cc_s, cc_e, imf, ph_max = [], [], [], [], [], []
    for r in range(u0.shape[0]):
        ph, ev = analytic(g, u0[r], mode, gam)
        ph_max.append(ph.max())
        c_an, im = cc_from_ev(ev)
        cc_a.append(c_an); imf.append(im)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = C.force_props(g, f, u0[r], pf)
        cc_s.append(pr["c_crit_pred"]); cc_e.append(pr["c_crit_eig"])
    row = base_ = L.base_row(meta, block=spec["block"], dist=dist, transform=tr, gamma=gam if gam is not None else np.nan,
                             c_mean=c_mean, phimax_u0=float(np.max(ph_max)),
                             pred_naive=float(c_mean / np.sqrt(np.max(ph_max))),
                             pred_analytic_eig=float(np.min(cc_a)), pred_num_sym=float(np.min(cc_s)),
                             pred_num_eig=float(np.min(cc_e)), imag_frac=float(np.max(imf)))
    emp = C.c_crit_empirical(g, "wave", f, u0=u0, pf=pf, steps=2000)
    row.update(c_emp=emp["c_crit"], status=emp["status"], non_monotone=emp["non_monotone"])
    # trajectory-sampled prediction: run at 0.5*c_emp (stable), sample states
    c_run = 0.5 * emp["c_crit"] if np.isfinite(emp["c_crit"]) and emp["c_crit"] > 0 else (0.05 if emp["status"] == "unstable_at_lo" else 1.0)
    run = C.run(g, "wave", f, u0, c=c_run, steps=400, pf=pf)
    cts = []
    for t in range(0, 400, 20):
        for r in range(0, 8, 2):
            if run.fail_step[r] >= 0 and t >= run.fail_step[r]:
                continue
            u = run.X[t, r]
            if np.sqrt(np.mean((u - u.mean()) ** 2)) < 1e-8:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr = C.force_props(g, f, u, pf)
            cts.append(pr["c_crit_eig"])
    row["pred_traj_eig_min"] = float(np.min(cts)) if cts else np.nan
    row["traj_n"] = len(cts)
    return [row]


if __name__ == "__main__":
    import sys
    jobs = [(s, d, v) for s in GRAPHS for d in DISTS for v in VARIANTS]
    if len(sys.argv) > 1:
        jobs = jobs[:int(sys.argv[1])]
    out = C.pmap(job, jobs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / ("t602_stiffness.csv" if len(sys.argv) == 1 else "t602_tiny.csv"), rows)
    print(len(rows), "rows")
