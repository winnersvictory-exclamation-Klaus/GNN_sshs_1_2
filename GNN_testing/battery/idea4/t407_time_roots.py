"""T4.07 time-series root analysis.  Characteristic polynomial derived from wt.time_even_step AS WRITTEN (see i4.time_poly, verified to
1e-15 against simulated modal recurrences).  Part A: on the (e4,e6) grid compute (i) parasitic roots at s=c^2 lam -> 0,
(ii) physical-branch growth rate gamma(s) and its small-s law, (iii) s*(e4,e6) = largest s with rho(s')<=1 for all s'<=s.
Part B: root vs simulation (exponent of modal amplitude).  Part C: 2-D simulation sweep (15x15 grid x c x 2 graphs) vs prediction.
Also t4 (single-term, reference only) for the explanation of the known 100% failures."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

VALS = [-0.5, -0.2, -0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5]
S_GRID = np.concatenate([np.logspace(-6, -2, 17), np.linspace(0.011, 4.2, 840)])
TOL = 1e-9


def analyse(eps):
    rho = i4.time_rho(eps, S_GRID)
    # parasitic roots at s -> 0: roots other than the (double) root at 1 (remove two roots closest to 1)
    r0 = i4.time_roots(eps, 1e-12)
    order = np.argsort(abs(r0 - 1))
    para = r0[order[2:]]
    rho_para0 = np.abs(para).max() if len(para) else 0.0
    bad = np.where(rho > 1 + TOL)[0]
    s_star = S_GRID[bad[0] - 1] if len(bad) and bad[0] > 0 else (0.0 if len(bad) else S_GRID[-1])
    # physical branch growth at small s: gamma = log|z_phys|; fit gamma ~ kappa s^p on s in [1e-3, 1e-2]
    ss = np.array([1e-3, 2e-3, 4e-3, 8e-3])
    gam = np.array([np.log(abs(i4.time_phys_root(eps, s))) for s in ss])
    return dict(rho_para0=rho_para0, s_star=s_star, rho_max=rho.max(), gamma_1e3=gam[0], gamma_8e3=gam[-1],
                kappa=gam[-1] / ss[-1] ** 2)


def partA():
    rows = []
    for e4 in VALS:
        for e6 in VALS:
            a = analyse((e4, e6))
            a.update(e4=e4, e6=e6, kappa_theory=e4 / 2)   # small-s law gamma ~ e4 s^2/2 - e6 s^3
            rows.append(a)
    for e in [x for x in np.round(np.linspace(-1, 1, 41), 3)]:
        a = analyse((e,))
        a.update(e4=e, e6=np.nan, kappa_theory=e / 2)
        rows.append(a)           # t4 reference rows have e6=NaN
    return rows


def sim_growth(spec, eps, c, T=300):
    """measured exponent of modal amplitudes (max over modes, running-max envelope) vs predicted log rho."""
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(1)
    u0 = C.init("uniform", g, 2, rng)
    run = C.run(g, "t6g", "mean", u0, c=c, steps=T, ps=dict(eps=tuple(eps)), fail_amp=1e30)
    A = np.einsum("trn,n,nk->trk", run.X, d, Phi)[:, 0, 1:]
    out = []
    for k in range(A.shape[1]):
        a = np.abs(A[:, k])
        if not np.all(np.isfinite(a)) or a[-1] == 0:
            continue
        blocks = np.maximum.reduceat(a[100:], np.arange(0, T - 100, 20))
        if (blocks <= 1e-250).any():
            continue
        emp = np.polyfit(np.arange(len(blocks)) * 20.0, np.log(blocks), 1)[0]
        th = np.log(i4.time_rho(eps, c * c * lam[k + 1])[0])
        out.append((k + 1, lam[k + 1], th, emp))
    return out


def partB():
    rows = []
    spec = dict(kind="two_block", block="cycle", sizes=(24, 24), bridges=2, seed=0)
    pts = [((0.1, 0.01), 0.5), ((0.1, 0.01), 1.0), ((-0.1, -0.01), 0.5), ((-0.1, -0.01), 1.0), ((-0.01, 0.0), 1.0), ((-0.1, 0.0), 1.0),
           ((0.0, -0.003), 0.5), ((0.0, 0.01), 1.0), ((-0.03, 0.003), 1.0), ((-1 / 12, 1 / 90), 1.0), ((-0.1, 0.01), 1.4), ((0.003, 0.0), 1.0)]
    for eps, c in pts:
        for k, lm, th, emp in sim_growth(spec, eps, c):
            rows.append(dict(e4=eps[0], e6=eps[1], c=c, mode=k, lam=lm, s=c * c * lm, theta=th, emp=emp))
    return rows


def sim_cell(args):
    spec, e4, e6, c = args
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(1)
    u0 = C.init("uniform", g, 4, rng)
    T = 1000
    run = C.run(g, "t6g", "mean", u0, c=c, steps=T, ps=dict(eps=(e4, e6)))
    rho = i4.time_rho((e4, e6), c * c * lam[1:])
    rmax = rho.max()
    return dict(graph=meta["id"], e4=e4, e6=e6, c=c, rho_pred=rmax, grow_T=np.log(max(rmax, 1e-300)) * T,
                fail_frac=float(run.fail.mean()), mean_fail_step=float(np.mean(run.fail_step[run.fail])) if run.fail.any() else np.nan,
                lam_max=lam[-1])


if __name__ == "__main__":
    A = pd.DataFrame(partA())
    C.write_csv(i4.RES / "t407_partA_roots.csv", A)
    B = pd.DataFrame(partB())
    C.write_csv(i4.RES / "t407_partB_rootsim.csv", B)
    specs = [dict(kind="two_block", block="cycle", sizes=(24, 24), bridges=2, seed=0),
             dict(kind="two_block", block="clique", sizes=(24, 24), bridges=2, seed=0)]
    jobs = [(sp, e4, e6, c) for sp in specs for e4 in VALS for e6 in VALS for c in (0.1, 0.5, 1.0)]
    rows = C.pmap(sim_cell, jobs, procs=2)
    Cdf = pd.DataFrame(rows)
    C.write_csv(i4.RES / "t407_partC_sweep.csv", Cdf)
    print("Part A done", len(A), "Part B", len(B), "Part C", len(Cdf))
