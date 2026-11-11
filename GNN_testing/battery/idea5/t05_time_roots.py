"""T5.05 time-term root analysis (t4 family, mean force). Theory (quartic) vs simulation.
Outputs: results/t05_time_roots.csv (graph x e x c), results/t05_theory_grid.csv (e x s root table)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *
import pandas as pd

E_LIST = [-0.3, -0.25, -0.2, -0.15, -0.1, -1/12, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5]
C_LIST = [0.1, 0.25, 0.5, 0.75, 1.0, 1.4]
T = 2000


def theory_grid():
    rows = []
    es = sorted(set(list(np.round(np.arange(-0.30, 0.5001, 0.0125), 6)) + [-1/12, -0.001, 0.001]))
    ss = np.linspace(0.0, 6.0, 1201)[1:]
    for e in es:
        rhos = np.array([t4_rho(e, s) for s in ss])
        bad = np.where(rhos > 1 + 1e-9)[0]
        s_crit = ss[bad[0]] if len(bad) else np.inf
        # first-crossing refinement by bisection
        if np.isfinite(s_crit) and bad[0] > 0:
            lo, hi = ss[bad[0] - 1], ss[bad[0]]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                (hi, lo) = (mid, lo) if t4_rho(e, mid) > 1 + 1e-9 else (hi, mid)
            s_crit = hi
        s_small = 0.05
        rows.append(dict(eps=e, s_crit=s_crit, rho_small=t4_rho(e, s_small),
                         slope_rho_over_s2=(t4_rho(e, s_small) - 1) / s_small ** 2, pred_slope=e / 2,
                         rho_s1=t4_rho(e, 1.0), rho_s2=t4_rho(e, 2.0), rho_s39=t4_rho(e, 3.9),
                         rho_max_0_4=float(rhos[ss <= 4].max()),
                         # parasitic (root far from wave roots) at s=0.05
                         ))
        r = np.roots(t4_poly(e, s_small)); w = np.roots([1, s_small - 2, 1])
        phys = [r[np.argmin(abs(r - wi))] for wi in w]
        par = [x for x in r if all(abs(x - p) > 1e-12 for p in phys)]
        rows[-1]["par_abs_small_s"] = max(abs(x) for x in par) if par else np.nan
        rows[-1]["phys_abs_small_s"] = max(abs(x) for x in phys)
    return rows


def modal_predict(g, e, c, u0, T):
    lam, Phi, d = C.spectrum(g)
    a0 = (u0 * d) @ Phi          # (R,nm)
    s = c * c * lam
    hist = [a0] * 4
    traj = [a0]
    fail = np.full(u0.shape[0], -1)
    for t in range(T):
        h0, h1, h2, h3 = hist
        nxt = (-s * h0 - (-2 * h0 + h1 + e * (-4 * h0 + 6 * h1 - 4 * h2 + h3))) / (1 + e)
        u = nxt @ Phi.T
        bad = ~(np.abs(u) < wt.FAIL_AMP).all(1) & (fail < 0)
        fail[bad] = t + 1
        nxt[fail >= 0] = 0.0
        hist = [nxt] + hist[:-1]
    return fail


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for e in E_LIST:
        for c in C_LIST:
            r = C.run(g, "t4", "mean", u0, c, T, ps=dict(eps=e))
            pf = modal_predict(g, e, c, u0, T)
            rho_modes = np.array([t4_rho(e, c * c * l) for l in lam[1:]])
            tm = C.traj_metrics(r)
            fs = r.fail_step
            rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], n=meta["n"], eps=e, c=c,
                             lam_max=lam[-1], fail_frac=float((fs >= 0).mean()),
                             med_fail_step=float(np.median(fs[fs >= 0])) if (fs >= 0).any() else np.nan,
                             pred_fail_frac=float((pf >= 0).mean()),
                             pred_med_fail_step=float(np.median(pf[pf >= 0])) if (pf >= 0).any() else np.nan,
                             fail_step_exact_match=bool(np.array_equal(fs, pf)),
                             max_fail_step_diff=int(np.abs(fs - pf).max()),
                             pred_ln_rho_max=float(np.log(rho_modes.max())),
                             pred_mode_argmax=int(1 + np.argmax(rho_modes)),
                             meas_growth_rate=float(np.nanmedian(tm["growth_rate"])) if np.isfinite(tm["growth_rate"]).any() else np.nan,
                             ac_final=float(np.nanmedian(C.ac_final(r.X, fs))) if r.ok.any() else np.nan))
    return rows


if __name__ == "__main__":
    full = "--small" not in sys.argv
    th = pd.DataFrame(theory_grid())
    C.write_csv(RES / "t05_theory_grid.csv", th)
    print(th[["eps", "s_crit", "rho_small", "slope_rho_over_s2", "pred_slope", "par_abs_small_s", "rho_max_0_4"]].round(5).to_string())
    seeds = C.DEV_SEEDS if full else C.DEV_SEEDS[:1]
    specs = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, specs, procs=2)
    df = pd.DataFrame([r for rs in out for r in rs])
    C.write_csv(RES / "t05_time_roots.csv", df)
    print("exact fail-step matches:", df.fail_step_exact_match.mean(), "max diff", df.max_fail_step_diff.max(), "rows", len(df))
