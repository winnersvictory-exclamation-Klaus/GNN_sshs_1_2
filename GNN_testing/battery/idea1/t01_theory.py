"""T1.01 theory check: damp/viscous/damp_visc + mean force vs per-mode characteristic roots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
import numpy as np, pandas as pd

KS = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
NUS = [0.0, 0.05, 0.1, 0.2, 0.4]
CS = [0.25, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
T = 2000

def scheme_for(k, nu):
    if nu == 0: return "damp", dict(k=k)
    if k == 0: return "viscous", dict(nu=nu)
    return "damp_visc", dict(k=k, nu=nu)

def job(args):
    spec, mode = args
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 17))
    rows = []
    if mode == "grid":
        for k in KS:
            for nu in NUS:
                for c in CS:
                    sch, ps = scheme_for(k, nu)
                    rho_modes = np.abs(C.modal_roots(-lam, c, k, nu)).max(-1)
                    rho = rho_modes.max()
                    run = C.run(g, sch, "mean", u0, c, T, ps)
                    s = C.traj_summary(run)
                    # predicted AC decay rate: dominant non-consensus mode
                    row = dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], k=k, nu=nu, c=c,
                               lam_max=lam[-1], rho_pred=rho, log_rho_pred=np.log(rho),
                               log_rho_ac_pred=np.log(rho_modes[1:].max()),
                               fail_frac=s["fail_frac"], mean_fail_step=s["mean_fail_step"],
                               growth_meas=s["growth_rate"], dead_frac=s["dead_frac"], ac_final=s["ac_final"])
                    rows.append(row)
    elif mode == "boundary":
        for k in [0.0, 0.05, 0.1, 0.3]:
            for nu in [0.0, 0.1, 0.4]:
                sch, ps = scheme_for(k, nu)
                cc = C.c_crit_linear(-lam, k, nu)
                for f in (0.97, 1.03):
                    for steps in (2000, 10000):
                        c = cc * f
                        ff = C.fail_frac(g, sch, "mean", u0, c, steps, ps)
                        rho = C.modal_rho(-lam, c, k, nu)
                        rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], k=k, nu=nu,
                                         c_crit_pred=cc, factor=f, c=c, steps=steps, rho_pred=rho, fail_frac=ff))
    else:  # per-mode decay
        for (k, nu, c) in [(0.05, 0, 1.0), (0.1, 0, 0.5), (0, 0.2, 1.0), (0.02, 0.1, 0.8), (0.3, 0, 1.0), (0.0, 0.4, 0.5)]:
            sch, ps = scheme_for(k, nu)
            steps = 300
            run = C.run(g, sch, "mean", u0, c, steps, ps)
            a = C.mode_energy(run.X, g, run.fail_step, full=True)["a"]  # (T+1,R,n)
            W = 20
            nw = (steps + 1) // W
            env = np.sqrt((a[:nw * W] ** 2).reshape(nw, W, a.shape[1], a.shape[2]).mean((1, 2)))  # (nw,n)
            tw = (np.arange(nw) + 0.5) * W
            roots = np.abs(C.modal_roots(-lam, c, k, nu)).max(-1)
            for j in range(1, len(lam)):
                e = env[:, j]
                m = e > 1e-9 * e[0]
                if m.sum() < 4: continue
                # fit second half of usable windows (asymptotic dominant root)
                idx = np.flatnonzero(m); idx = idx[len(idx) // 2:] if len(idx) >= 8 else idx
                if len(idx) < 3: continue
                sl = np.polyfit(tw[idx], np.log(e[idx]), 1)[0]
                rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], k=k, nu=nu, c=c,
                                 mode=j, lam=lam[j], pred=np.log(roots[j]), meas=sl))
    return rows

if __name__ == "__main__":
    specs = [dict(kind=k, sizes=(24, 24), bridges=2, wb=1.0, seed=s) for k in ("cycle", "clique", "sbm") for s in (0, 1)]
    out = {}
    for mode in ("grid", "boundary", "modes"):
        rs = C.pmap(job, [(s, mode) for s in specs], procs=2)
        rows = [r for x in rs for r in x]
        C.write_csv(Path(__file__).parent / "results" / f"t01_theory_{mode}.csv", rows)
        print(mode, len(rows))
