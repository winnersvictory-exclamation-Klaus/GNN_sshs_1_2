"""T2.05 cfl estimator quality.  lam_hat_t = rms(F-Fp)/rms(v) (what the scheme uses) vs true stiffness lam_true(t) =
lam_max(-sym(DJ)) from the numerical Jacobian of the force at the trajectory state (C.force_props; mean: exact spectrum).
scheme cfl (s=0.9) at c in {1.0,1.4,1.8}; forces mean, huber, trim, owa; cycle/clique/sbm bridges 2 seeds 0-1; R=4, T=300,
Jacobian sampled every 15 steps for all R."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 300
EVERY = 15

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    lam_all, _, d = C.spectrum(g)
    lam_exact = float(lam_all[-1])
    rows, samples = [], []
    for f in ("mean", "huber", "trim", "owa"):
        for c in (1.0, 1.4, 1.8):
            for sch, ps in (("cfl", {"s": 0.9}), ("cfl_ac", {"s": 0.9})):
                hat = np.full((T, R), np.nan); c2t = np.full((T, R), np.nan)
                snaps = {}
                def obs(t, ctx, cur, prev, F, Fp, nxt, alive):
                    v = cur - prev
                    if sch == "cfl":
                        nv = L.rms_rows(v)[:, 0]; lam = L.rms_rows(F - Fp)[:, 0] / np.maximum(nv, L.EPS)
                    else:
                        nv = L.ac_rms_rows(v)[:, 0]; lam = L.ac_rms_rows(F - Fp)[:, 0] / np.maximum(nv, L.EPS)
                    ct = np.where(nv > 1e-9, np.minimum(c * c, ps["s"] * 4 / np.maximum(lam, L.EPS)), c * c)
                    hat[t] = np.where(alive & (nv > 1e-9), lam, np.nan); c2t[t] = np.where(alive, ct, np.nan)
                    if t % EVERY == 0 and t > 0:
                        snaps[t] = (cur.copy(), alive.copy())
                run, _ = L.instr_run(g, sch, f, u0, c, T, ps, extra=obs)
                wv = C.run(g, "wave", f, u0, c, T)
                row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], force=f, c=c, scheme=sch,
                           fail_frac=float(run.fail.mean()), wave_fail_frac=float(wv.fail.mean()), lam_exact=lam_exact,
                           c_crit_mean=2 / np.sqrt(lam_exact), frac_ct_lt_c=L.nanmean(c2t < c * c * 0.999),
                           ct_ratio_median=L.nanmed(np.sqrt(c2t) / c))
                # true stiffness along trajectory
                lt, lh, cs = [], [], []
                for t, (U, alive) in snaps.items():
                    for r in range(R):
                        if not alive[r] or not np.isfinite(hat[t, r]):
                            continue
                        if np.abs(U[r]).max() > 5:        # skip pre-blow-up states
                            continue
                        pr = C.force_props(g, f, U[r], return_jac=False)
                        lt.append(pr["lam_max"]); lh.append(hat[t, r]); cs.append(c2t[t, r])
                        samples.append(dict(graph=meta["id"], force=f, c=c, scheme=sch, t=t, r=r, lam_true=pr["lam_max"], lam_hat=hat[t, r],
                                            c2t=c2t[t, r], c2=c * c, margin=c2t[t, r] * pr["lam_max"] / 4.0))
                lt, lh, cs = map(np.array, (lt, lh, cs))
                if len(lt):
                    row.update(n_samp=len(lt), bias_median=float(np.median(lh / lt)), bias_p10=float(np.quantile(lh / lt, .1)),
                               bias_p90=float(np.quantile(lh / lt, .9)),
                               corr=float(np.corrcoef(lt, lh)[0, 1]) if len(lt) > 2 and lt.std() > 1e-9 and lh.std() > 1e-9 else np.nan,
                               unsafe_frac=float(np.mean(cs * lt >= 4.0)),            # c_t^2 * lam_true >= 4 : linear-stability violated
                               unsafe_frac_c=float(np.mean(c * c * lt >= 4.0)),        # same with the unprotected c
                               lam_true_med=float(np.median(lt)), lam_hat_med=float(np.median(lh)))
                rows.append(row)
    return rows, samples

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 4) for s in specs], procs=2)
    rows = [r for rs, _ in out for r in rs]
    samples = [s for _, ss in out for s in ss]
    C.write_csv(Path(__file__).parent / "results" / "t05_cfl_estimator.csv", rows)
    C.write_csv(Path(__file__).parent / "results" / "t05_cfl_estimator_samples.csv", samples)
    print("rows", len(rows), len(samples), "time %.0fs" % (time.time() - t0))
