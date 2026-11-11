"""T5.02 dispersion measurement (space, mean base). Modal coordinates a_k(t) of the simulated wave+d4:mean
trajectory; measured frequency (a) exact three-term estimator cos w = sum a_t(a_{t+1}+a_{t-1}) / (2 sum a_t^2),
(b) FFT peak with parabolic interpolation, vs predicted cos w = 1 - c^2 g(lam)/2.
Output results/t02_dispersion.csv: one row per graph x eps x c (summary over modes) and
results/t02_dispersion_modes.csv: per mode for seed 0 graphs (for plotting)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib5 import *
import pandas as pd

EPS = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.3, 1.0]
CS = [0.5, 1.0]
BIG = 1e200


def est_omega(a):
    """a: (T+1, nm) -> cos w estimate per mode (exact for pure sinusoid)."""
    num = (a[1:-1] * (a[2:] + a[:-2])).sum(0)
    den = 2 * (a[1:-1] ** 2).sum(0)
    return num / np.maximum(den, 1e-300)


def fft_omega(a):
    T1 = a.shape[0]
    w = np.hanning(T1)[:, None]
    P = np.abs(np.fft.rfft(a * w, n=4 * T1, axis=0)) ** 2
    k = P.argmax(0)
    out = np.empty(a.shape[1])
    N = 4 * T1
    for m in range(a.shape[1]):
        kk = k[m]
        if 0 < kk < P.shape[0] - 1:
            y0, y1, y2 = np.log(P[kk - 1, m] + 1e-300), np.log(P[kk, m] + 1e-300), np.log(P[kk + 1, m] + 1e-300)
            den = y0 - 2 * y1 + y2
            dlt = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        else:
            dlt = 0.0
        out[m] = 2 * np.pi * (kk + dlt) / N
    return out


def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, R, rng)
    lam, Phi, d = C.spectrum(g)
    rows, mrows = [], []
    for e in EPS:
        gv = spatial_g(lam, e)
        for c in CS:
            s = c * c * gv
            stable = bool((s[1:] > 1e-9).all() and (s[1:] < 4 - 1e-9).all())
            T = 1024 if stable else 120
            r = C.run(g, "wave", "d4:mean", u0, c, T, pf=dict(eps=e), fail_amp=BIG)
            a = np.einsum("trn,nk->trk", r.X * d, Phi)          # (T+1,R,nm) modal coordinates
            # use the realisation-pooled estimator (sum over R)
            num = (a[1:-1] * (a[2:] + a[:-2])).sum((0, 1)); den = 2 * (a[1:-1] ** 2).sum((0, 1))
            cosw = num / np.maximum(den, 1e-300)
            cos_pred = 1 - s / 2
            ok = (np.abs(cos_pred) <= 1) & (np.arange(len(lam)) > 0)
            amp = np.sqrt((a ** 2).mean((0, 1)))
            w_est = np.arccos(np.clip(cosw, -1, 1)); w_pred = np.arccos(np.clip(cos_pred, -1, 1))
            if stable:
                afft = a.mean(1) if False else a[:, 0, :]
                w_fft = fft_omega(a[:, 0, :])
                err_fft = np.abs(w_fft - w_pred)[1:]
                err_est = np.abs(w_est - w_pred)[1:]
                rel_est = err_est / np.maximum(w_pred[1:], 1e-9)
                rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], eps=e, c=c, stable=True,
                                 max_abs_err_exact=float(err_est.max()), max_rel_err_exact=float(rel_est.max()),
                                 max_abs_err_fft=float(err_fft.max()), median_abs_err_fft=float(np.median(err_fft)),
                                 w_pred_min=float(w_pred[1:].min()), w_pred_max=float(w_pred[1:].max()),
                                 monotone=bool(np.all(np.diff(w_pred) >= -1e-12)),
                                 frac_modes_w_below_wave=float(np.mean(w_pred[1:] < np.arccos(np.clip(1 - c * c * lam[1:] / 2, -1, 1)) - 1e-12))))
            else:
                # unstable: compare measured growth of each mode amplitude with ln rho prediction
                lnr_pred = pred_ln_rho(s)
                ma = np.sqrt((a ** 2).sum(1))     # (T+1,nm)
                tt = np.arange(T // 2, T + 1)
                slope = np.array([np.polyfit(tt, np.log(ma[T // 2:, m] + 1e-300), 1)[0] for m in range(len(lam))])
                unstable = lnr_pred > 1e-6
                dev = np.abs(slope[unstable] - lnr_pred[unstable]) if unstable.any() else np.array([np.nan])
                rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], eps=e, c=c, stable=False,
                                 n_unstable_modes=int(unstable.sum()), max_growth_dev=float(np.nanmax(dev)),
                                 max_pred_ln_rho=float(lnr_pred.max())))
            if spec["seed"] == 0:
                for m in range(1, len(lam)):
                    mrows.append(dict(graph=meta["id"], eps=e, c=c, mode=m, lam=lam[m], g=gv[m], w_pred=w_pred[m],
                                      w_meas=w_est[m], stable=stable))
    return rows, mrows


if __name__ == "__main__":
    small = "--small" in sys.argv
    seeds = C.DEV_SEEDS[:1] if small else C.DEV_SEEDS[:4]
    specs = [dict(kind=k, bridges=2, seed=s) for k in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    df = pd.DataFrame([r for o in out for r in o[0]]); dm = pd.DataFrame([r for o in out for r in o[1]])
    C.write_csv(RES / ("t02_dispersion.csv" if not small else "t02_small.csv"), df)
    C.write_csv(RES / ("t02_dispersion_modes.csv" if not small else "t02_small_modes.csv"), dm)
    print(df[df.stable].groupby(["eps", "c"])[["max_abs_err_exact", "max_abs_err_fft", "monotone"]].agg("max").round(6))
    print(df[~df.stable.astype(bool)].groupby(["eps", "c"])[["n_unstable_modes", "max_growth_dev"]].agg("max"))
