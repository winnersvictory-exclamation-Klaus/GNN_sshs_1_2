"""T4.09 matched space-time orders (mean base).  Modified-equation family: time series e4=-theta/12 (plus e6 to cancel the next order) and spatial weight
a2=-(1-theta) c^2/12 (a3, a4 chosen to cancel the next orders: space-only is a_k=(-1)^{k+1} 2 c^{2(k-1)}/(2k)!*... i.e. g(l)=(2-2cos(c sqrt l))/c^2),
theta in {0,.25,.5,.75,1}.  Dispersion error vs omega=c sqrt(lam): (i) theory from the roots of the characteristic polynomial with s=c^2 g(lam),
(ii) simulation (modal least-squares ARMA(2)).  Reference: plain leapfrog (2 asin(c sqrt(l)/2)).
Note on the code: the delta^4/delta^6 terms are NOT centred at the same time level as delta^2 (see T4.07), so time-side cancellation is imperfect:
we measure both phase error and amplitude error (log|z| per step)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
from math import factorial
C = i4.C

THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CS = (0.3, 0.6, 1.0, 1.3)


def space_w(theta, c):
    """space weights: full space part removes fraction (1-theta) of the leapfrog truncation: 2-2cos(x) = sum 2(-1)^{j+1} x^{2j}/(2j)!
    leapfrog numerical omega: 2 sin(w/2)=x -> s=x^2 ; we want c^2 g = 2-2cos(x*) with x*^2 = c^2 lam ... use series of the exact cosine:
    g(l) = [ (2-2cos(c sqrt l)) - theta*(c^2 l - (c^2 l)^2/12... ] simplified: a_j = (1-theta)... see body."""
    # exact: c^2 g_exact(l) = 2-2cos(c sqrt l) = sum_j 2 (-1)^{j+1} (c^2 l)^j/(2j)!   -> weights a_j(exact)=2(-1)^{j+1} c^{2(j-1)}/(2j)!
    # leapfrog alone (theta=0 for space): a=(1,0,0,0).  time corrections provide theta of the cancellation.
    w = [1.0]
    for j in range(2, 5):
        w.append((1 - theta) * 2 * (-1) ** (j + 1) * c ** (2 * (j - 1)) / factorial(2 * j))
    return tuple(w)


def time_eps(theta):
    return (-theta / 12.0, theta * (1 / 90.0))   # theta=1: centred-ideal values (-1/12, 1/90)


def job_theory(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rows = []
    for th in THETAS:
        for c in CS:
            w = space_w(th, c)
            eps = time_eps(th)
            for k in range(1, g.n):
                s = c * c * float(i4.g_of(lam[k], w))
                if not (1e-9 < s):
                    continue
                om_true = c * np.sqrt(lam[k])
                if om_true >= 3.0:
                    continue
                zr = i4.time_roots(eps, s) if th > 0 else i4.time_roots((0.0, 0.0), s)
                zr = zr[np.argmin(abs(zr - np.exp(1j * om_true)))]
                rho_all = np.abs(i4.time_roots(eps, s)).max()
                rows.append(dict(graph=meta["id"], theta=th, c=c, mode=k, lam=lam[k], om_true=om_true, om_num=abs(np.angle(zr)),
                                 amp_logz=np.log(abs(zr)), rho_all=rho_all,
                                 om_leapfrog=2 * np.arcsin(min(om_true / 2 * 1.0, 1.0)) if True else np.nan))
    return rows


def arma2(a):
    y = a[2:]
    Xm = np.stack([a[1:-1], -a[:-2]], 1)
    (p, q), *_ = np.linalg.lstsq(Xm, y, rcond=None)
    z = np.roots([1.0, -p, q])
    return z[np.argmax(z.imag)] if np.iscomplexobj(z) and abs(z.imag).max() > 0 else z[0]


def job_sim(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    u0 = C.init("uniform", g, 2, np.random.default_rng(3))
    rows = []
    for th in THETAS:
        for c in (0.6, 1.0):
            w = space_w(th, c)
            eps = time_eps(th)
            run = C.run(g, "t6g", "series:mean", u0, c=c, steps=400, ps=dict(eps=eps), pf=dict(w=w), fail_amp=1e12)
            A = np.einsum("trn,n,nk->trk", run.X, d, Phi)
            for k in range(1, g.n):
                s = c * c * float(i4.g_of(lam[k], w))
                if s < 1e-3 or c * np.sqrt(lam[k]) >= 3.0:
                    continue
                a = A[30:200, 0, k]
                if np.abs(a).max() < 1e-9:
                    continue
                z = arma2(a)
                zth = i4.time_phys_root(eps if th > 0 else (0.0, 0.0), s)
                rows.append(dict(graph=meta["id"], theta=th, c=c, mode=k, om_sim=abs(np.angle(z)), om_th=abs(np.angle(zth)), logz_sim=np.log(abs(z)),
                                 logz_th=np.log(abs(zth))))
    return rows


if __name__ == "__main__":
    specs = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=2, seed=0) for b in ("cycle", "clique", "sbm")]
    rows = [r for s in specs for r in job_theory(s)]
    C.write_csv(i4.RES / "t409_theory.csv", rows)
    df = pd.DataFrame(rows)
    df["phase_err"] = (df.om_num - df.om_true) / df.om_true
    df["leap_err"] = (2 * np.arcsin(df.om_true / 2) - df.om_true) / df.om_true   # plain leapfrog reference (c sqrt(lam) <3)
    pd.set_option("display.width", 250)
    print(df.groupby(["theta", "c"]).agg(max_abs_phase_err=("phase_err", lambda x: x.abs().max()), med_abs_phase_err=("phase_err", lambda x: x.abs().median()),
          min_logz=("amp_logz", "min"), max_logz=("amp_logz", "max"), rho_all=("rho_all", "max"), leap_max=("leap_err", lambda x: x.abs().max())).round(5).to_string())
    sim = [r for s in specs for r in job_sim(s)]
    C.write_csv(i4.RES / "t409_sim.csv", sim)
    sd = pd.DataFrame(sim)
    sd["d_om"] = (sd.om_sim - sd.om_th).abs()
    sd["d_logz"] = (sd.logz_sim - sd.logz_th).abs()
    print(sd.groupby(["theta", "c"]).agg(n=("mode", "count"), max_dom=("d_om", "max"), max_dlogz=("d_logz", "max")).to_string(float_format="%.2e"))
