"""T4.02 dispersion, mean base: modal frequencies measured from simulation vs cos(w)=1-c^2 g(lam)/2.
Graphs: core seed 0 (cycle/clique/sbm, bridges 2) + sbm bridges 1; forces: mean, hi_acc, hi_alt, hi_geo(.2), hi_geoneg(-.2);
c in {0.3, 0.9*c_pred}; T=4096, R=2.  Frequency estimators: (a) least-squares recurrence cos(w)=sum(a_{t+1}+a_{t-1})a_t/(2 sum a_t^2),
(b) FFT peak (Hann, zero-padded x16, parabolic interpolation) -> shows resolution limits."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

CONFIGS = [("mean", "mean", {}, (1.0,)), ("hi_acc", "hi_acc:mean", {}, i4.weights("hi_acc")),
           ("hi_alt", "hi_alt:mean", {}, i4.weights("hi_alt")),
           ("hi_geo.2", "hi_geo:mean", {}, i4.weights("hi_geo", 0.2)),
           ("hi_geoneg-.2", "hi_geoneg:mean", {}, i4.weights("hi_geoneg", -0.2))]
T = 4096


def fft_peak(a):
    n = len(a)
    win = np.hanning(n)
    Nf = 16 * n
    sp = np.abs(np.fft.rfft((a - a.mean()) * win, Nf))
    k = int(np.argmax(sp[1:-1])) + 1
    y0, y1, y2 = np.log(sp[k - 1] + 1e-300), np.log(sp[k] + 1e-300), np.log(sp[k + 1] + 1e-300)
    den = y0 - 2 * y1 + y2
    dk = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return 2 * np.pi * (k + dk) / Nf


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 2, rng)
    rows = []
    for label, fname, pf, w in CONFIGS:
        cp = i4.c_crit_pred(lam, w)
        for cname, c in (("c=0.3", 0.3), ("c=0.9cp", 0.9 * cp)):
            run = C.run(g, "wave", fname, u0, c=c, steps=T, pf=pf)
            assert run.ok.all()
            A = np.einsum("trn,n,nk->trk", run.X, d, Phi)  # modal coordinates
            for k in range(1, g.n):
                gk = float(i4.g_of(lam[k], w))
                s = c * c * gk
                if not (1e-9 < s < 4 - 1e-9):
                    continue
                om_pred = np.arccos(1 - s / 2)
                a = A[:, 0, k]
                cosw = ((a[2:] + a[:-2]) * a[1:-1]).sum() / (2 * (a[1:-1] ** 2).sum())
                om_ls = np.arccos(np.clip(cosw, -1, 1))
                om_fft = fft_peak(a)
                rows.append(dict(graph=meta["id"], seed=spec["seed"], config=label, cmode=cname, c=c, mode=k, lam=lam[k], g=gk,
                                 s=s, om_pred=om_pred, om_ls=om_ls, om_fft=om_fft, amp=np.abs(a).max(),
                                 om_plain=c * np.sqrt(lam[k]), om_exact_wave=2 * np.arcsin(c * np.sqrt(lam[k]) / 2) if c * np.sqrt(lam[k]) < 2 else np.nan))
    return rows


if __name__ == "__main__":
    specs = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=br, seed=0) for b, br in (("cycle", 2), ("clique", 2), ("sbm", 2), ("sbm", 1))]
    out = [job(s) for s in specs]
    rows = [r for rs in out for r in rs]
    path = C.write_csv(i4.RES / "t402_dispersion.csv", rows)
    df = pd.DataFrame(rows)
    df["rel_ls"] = (df.om_ls - df.om_pred).abs() / df.om_pred
    df["rel_fft"] = (df.om_fft - df.om_pred).abs() / df.om_pred
    df["rel_plain"] = (df.om_plain - df.om_pred).abs() / df.om_pred
    print(df.groupby(["config", "cmode"], sort=False).agg(n=("mode", "count"), ls_max=("rel_ls", "max"), fft_med=("rel_fft", "median"),
          fft_max=("rel_fft", "max"), plain_dev_med=("rel_plain", "median"), plain_dev_max=("rel_plain", "max")).to_string(float_format="%.2e"))
    # fft error vs period resolution: omega*T/(2pi) cycles in window
    df["cycles"] = df.om_pred * T / (2 * np.pi)
    print(df[df.cycles >= 5].groupby(["config", "cmode"], sort=False).rel_fft.max().to_string(float_format="%.2e"))
