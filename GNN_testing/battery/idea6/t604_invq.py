"""T6.04 non-monotone invq: late-state bimodality, sign-split alignment, negative-control graphs."""
import i6lib as L
from i6lib import C, np
from sklearn.metrics import adjusted_rand_score
from scipy import stats as sst

CFG = [("mean", None), ("tanh_g", None), ("invq_g", 0.3), ("invq_g", 1.0), ("invq_g", 4.0), ("invq_g", 10.0),
       ("invq_l", 0.3), ("invq_l", 1.0), ("invq_l", 4.0), ("invq_l", 10.0)]
DYN = [("diff", 0.25), ("diff", 0.5), ("wave", 0.25), ("wave", 0.5)]


def bc(x):
    n = len(x)
    sk = sst.skew(x); ku = sst.kurtosis(x)
    return (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def cond_of(g, lab):
    a = lab == 0
    if a.all() or (~a).all():
        return np.nan
    A = g.A
    cut = A[np.ix_(a, ~a)].sum()
    d = A.sum(1)
    return cut / min(d[a].sum(), d[~a].sum())


def job(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 8, rng)
    lam, Phi, d = C.spectrum(g)
    fied = Phi[:, 1] > np.median(Phi[:, 1])
    fied_cond = cond_of(g, fied.astype(int))
    rows = []
    for (tr, gam) in CFG:
        f = "mean" if tr == "mean" else L.fname(tr)
        pf = None if gam is None else {"gamma": gam}
        for dyn, par in DYN:
            run = C.run_diff(g, f, u0, eta=par, steps=500, pf=pf) if dyn == "diff" else C.run(g, "wave", f, u0, c=par, steps=500, pf=pf)
            ok = run.ok
            row = L.base_row(meta, block=spec.get("block", spec["kind"]), neg=truth is None, transform=tr, gamma=gam if gam else np.nan,
                             dyn=dyn, par=par, fail_frac=float(1 - ok.mean()))
            if ok.any():
                X = run.X[:, ok]
                late = X[-50:].mean(0) if dyn == "wave" else X[-1]       # time-avg for oscillating waves
                late = late - late.mean(1, keepdims=True)
                sd = late.std(1)
                alive = sd > 1e-9
                row["alive_frac"] = float(alive.mean())
                if alive.any():
                    bcs, aris, conds, mf, bim = [], [], [], [], []
                    for r in np.flatnonzero(alive):
                        x = late[r] / sd[r]
                        b = bc(x); bcs.append(b)
                        lab = (x > 0).astype(int)
                        mf.append(min(lab.mean(), 1 - lab.mean()))
                        conds.append(cond_of(g, lab))
                        if truth is not None:
                            aris.append(adjusted_rand_score(truth, lab))
                    row.update(bc=float(np.median(bcs)), bimodal_frac=float(np.mean(np.array(bcs) > 0.555)),
                               ari_sign=float(np.median(aris)) if aris else np.nan, cond_sign=float(np.nanmedian(conds)),
                               minfrac_sign=float(np.median(mf)), fied_cond=fied_cond,
                               kurt=float(np.median([sst.kurtosis(late[r] / sd[r]) for r in np.flatnonzero(alive)])))
            rows.append(row)
    return rows


if __name__ == "__main__":
    specs = L.core_specs(range(4)) + C.zoo("neg", seeds=range(4))
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(L.RES / "t604_invq.csv", rows)
    print(len(rows), "rows")
