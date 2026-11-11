"""T4.03 order/weight sweep (mean base).  Part A: geometric series a_k=eps^{k-1}, K=1..6, eps in linspace(-0.5,0.5,21) via generalised
`series:mean` force; theory c_pred (max g) and simulation fail_frac at c in {0.5,1,1.4,1.8} (T=1000, R=4).
Part B: 150 random weight vectors (K=4, a_1=1, a_2..4 ~ U(-1,1)) -> theory (min g, predicted modal growth rate) vs simulation at c in {0.5,1.0}.
Graphs: cycle/clique/sbm blocks 24+24 bridges 2 (seed 0)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import i4
import numpy as np, pandas as pd
C = i4.C

T = 1000
EPS = np.round(np.linspace(-0.5, 0.5, 21), 3)
CS = (0.5, 1.0, 1.4, 1.8)


def job(spec):
    g, truth, meta = C.make_graph(spec)
    lam, Phi, d = C.spectrum(g)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, 4, rng)
    rows = []
    for K in range(1, 7):
        for e in EPS:
            w = tuple(float(e) ** k for k in range(K))
            gv = i4.g_of(lam[1:], w)
            cp = i4.c_crit_pred(lam, w)
            row = dict(part="A", graph=meta["id"], K=K, eps=e, g_min=gv.min(), g_max=gv.max(), c_pred=cp, n_neg=int((gv < -1e-12).sum()))
            for c in CS:
                rate = float(np.log(i4.modal_rho_wave(c * c * gv)).max())
                row[f"rate_c{c}"] = rate
                row[f"ff_c{c}"] = C.fail_frac(g, "wave", "series:mean", u0, c, steps=T, pf=dict(w=w))
            rows.append(row)
    rs = np.random.default_rng(12345)
    for i in range(150):
        w = (1.0,) + tuple(rs.uniform(-1, 1, 3))
        gv = i4.g_of(lam[1:], w)
        row = dict(part="B", graph=meta["id"], K=4, vec=i, w2=w[1], w3=w[2], w4=w[3], g_min=gv.min(), g_max=gv.max(),
                   n_neg=int((gv < -1e-12).sum()), c_pred=i4.c_crit_pred(lam, w))
        for c in (0.5, 1.0):
            row[f"rate_c{c}"] = float(np.log(i4.modal_rho_wave(c * c * gv)).max())
            row[f"ff_c{c}"] = C.fail_frac(g, "wave", "series:mean", u0, c, steps=T, pf=dict(w=w))
        rows.append(row)
    return rows


if __name__ == "__main__":
    specs = [dict(kind="two_block", block=b, sizes=(24, 24), bridges=2, seed=0) for b in ("cycle", "clique", "sbm")]
    out = C.pmap(job, specs, procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i4.RES / "t403_order_weight_sweep.csv", rows)
    df = pd.DataFrame(rows)
    for c in CS:
        a = df[df.part == "A"]
        pred_fail = a[f"rate_c{c}"] * T > 8
        pred_ok = a[f"rate_c{c}"] <= 1e-9
        obs_fail = a[f"ff_c{c}"] > 0
        print("A c=%.1f  pred_fail&obs_fail=%d pred_fail&!obs=%d pred_ok&obs_fail=%d pred_ok&!obs=%d ambiguous=%d" % (
            c, (pred_fail & obs_fail).sum(), (pred_fail & ~obs_fail).sum(), (pred_ok & obs_fail).sum(), (pred_ok & ~obs_fail).sum(),
            (~pred_fail & ~pred_ok).sum()))
    for c in (0.5, 1.0):
        b = df[df.part == "B"]
        pred_fail = b[f"rate_c{c}"] * T > 8
        pred_ok = b[f"rate_c{c}"] <= 1e-9
        obs_fail = b[f"ff_c{c}"] > 0
        print("B c=%.1f  pred_fail&obs_fail=%d pred_fail&!obs=%d pred_ok&obs_fail=%d pred_ok&!obs=%d ambiguous=%d  n_neg_g>0 vectors=%d" % (
            c, (pred_fail & obs_fail).sum(), (pred_fail & ~obs_fail).sum(), (pred_ok & obs_fail).sum(), (pred_ok & ~obs_fail).sum(),
            (~pred_fail & ~pred_ok).sum(), (b.n_neg > 0).sum()))
