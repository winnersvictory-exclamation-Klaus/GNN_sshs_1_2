"""T3.04 follow-up: robust linear prediction from the true Jacobian spectrum (consensus mode removed), horizon-aware.
Replaces the library's c_crit_eig (rho_tol=1e-7, sensitive to round-off in the zero eigenvalue)."""
import i3
from i3 import C, np
import pandas as pd

STEPS = 2000
THR = np.log(35.0) / STEPS            # growth per step that takes amplitude 0.6 -> FAIL_AMP(20) within T
CGRID = np.round(np.arange(0.01, 3.0, 0.01), 2)

def c_lin(mus, thr=THR):
    """smallest c whose worst per-step log spectral radius exceeds thr (inf if none up to 3)."""
    for c in CGRID:
        w = max(np.log(max(C.modal_rho(m, c), 1e-300)) for m in mus)
        if w > thr: return float(c)
    return float("inf")

def job(args):
    row, = args
    spec = dict(kind="two_block", block=row["gkind"], bridges=2, sizes=(24, 24), wb=1.0, seed=int(row["seed"]))
    g, truth, meta = C.make_graph(spec)
    f = row["force"]
    out = dict(graph=row["graph"], gkind=row["gkind"], seed=row["seed"], force=f, c_crit_emp=row["c_crit_emp"])
    rng = np.random.default_rng(int(row["seed"]) * 1000 + 304)
    u0 = C.init("uniform", g, 8, rng)
    cs = row.get("c_sample", np.nan)
    states = [u0[r] for r in range(4)]
    if np.isfinite(cs):
        run = C.run(g, "wave", f, u0, float(cs), STEPS)
        for t in (10, 50, 200, 500, 1000, 2000):
            for r in range(4):
                if run.fail[r] and t >= run.fail_step[r]: continue
                states.append(run.X[t, r])
    mus, im_ratio, kinks = [], [], []
    for u in states:
        J, kink = C.jacobian(g, f, u, return_kink=True)
        mu = np.linalg.eigvals(J)
        mu = np.delete(mu, np.argmin(np.abs(mu)))             # drop consensus mode
        mus.append(mu); kinks.append(kink)
        im_ratio.append(float(np.abs(mu.imag).max() / max(np.abs(mu).max(), 1e-12)))
    out.update(c_lin_worst=c_lin(mus), c_lin_med=float(np.median([c_lin([m]) for m in mus])), max_re=float(max(m.real.max() for m in mus)),
               max_im_ratio=float(max(im_ratio)), kink=float(max(kinks)), n_states=len(states))
    # sym (D-symmetric) prediction for comparison
    return out

if __name__ == "__main__":
    src = pd.read_csv(i3.RES / "t304_ccrit.csv")
    rows = [job((r,)) for r in src.to_dict("records")]
    C.write_csv(i3.RES / "t304b_linear.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(df.groupby("force")[["c_crit_emp", "c_lin_worst", "c_lin_med", "max_re", "max_im_ratio"]].median().round(3).to_string())
