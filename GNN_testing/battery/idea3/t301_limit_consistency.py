"""T3.01 limit consistency: each force's limiting parameter must reproduce mean to 1e-9."""
import i3
from i3 import C, np
import pandas as pd

LIMITS = [  # force, params, (note)
    ("huber", dict(tau=1e9)), ("tanh", dict(tau=1e5)), ("tanh", dict(tau=1e6)), ("tanh", dict(tau=1e7)), ("tanh", dict(tau=1e8)), ("amp", dict(beta=0.0)),
    ("amp", dict(beta=0.0, tau=5.0)), ("gauss", dict(sigma=1e9)), ("absvw", dict(beta=0.0)),
    ("symowa_ext", dict(gamma=0.0)), ("symowa_med", dict(gamma=0.0)), ("edge_memory", dict(rho=0.0)),
    ("huber_local", dict(kappa=1e9)), ("huber_global", dict(kappa=1e9)),
    ("amp_gated", dict(kappa=1e9, beta=0.0)), ("amp_gated", dict(kappa=1e9, beta=1.0)),
    ("blend_agree", dict(tau=1e9)), ("huber_anneal", dict(tau0=1e9, tau_inf=1e9)),
    ("trim", {}), ("owa", {}), ("vw", {}), ("median", {}), ("odd_owa", {}), ("odd_vw", {}),
]
SPECS = ([dict(kind="cycle", bridges=b, seed=s) for b in (1, 2) for s in (0, 1)] +
         [dict(kind="clique", bridges=2, seed=0), dict(kind="sbm", bridges=2, seed=0),
          dict(kind="two_block", block="sbm", bridges=2, wb=0.25, seed=0),
          dict(kind="two_block", block="cycle", sizes=(16, 32), bridges=4, wb=0.25, seed=1),
          dict(kind="random_regular", n=48, d=3, seed=0)])

def run():
    rows = []
    for spec in SPECS:
        g, truth, meta = C.make_graph(spec)
        rng = np.random.default_rng(1)
        U = C.init("uniform", g, 6, rng); U[3] = rng.standard_normal(g.n) * 5   # large-amplitude row
        Fm = C.eval_force(g, "mean", U)
        for f, pf in LIMITS:
            F = C.eval_force(g, f, U, pf=pf, t=0)
            # edge_memory stateful: fresh ctx = initial state; also check after a few steps below
            rows.append(dict(graph=meta["id"], force=f, params=str(pf), maxerr=float(np.abs(F - Fm).max()),
                             relerr=float(np.abs(F - Fm).max() / np.abs(Fm).max())))
    # edge_memory over time: rho=0 along a trajectory equals mean wave trajectory
    for spec in SPECS[:3]:
        g, truth, meta = C.make_graph(spec)
        u0 = C.init("uniform", g, 4, np.random.default_rng(2))
        a = C.run(g, "wave", "mean", u0, 0.7, 300); b = C.run(g, "wave", "edge_memory", u0, 0.7, 300, pf=dict(rho=0.0))
        rows.append(dict(graph=meta["id"], force="edge_memory(traj rho=0, T=300)", params="rho=0", maxerr=float(np.abs(a.X - b.X).max()), relerr=np.nan))
        a = C.run(g, "wave", "mean", u0, 0.7, 300); b = C.run(g, "wave", "huber_anneal", u0, 0.7, 300, pf=dict(tau0=1e9, tau_inf=1e9))
        rows.append(dict(graph=meta["id"], force="huber_anneal(traj big tau, T=300)", params="", maxerr=float(np.abs(a.X - b.X).max()), relerr=np.nan))
    # trim on degree-2 nodes == mean : evaluate on deg<=2 nodes of every graph
    for spec in SPECS:
        g, truth, meta = C.make_graph(spec)
        U = C.init("uniform", g, 6, np.random.default_rng(3))
        d = (g.deg <= 2)
        if d.sum() == 0:
            continue
        err = np.abs(C.eval_force(g, "trim", U) - C.eval_force(g, "mean", U))[:, d].max()
        errhi = np.abs(C.eval_force(g, "trim", U) - C.eval_force(g, "mean", U))[:, ~d].max() if (~d).any() else np.nan
        rows.append(dict(graph=meta["id"], force="trim@deg<=2 nodes", params="", maxerr=float(err), relerr=np.nan))
        rows.append(dict(graph=meta["id"], force="trim@deg>=3 nodes", params="", maxerr=float(errhi), relerr=np.nan))
    return rows

if __name__ == "__main__":
    rows = run()
    C.write_csv(i3.RES / "t301_limit_consistency.csv", rows)
    df = pd.DataFrame(rows)
    s = df.groupby(["force", "params"]).maxerr.max()
    pd.set_option("display.width", 200)
    print(s.to_string())
