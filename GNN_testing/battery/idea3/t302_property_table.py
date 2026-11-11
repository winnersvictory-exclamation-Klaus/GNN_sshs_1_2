"""T3.02 property table: force_props for every force x 4 graph kinds x 3 states x 3 seeds (dev)."""
import i3
from i3 import C, np
import pandas as pd

FORCES = i3.FORCES19 + ["mean", "owa_mc", "vw_mc", "trim_mc"]
KINDS = [dict(kind="cycle", bridges=2), dict(kind="clique", bridges=2), dict(kind="sbm", bridges=2),
         dict(kind="barbell_path")]
SEEDS = (0, 1, 2)

def states(g, seed):
    rng = np.random.default_rng(seed * 7 + 1)
    u_rand = C.init("uniform", g, 1, rng)[0]
    # mid-trajectory: damped wave+mean state at step 100 from a fresh uniform start (generic smooth-ish state)
    u0 = C.init("uniform", g, 1, rng)
    r = C.run(g, "damp", "mean", u0, 0.5, 100, ps=dict(k=0.1))
    u_mid = r.X[-1, 0]
    u_cons = 0.7 + 1e-3 * rng.standard_normal(g.n)     # near-consensus (small amplitude differences)
    return dict(random=u_rand, mid=u_mid, near_consensus=u_cons)

def job(args):
    fk, seed = args
    spec = dict(fk, seed=seed)
    g, truth, meta = C.make_graph(spec)
    rows = []
    for st, u in states(g, seed).items():
        for f in FORCES:
            try:
                p = C.force_props(g, f, u=u)
            except Exception as e:
                p = dict(error=str(e))
            p.update(graph=meta["id"], gkind=fk["kind"], seed=seed, state=st, force=f)
            rows.append(p)
    return rows

if __name__ == "__main__":
    out = C.pmap(job, [(k, s) for k in KINDS for s in SEEDS], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / "t302_property_table.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["oddness", "shift_rel", "homog_rel", "momentum_rel", "dsym_rel", "lam_max", "eig_sym_max", "jac_kink"]
    print(df.groupby(["state", "force"])[cols].median().round(4).to_string())
