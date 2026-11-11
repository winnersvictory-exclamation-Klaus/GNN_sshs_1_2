"""T3.06 addendum: spatial profile of the state at the last valid step before failure (localised vs global mode)."""
import i3
from i3 import C, np
import pandas as pd
from t306_localisation import hop_dist

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    dist, ends = hop_dist(g, meta["bridge_mask"])
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 306))
    rows = []
    for f in ("trim", "median", "owa", "vw", "symowa_ext", "gauss"):
        for c in (0.5,):
            run = C.run(g, "wave", f, u0, c, 500)
            for r in range(run.R):
                if not run.fail[r]: continue
                t = int(run.fail_step[r]) - 1
                u = run.X[t, r]; u2 = u ** 2
                pr = float(u2.sum() ** 2 / ((u2 ** 2).sum() * g.n))
                near2 = dist <= 2
                # first step where AC rms exceeds 3x initial; profile at that time
                ac = run.X[:, r] - run.X[:, r].mean(1, keepdims=True)
                rms = np.sqrt((ac ** 2).mean(1)); t3 = np.where(rms > 3 * rms[0])[0]
                t3 = int(t3[0]) if len(t3) else t
                v = run.X[t3, r]; v2 = v ** 2
                rows.append(dict(graph=meta["id"], gkind=spec["block"], bridges=spec["bridges"], seed=spec["seed"], force=f, c=c, real=r, fail_step=t + 1,
                                 pr_fail=pr, e_near2_fail=float(u2[near2].sum() / u2.sum()), base_near2=float(near2.mean()),
                                 dc_frac_fail=float(abs(u.mean()) / np.sqrt(u2.mean())), t3=t3,
                                 pr_t3=float(v2.sum() ** 2 / ((v2 ** 2).sum() * g.n)), e_near2_t3=float(v2[near2].sum() / v2.sum()),
                                 amp_ends_over_rms_t3=float(np.abs(v[dist == 0]).mean() / np.sqrt(v2.mean()))))
    return rows

if __name__ == "__main__":
    specs = [dict(kind="two_block", block="cycle", bridges=b, sizes=(24, 24), wb=1.0, seed=s) for b in (1, 2) for s in C.DEV_SEEDS]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / "t306b_profile.csv", rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["force"])[["fail_step", "pr_fail", "e_near2_fail", "base_near2", "dc_frac_fail", "t3", "pr_t3", "e_near2_t3", "amp_ends_over_rms_t3"]].mean().round(3).to_string())
