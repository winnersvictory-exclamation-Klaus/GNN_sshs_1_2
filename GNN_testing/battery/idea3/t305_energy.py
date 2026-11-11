"""T3.05 energy conservation.  Form A: potential energy  E = 1/2 sum d v^2 + c^2 Phi(u),  Phi = sum_edges w rho(x).
Form B (no potential): drift of the quadratic wave energy (C.energy) -- secular drift rate."""
import i3
from i3 import C, np
import pandas as pd

CS = (0.1, 0.25, 0.5)
STEPS = 1000
FORM_A = [("mean", {}), ("huber", {}), ("huber", dict(tau=0.2)), ("tanh", {}), ("amp", {}), ("edge_memory", {}), ("edge_memory", dict(rho=0.005))]
FORM_B = ["gauss", "absvw", "symowa_ext", "symowa_med", "median", "trim", "owa", "vw", "odd_owa", "odd_vw", "huber_local", "huber_global", "amp_gated", "blend_agree", "huber_anneal"]

def rho_fn(f, pf, x):
    if f == "mean": return x ** 2 / 2
    if f == "huber":
        t = pf["tau"]; ax = np.abs(x); return np.where(ax <= t, x ** 2 / 2, t * ax - t ** 2 / 2)
    if f == "tanh":
        t = pf["tau"]; return t ** 2 * np.log(np.cosh(x / t))
    if f == "amp":
        t, b = pf["tau"], pf["beta"]; return (1 + b) * x ** 2 / 2 - b * t ** 2 * np.log(np.cosh(x / t))
    if f == "edge_memory": return x ** 2 / 2     # weights vary in time: potential only 'frozen-weight'
    raise KeyError(f)

def phi(g, X, f, pf):
    d = X[..., g.ej] - X[..., g.ei]
    return (g.ew * rho_fn(f, pf, d)).sum(-1)          # (T+1,R)

def kin(g, X):
    d = C.wdeg(g)
    V = X[1:] - X[:-1]
    return 0.5 * (V ** 2 * d).sum(-1)                  # (T,R)

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 305)
    u0 = C.init("uniform", g, 4, rng)
    rows = []
    for f, over in FORM_A:
        pf = C.resolve_params(C.wt.FORCES, f, over)
        for c in CS:
            run = C.run(g, "wave", f, u0, c, STEPS, pf=over or None)
            Ph = phi(g, run.X, f, pf); K = kin(g, run.X)
            E = K + c ** 2 * 0.5 * (Ph[:-1] + Ph[1:])
            # exact-for-mean staggered quadratic energy as a second measure
            row = dict(graph=meta["id"], gkind=spec["block"], seed=spec["seed"], form="A", force=f, params=str(over), c=c,
                       fail_frac=float(run.fail.mean()))
            ok = run.ok
            if ok.any():
                Eo = E[:, ok]; e0 = Eo[:50].mean(0); e1 = Eo[-50:].mean(0)
                row.update(drift_ratio=float(np.median(e1 / e0)), osc=float(np.median((Eo.max(0) - Eo.min(0)) / e0)),
                           slope_per1000=float(np.median(np.polyfit(np.arange(Eo.shape[0]), Eo / e0, 1)[0] * 1000)) if False else float(np.median([(np.polyfit(np.arange(Eo.shape[0]), Eo[:, k] / e0[k], 1)[0]) * 1000 for k in range(Eo.shape[1])])))
                # fraction of time in nonlinear regime (|x| > tau)
                if f in ("huber", "tanh", "amp"):
                    d = np.abs(run.X[:, :, g.ej] - run.X[:, :, g.ei])
                    row["frac_edges_gt_tau"] = float((d[:, ok] > pf["tau"]).mean())
            rows.append(row)
    for f in FORM_B:
        for c in CS:
            run = C.run(g, "wave", f, u0, c, STEPS)
            row = dict(graph=meta["id"], gkind=spec["block"], seed=spec["seed"], form="B", force=f, params="", c=c, fail_frac=float(run.fail.mean()))
            ratios, rates = [], []
            for r in range(run.R):
                T = run.valid_len[r] if hasattr(run, "valid_len") else STEPS
                T = int(np.atleast_1d(T)[r]) if np.ndim(T) else int(T)
                if T < 30: continue
                Er = C.energy(run.X[:T + 1, r:r + 1], g, c, np.array([-1]))[:, 0]
                if not np.all(np.isfinite(Er)) or Er[0] <= 0: continue
                e0 = Er[:10].mean(); e1 = Er[-10:].mean()
                ratios.append(e1 / e0)
                tt = np.arange(len(Er)); rates.append(np.polyfit(tt, np.log(np.maximum(Er, 1e-300) / e0), 1)[0])
            row.update(n_valid=len(ratios), energy_ratio_end=float(np.median(ratios)) if ratios else np.nan,
                       log_energy_rate=float(np.median(rates)) if rates else np.nan,
                       mean_valid_len=float(np.mean(run.valid_len)))
            rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = (0,) if quick else (0, 1, 2, 3)
    specs = [dict(kind="two_block", block=b, bridges=2, sizes=(24, 24), wb=1.0, seed=s) for b in ("cycle", "clique", "sbm") for s in seeds]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t305_energy_quick.csv" if quick else "t305_energy.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(df[df.form == "A"].groupby(["force", "params", "c"])[["drift_ratio", "osc", "slope_per1000", "fail_frac", "frac_edges_gt_tau"]].median().round(4).to_string())
    print(df[df.form == "B"].groupby(["force", "c"])[["energy_ratio_end", "log_energy_rate", "fail_frac", "mean_valid_len"]].median().round(4).to_string())
