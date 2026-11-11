"""T3.09 mechanism test: which edges do clipping / down-weighting forces flag, vs the true bridge edges?"""
import i3
from i3 import C, np
import pandas as pd
wt = C.wt
EPS = 1e-12
REC = []     # per-call (t, per-edge flag (R,m), extra dict)
_G = {}

def _edge_slots(g):
    if id(g) in _G: return _G[id(g)]
    n, m = g.n, len(g.ei)
    s_ij = np.zeros(m, int); s_ji = np.zeros(m, int)
    for e in range(m):
        i, j = g.ei[e], g.ej[e]
        s_ij[e] = np.where((g.NB[i] == j) & g.valid[i])[0][0]
        s_ji[e] = np.where((g.NB[j] == i) & g.valid[j])[0][0]
    _G[id(g)] = (s_ij, s_ji); return _G[id(g)]

def _to_edges(g, D):
    s_ij, s_ji = _edge_slots(g)
    return 0.5 * (D[:, g.ei, s_ij] + D[:, g.ej, s_ji])

def _node_to_edges(g, a):
    return 0.5 * (a[:, g.ei] + a[:, g.ej])

def _wrap(name):
    orig = wt.FORCES[name][0]
    defaults = dict(wt.FORCES[name][1])
    @wt.force("i3inst_" + name, **defaults)
    def fn(ctx, u):
        g = ctx.g; pf = ctx.pf
        x = wt.diffs(g, u); ax = np.abs(x) + EPS
        extra = {}
        if name in ("huber", "huber_anneal", "huber_local", "huber_global"):
            if name == "huber": tau = pf["tau"]
            elif name == "huber_anneal": tau = pf["tau0"] * np.exp(-ctx.t / pf["Ta"]) + pf["tau_inf"]
            elif name == "huber_local": tau = pf["kappa"] * wt.nb_median(g, np.abs(x))[..., None] + EPS
            else: tau = pf["kappa"] * wt.rms_rows(wt.edge_values(g, x))[..., None] + EPS
            down = np.where(g.valid, 1 - np.minimum(1.0, tau / ax), 0.0)
            extra["tau"] = float(np.median(np.broadcast_to(tau, x[..., :1].shape)))
            E = _to_edges(g, down)
        elif name in ("amp", "amp_gated"):
            if name == "amp": tau = pf["tau"]
            else: tau = pf["kappa"] * np.median(np.abs(wt.edge_values(g, x)), axis=1)[:, None, None] / 0.6745 + EPS
            act = np.where(g.valid, 1 - tau * np.tanh(x / tau) / x.clip(1e-300) * np.sign(x) if False else 1 - tau * np.tanh(ax / tau) / ax, 0.0)
            extra["tau"] = float(np.median(np.broadcast_to(tau, x[..., :1].shape)))
            E = _to_edges(g, act)
        elif name == "gauss":
            down = np.where(g.valid, 1 - np.exp(-(x / pf["sigma"]) ** 2), 0.0)
            E = _to_edges(g, down)
        elif name == "edge_memory":
            w = ctx.state.get("edge_w", np.ones_like(x))
            E = 1 - _to_edges(g, w)
        elif name == "blend_agree":
            m = wt.edge_mean(g, x); s = np.sqrt(wt.edge_mean(g, (x - m[..., None]) ** 2))
            sbar = np.median(s, axis=1, keepdims=True) + EPS
            a = 1 - np.exp(-(s / sbar) ** 2)          # weight on the clipped branch
            E = _node_to_edges(g, a)
        REC.append((ctx.t, E, extra))
        return orig(ctx, u)
    return fn

NAMES = ["huber", "huber_local", "huber_global", "huber_anneal", "amp", "amp_gated", "gauss", "edge_memory", "blend_agree"]
for _n in NAMES: _wrap(_n)

DYNS = [("damp", 0.5), ("wave", 0.5), ("diff", None)]
WINDOWS = [(0, 20), (20, 100), (100, 300), (300, 500)]
STEPS = 500

def job(args):
    spec, = args
    g, truth, meta = C.make_graph(spec)
    mask = meta["bridge_mask"]; nb = int(mask.sum())
    dg = C.wdeg(g); deg_score = dg[g.ei] + dg[g.ej]
    u0 = C.init("uniform", g, 8, np.random.default_rng(spec["seed"] * 1000 + 309))
    rows = []
    for name in NAMES:
        for dyn, c in DYNS:
            REC.clear()
            f = "i3inst_" + name
            if dyn == "diff": run = C.run_diff(g, f, u0, 0.5, STEPS)
            else: run = C.run(g, dyn, f, u0, c, STEPS, ps=dict(k=0.1) if dyn == "damp" else None)
            ok = run.ok
            if not ok.any(): continue
            T = len(REC)
            Es = np.stack([r[1] for r in REC])                        # (T,R,m)
            taus = np.array([r[2].get("tau", np.nan) for r in REC])
            for (a, b) in WINDOWS:
                S = Es[a:b][:, ok].mean((0, 1))                         # per-edge flag score
                X = np.abs(run.X[a:b + 1][:, ok][:, :, g.ei] - run.X[a:b + 1][:, ok][:, :, g.ej])
                xs = np.sqrt((X ** 2).mean((0, 1)))                     # plain edge-difference score in the window
                top = np.argsort(-S)[:nb]
                row = dict(graph=meta["id"], gkind=spec["block"], bridges=nb, seed=spec["seed"], force=name, dyn=dyn, win=f"{a}-{b}",
                           n_ok=int(ok.sum()), auc=C.bridge_auc(S, mask), prec_at_nb=float(mask[top].mean()),
                           auc_absx=C.bridge_auc(xs, mask), auc_deg=C.bridge_auc(deg_score, mask),
                           flag_bridge=float(S[mask].mean()), flag_intra=float(S[~mask].mean()),
                           flag_intra_max=float(S[~mask].max()), tau=float(np.nanmedian(taus[a:b])) if np.isfinite(taus[a:b]).any() else np.nan,
                           rms_end=float(np.sqrt((run.X[min(b, T)][ok] ** 2).mean())))
                # does S merely track |x|?  Spearman of S with window edge-difference score
                from scipy.stats import spearmanr
                row["rho_flag_absx"] = float(spearmanr(S, xs)[0]) if S.std() > 0 and xs.std() > 0 else np.nan
                rows.append(row)
    return rows

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = (0,) if quick else C.DEV_SEEDS[:4]
    specs = C.zoo("core", seeds=seeds)
    if quick: specs = specs[:2] + specs[6:7]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t309_mechanism_quick.csv" if quick else "t309_mechanism.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    print(df.groupby(["force", "dyn", "win"])[["auc", "prec_at_nb", "auc_absx", "auc_deg", "flag_bridge", "flag_intra", "tau"]].mean().round(2).to_string())
