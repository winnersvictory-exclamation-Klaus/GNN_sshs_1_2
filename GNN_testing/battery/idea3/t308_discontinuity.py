"""T3.08 discontinuity sensitivity: A twin divergence, B local Lipschitz ratio, C neighbour-order invariance, D Rademacher (all-ties) start."""
import i3
from i3 import C, np
import pandas as pd

FORCES = ["mean"] + i3.FORCES19 + ["owa_mc"]
EPS0 = 1e-10
STEPS = 500

def permuted(g, seed):
    rng = np.random.default_rng(seed)
    E = list(g.edges); rng.shuffle(E)
    E = [(j, i, w) if rng.random() < 0.5 else (i, j, w) for i, j, w in E]
    return C.wt.rx.G(g.n, E)

def part_AB(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 308)
    u0 = C.init("uniform", g, 4, rng)
    dl = rng.standard_normal(u0.shape); dl *= EPS0 / np.sqrt((dl ** 2).mean(1, keepdims=True))
    rows = []
    for f in FORCES:
        for sch, c, ps in (("damp", 0.5, dict(k=0.1)), ("wave", 0.5, None)):
            a = C.run(g, sch, f, u0, c, STEPS, ps=ps); b = C.run(g, sch, f, u0 + dl, c, STEPS, ps=ps)
            L = np.minimum(a.valid_len, b.valid_len)
            for r in range(4):
                T = int(L[r])
                d = np.sqrt(((a.X[:T, r] - b.X[:T, r]) ** 2).mean(1))
                amp = np.sqrt((a.X[:T, r] ** 2).mean(1))
                hit = np.where(d > 1e-6)[0]
                rows.append(dict(part="A", graph=meta["id"], gkind=spec["block"], seed=spec["seed"], force=f, scheme=sch, c=c, real=r,
                                 div_step=int(hit[0]) if len(hit) else -1, d_max=float(d.max()), d_end=float(d[-1]), growth=float(d.max() / EPS0),
                                 rel_d_end=float(d[-1] / max(amp[-1], 1e-300)), failed=bool(a.fail[r] or b.fail[r]), valid_len=T))
    # B Lipschitz ratio on random pairs
    for f in FORCES:
        for delta in (1e-10, 1e-6, 1e-3):
            U = C.init("uniform", g, 64, rng); D = rng.standard_normal(U.shape); D *= delta / np.sqrt((D ** 2).sum(1, keepdims=True))
            # note: stateful forces (edge_memory, huber_anneal) evaluated from fresh state (t=0)
            ratio = np.sqrt(((C.eval_force(g, f, U + D) - C.eval_force(g, f, U)) ** 2).sum(1)) / delta
            rows.append(dict(part="B", graph=meta["id"], gkind=spec["block"], seed=spec["seed"], force=f, delta=delta,
                             lip_max=float(ratio.max()), lip_med=float(np.median(ratio))))
    return rows

def part_CD(spec):
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 309)
    g2 = permuted(g, spec["seed"] + 5)
    wgt = bool(np.any(g.ew != 1.0))
    rows = []
    kinds = {"uniform": C.init("uniform", g, 8, rng), "rademacher": C.init("rademacher", g, 8, rng),
             "blocky_ties": np.where(np.arange(g.n)[None, :] < g.n // 2, 1.0, -1.0) * np.ones((8, 1))}
    for f in FORCES:
        for k, U in kinds.items():
            d = float(np.abs(C.eval_force(g, f, U) - C.eval_force(g2, f, U)).max())
            rows.append(dict(part="C", graph=meta["id"], gkind=spec["block"], seed=spec["seed"], weighted=wgt, force=f, init=k, perm_defect=d))
    for f in FORCES:
        for ik in ("uniform", "rademacher", "gauss"):
            u0 = C.init(ik, g, 8, rng)
            for sch, c, ps in (("damp", 0.5, dict(k=0.1)), ("wave", 0.5, None)):
                run = C.run(g, sch, f, u0, c, 500, ps=ps)
                s = C.traj_summary(run)
                rows.append(dict(part="D", graph=meta["id"], gkind=spec["block"], seed=spec["seed"], force=f, init=ik, scheme=sch, c=c,
                                 fail_frac=s["fail_frac"], dead_frac=s["dead_frac"], ac_final=s["ac_final"]))
    return rows

def job(args):
    spec, = args
    return part_AB(spec) + part_CD(spec)

if __name__ == "__main__":
    import sys
    quick = len(sys.argv) > 1
    seeds = (0,) if quick else (0, 1, 2, 3)
    specs = [dict(kind="two_block", block=b, bridges=2, sizes=(24, 24), wb=wb, seed=s) for b in ("cycle", "clique", "sbm") for wb in (1.0, 0.25) for s in seeds]
    out = C.pmap(job, [(s,) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(i3.RES / ("t308_discontinuity_quick.csv" if quick else "t308_discontinuity.csv"), rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800)
    A = df[df.part == "A"]; print(A.groupby(["force", "scheme"]).agg(div=("div_step", lambda x: (x >= 0).mean()), growth=("growth", "median"), failed=("failed", "mean")).round(3).to_string())
    B = df[df.part == "B"]; print(B.groupby(["force", "delta"]).lip_max.max().unstack().round(2).to_string())
    Cc = df[df.part == "C"]; print(Cc.groupby(["force", "weighted", "init"]).perm_defect.max().unstack(["weighted", "init"]).to_string())
