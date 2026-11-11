"""T2.09 symmetry / conservation.  (a) u0 vs -u0: X(-u0) should equal -X(u0) (oddness of scheme AND force);  (b) node relabelling
(a) order-preserving permutation, (b) permutation + shuffled edge order -> neighbour order changes (tie handling / summation order);
(c) conservation of the degree-weighted (and plain) node mean under the mean force.  c=0.5, fail threshold 1e6, T=100 for (a,b), 300 for (c).
Schemes: wave + each stabiliser at default and at 'mid' parameter.  Graphs cycle/clique/sbm bridges 2 seeds 0-1; R=8."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
rx = L.rx
T = 100
FORCES = ["mean", "huber", "trim", "owa", "vw", "gauss", "absvw", "median"]

def reld(A, B, n, ref=None):
    ref = B if ref is None else ref
    return float(np.linalg.norm(A[:n] - B[:n]) / max(np.linalg.norm(ref[:n]), 1e-300))

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    n = g.n
    rng = np.random.default_rng(123 + spec["seed"])
    perm = rng.permutation(n)
    e1 = [(int(perm[i]), int(perm[j]), w) for i, j, w in g.edges]
    e2 = list(e1); rng.shuffle(e2)
    gp1, gp2 = rx.G(n, e1), rx.G(n, e2)
    u0 = L.make_u0(spec, g, R)
    up = np.empty_like(u0); up[:, perm] = u0
    d = C.wdeg(g)
    cfgs = [("wave", {}, "-")]
    for b in L.ORIG:
        cfgs.append((b, {}, "default"))
        cfgs.append((b, {L.PARAM[b][0]: L.GRID6[b][2]}, "mid"))
    rows = []
    for sch, ps, lvl in cfgs:
        for f in FORCES:
            c = 0.5
            kw = dict(c=c, ps=ps, fail_amp=1e6)
            a = C.run(g, sch, f, u0, steps=T, **kw)
            b_ = C.run(g, sch, f, -u0, steps=T, **kw)
            r1 = C.run(gp1, sch, f, up, steps=T, **kw)
            r2 = C.run(gp2, sch, f, up, steps=T, **kw)
            row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=sch, level=lvl, force=f, c=c,
                       neg20=reld(a.X + b_.X, 0 * a.X, 21, a.X), neg100=reld(a.X + b_.X, 0 * a.X, T + 1, a.X),
                       perm20=reld(r1.X[:, :, perm], a.X, 21), perm100=reld(r1.X[:, :, perm], a.X, T + 1),
                       perm_shuf20=reld(r2.X[:, :, perm], a.X, 21), perm_shuf100=reld(r2.X[:, :, perm], a.X, T + 1))
            if f == "mean":
                for cc in (0.5, 1.0):
                    r = C.run(g, sch, "mean", u0, cc, 300, ps, fail_amp=1e6)
                    X = r.X
                    ac0 = float(np.sqrt(((u0 - u0.mean(1, keepdims=True)) ** 2).mean()))
                    mw = (X * d[None, None, :]).sum(2) / d.sum()
                    mp = X.mean(2)
                    ok = r.ok
                    row[f"dcdw_max_c{cc}"] = float(np.nanmedian(np.abs(mw - mw[0]).max(0)[ok]) / ac0) if ok.any() else np.nan
                    row[f"dcplain_max_c{cc}"] = float(np.nanmedian(np.abs(mp - mp[0]).max(0)[ok]) / ac0) if ok.any() else np.nan
                    row[f"fail_c{cc}"] = float(r.fail.mean())
            rows.append(row)
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t09_symmetry.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
