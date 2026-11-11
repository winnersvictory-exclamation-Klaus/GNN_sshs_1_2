"""T1.03 minimal rescue: smallest k (damp) / nu (viscous) with fail_frac=0 at T=2000, scan + bisection."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

FORCES = ["mean", "trim", "owa", "vw", "median", "gauss", "absvw", "symowa_ext", "symowa_med", "huber"]
T = 2000; R = 8
GRID = {"damp": np.geomspace(1e-3, 1.0, 19), "viscous": np.geomspace(1e-2, 2.0, 17)}

def ff(g, scheme, f, u0, c, x):
    ps = dict(k=x) if scheme == "damp" else dict(nu=x)
    return C.fail_frac(g, scheme, f, u0, c, T, ps)

def job(args):
    spec, force, cs = args
    g, truth, meta = C.make_graph(spec)
    u0 = u0_for(g, spec, R)
    rows = []
    for c in cs:
        for scheme in ("damp", "viscous"):
            grid = GRID[scheme]
            fl = np.array([ff(g, scheme, force, u0, c, x) for x in grid])
            st = np.flatnonzero(fl == 0)
            if len(st) == 0:
                kmin, status = np.nan, "never_stable"
            elif st[0] == 0:
                kmin, status = grid[0], "stable_at_lowest"
            else:
                lo, hi = grid[st[0] - 1], grid[st[0]]
                while hi / lo > 1.05:
                    mid = np.sqrt(lo * hi)
                    if ff(g, scheme, force, u0, c, mid) == 0: hi = mid
                    else: lo = mid
                kmin, status = hi, "ok"
            nonmono = bool(len(st) > 0 and (fl[st[0]:] > 0).any())
            rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=force, c=c, scheme=scheme,
                             x_min=kmin, status=status, nonmonotone=nonmono, stable_frac_of_grid=float((fl == 0).mean()),
                             fail_at_max=fl[-1], first_stable_idx=int(st[0]) if len(st) else -1))
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2])
    if len(sys.argv) > 1:
        t0 = time.time(); print(job((specs[0], "owa", [1.0]))); print(time.time() - t0); sys.exit()
    jobs = [(s, f, [0.5, 1.0]) for s in specs for f in FORCES]
    t0 = time.time()
    rs = C.pmap(job, jobs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t03_rescue.csv", rows)
    print(len(rows), time.time() - t0)
