"""T2.04 normalisation-flaw check.  Schemes normalise by RMS(u0) (DC included).  Conditions: u0, u0+3 (shift), 0.01*u0, 10*u0
(scale), impulse (single node).  mean force is shift- and scale-equivariant, so a well-normalised scheme must satisfy
X(u0+3)-3 == X(u0), X(s*u0)/s == X(u0).  Compare original (rms) vs *_ac variants.  Also: owa (not shift invariant) at dc=+3:
does the stabiliser still rescue?  Graphs cycle/clique/sbm bridges 2 seeds 0-1; R=8; T=300 (invariance), 500 (owa)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2lib as L
import numpy as np
C = L.C
T = 300

def acn(X):
    return X - X.mean(2, keepdims=True)

def dev(Xa, Xb, n=T + 1):
    A, B = acn(Xa[:n]), acn(Xb[:n])
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300))

def job(args):
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    u0 = L.make_u0(spec, g, R)
    imp = C.init("impulse", g, R, np.random.default_rng(0), node=0, amp=1.0)
    rows = []
    for b in L.ORIG:
        for mode in ("rms", "ac"):
            sch = b if mode == "rms" else b + "_ac"
            for lvl, i in (("mid", 2), ("strong", 4)):
                ps = {L.PARAM[b][0]: L.GRID6[b][i]}
                for c in (0.5, 1.0):
                    base = C.run(g, sch, "mean", u0, c, T, ps, fail_amp=1e6)
                    wave = C.run(g, "wave", "mean", imp, c, T, fail_amp=1e6)   # plain-wave reference FROM THE IMPULSE IC
                    conds = {"shift+3": (u0 + 3, lambda X: X - 3, base.X), "scale0.01": (0.01 * u0, lambda X: X / 0.01, base.X),
                             "scale10": (10 * u0, lambda X: X / 10, base.X), "impulse": (imp, lambda X: X, wave.X)}
                    for name, (uu, back, ref) in conds.items():
                        r = C.run(g, sch, "mean", uu, c, T, ps, fail_amp=1e6)
                        row = dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=b, mode=mode, level=lvl, param=list(ps.values())[0],
                                   c=c, force="mean", cond=name, dev=dev(back(r.X), ref),
                                   ref_blowup=float(C.ac_rms(base.X, g)[-1].max() > 20 * C.ac_rms(base.X, g)[0].max()))
                        if name == "impulse":   # deviation from plain wave (the "do-no-harm" version for a non-uniform IC) + activation
                            _, act = L.instr_run(g, sch, "mean", imp, c, T, ps, fail_amp=1e6)
                            row["act_mean"] = L.nanmean(act); row["act_first50"] = L.nanmean(act[:50])
                        rows.append(row)
            # owa at DC offset
            for lvl, i in (("mid", 2), ("strong", 4)):
                ps = {L.PARAM[b][0]: L.GRID6[b][i]}
                for f in ("owa", "trim"):
                    for dc in (0.0, 3.0):
                        r = C.run(g, sch, f, u0 + dc, 1.0, 500, ps)
                        rows.append(dict(graph=meta["id"], seed=spec["seed"], kind=spec["kind"], scheme=b, mode=mode, level=lvl,
                                         param=list(ps.values())[0], c=1.0, force=f, cond=f"dc{dc:g}", fail_frac=float(r.fail.mean()),
                                         ac_final=L.nanmed(C.ac_final(r.X, r.fail_step))))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    specs = L.graphs_dev(seeds=(0, 1))
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)
    rows = [r for rs in out for r in rs]
    C.write_csv(Path(__file__).parent / "results" / "t04_normalisation.csv", rows)
    print("rows", len(rows), "time %.0fs" % (time.time() - t0))
