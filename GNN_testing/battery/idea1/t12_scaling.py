"""T1.12 amplitude scaling: u0 x{0.01,0.1,1,10} and dc=+3; invariance for homogeneous forces."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib1 import *

FORCES = ["mean", "trim", "owa", "median", "symowa_ext", "symowa_med", "vw", "gauss", "absvw", "huber"]
SCHEMES = [("damp", dict(k=0.1)), ("damp", dict(k=0.3)), ("viscous", dict(nu=0.2)), ("wave", {})]
CS = [0.5, 1.0]
SCALES = [0.01, 0.1, 1.0, 10.0]
T = 500; R = 8

def job(spec):
    g, truth, meta = C.make_graph(spec)
    base = u0_for(g, spec, R)
    rows = []
    for f in FORCES:
        for sch, ps in SCHEMES:
            for c in CS:
                ref = erun(g, sch, f, base, c, T, ps)
                ref_ac = C.ac_rms(ref.X)
                for tag, u0, amp in [(f"x{s}", base * s, 20 * s) for s in SCALES] + [("dc+3", base + 3.0, 23.0)]:
                    run = erun(g, sch, f, u0, c, T, ps, fail_amp=amp)
                    s_ = C.traj_summary(run)
                    # normalised-trajectory deviation from the reference on the common valid window (relative)
                    scale = 1.0 if tag == "dc+3" else float(tag[1:])
                    L = min(ref.valid_len.min(), run.valid_len.min())
                    if tag == "dc+3":
                        A = run.X[:L] - run.X[:L].mean(2, keepdims=True); B = ref.X[:L] - ref.X[:L].mean(2, keepdims=True)
                    else:
                        A = run.X[:L] / scale; B = ref.X[:L]
                    dev = float(np.abs(A - B).max() / max(np.abs(B).max(), 1e-12))
                    rows.append(dict(graph=meta["id"], kind=spec["kind"], seed=spec["seed"], force=f, scheme=sch, ps=str(ps), c=c,
                                     case=tag, fail_frac=s_["fail_frac"], fail_ref=float(ref.fail.mean()),
                                     fail_step_med=s_["mean_fail_step"], dead_frac=s_["dead_frac"], ac_final=s_["ac_final"],
                                     ac_final_ref=C.traj_summary(ref)["ac_final"], rel_dev_common_window=dev, L=int(L)))
    return rows

if __name__ == "__main__":
    specs = core_specs([0, 1, 2])
    if len(sys.argv) > 1:
        t0 = time.time(); r = job(specs[0]); print(len(r), time.time() - t0); sys.exit()
    rs = C.pmap(job, specs, procs=2)
    rows = [r for x in rs for r in x]
    C.write_csv(OUT / "t12_scaling.csv", rows)
    print(len(rows))
