"""Idea-1 local helpers (early-exit runs, scheme selection, extra schemes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
import numpy as np
wt = C.wt

def scheme_for(k=0.0, nu=0.0):
    if nu == 0: return ("wave", {}) if k == 0 else ("damp", dict(k=k))
    if k == 0: return "viscous", dict(nu=nu)
    return "damp_visc", dict(k=k, nu=nu)

def erun(g, scheme, force, u0, c, steps, ps=None, pf=None, fail_amp=None):
    """Like C.run but stops the loop once every realisation has failed (X beyond is 0, as in C.run)."""
    ps = C.resolve_params(wt.SCHEMES, scheme, ps); pf = C.resolve_params(wt.FORCES, force, pf)
    amp = wt.FAIL_AMP if fail_amp is None else float(fail_amp)
    X, fs = C._loop(g, scheme, force, C._as2d(u0), c, steps, ps, pf, amp, early_stop=True, record=True)
    return C.Run(X, fs, g, C._as2d(u0), float(c), scheme, force, ps, pf, steps, amp)

def core_specs(seeds, bridges=2, sizes=(24, 24), wb=1.0, kinds=("cycle", "clique", "sbm")):
    return [dict(kind=k, sizes=sizes, bridges=bridges, wb=wb, seed=s) for k in kinds for s in seeds]

def u0_for(g, spec, R, kind="uniform", **kw):
    return C.init(kind, g, R, np.random.default_rng(spec["seed"] * 1000 + 17), **kw)

OUT = Path(__file__).parent / "results"
