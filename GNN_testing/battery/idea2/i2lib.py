"""battery/idea2/i2lib.py -- shared helpers for the idea-2 tests (nonlinear stabilisers).

Registers (from this module, via @wt.scheme):
  * *_ac   : same schemes, but normalised by the AC-RMS (node-mean removed) instead of the DC-inclusive RMS
             (vel_shrink_ac, drag_ac, limiter_ac, cfl_ac, energy_ac, saturate_ac)
  * *_abs  : same schemes with a FIXED absolute scale `sigma` (param) instead of a per-realisation normaliser
             (vel_shrink_abs, drag_abs, limiter_abs, energy_abs, saturate_abs)
  * composites: limiter_damp(vmax,k), saturate_visc(kappa,nu), drag_damp(alpha,k)
and the scheme-specific "activation" instrumentation (`instr_run`) used by T2.03/T2.06/T2.13.
Every registered scheme is implemented by ONE core function `_core(base, mode, ctx, ...)` that returns
(next_state, active_fraction); the scheme returns the first, the observer the second.
Mode "rms" reproduces the original wave_test schemes (checked bit-exact by selftest()).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # battery/
import common as C           # noqa: E402  (pins BLAS threads)
import numpy as np           # noqa: E402

wt = C.wt
rx = C.rx
EPS = 1e-12
SIG_ABS = float(C.RMS_STD)   # default fixed scale for *_abs: RMS of U(-1,1)
ACT_THR = 0.1                # "active" = the stabiliser changes its controlled quantity by > 10 %

ORIG = ("vel_shrink", "drag", "limiter", "cfl", "energy", "saturate")
# strength direction: +1 = larger parameter is stronger, -1 = smaller parameter is stronger
PARAM = {"vel_shrink": ("alpha", +1), "drag": ("alpha", +1), "limiter": ("vmax", -1), "cfl": ("s", -1),
         "energy": ("kmax", +1), "saturate": ("kappa", -1)}
GRID6 = {  # the six log-spaced values of T2.01, weak -> strong
    "vel_shrink": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "drag": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "limiter": [10.0, 3.0, 1.0, 0.3, 0.1, 0.03],
    "cfl": [1.4, 1.0, 0.7, 0.4, 0.2, 0.1],
    "energy": [0.01, 0.03, 0.1, 0.3, 0.6, 1.0],
    "saturate": [30.0, 10.0, 3.0, 1.0, 0.3, 0.1],
}
GRID10 = {  # finer weak->strong grid for rescue searches (T2.07)
    "vel_shrink": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
    "drag": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
    "limiter": [30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    "cfl": [2.0, 1.4, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01],
    "energy": [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0],   # kmax>1 => k_t can exceed 1 (over-damped/reversing)
    "saturate": [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003],
}
FORCES10 = ["mean", "trim", "owa", "vw", "median", "gauss", "absvw", "symowa_ext", "symowa_med", "huber"]
NONMEAN = [f for f in FORCES10 if f != "mean"]


# --------------------------------------------------------------------------------------------------
def rms_rows(a):
    return wt.rms_rows(a)


def ac_rms_rows(a):
    return wt.rms_rows(a - a.mean(1, keepdims=True))


def _sigma(ctx, mode, key="u0"):
    """per-realisation normaliser (R,1) from u0 (cached)."""
    kk = ("sig", mode, key)
    if kk not in ctx.state:
        if mode == "rms":
            s = rms_rows(ctx.u0)
        elif mode == "ac":
            s = ac_rms_rows(ctx.u0) + EPS
        else:
            s = np.full((ctx.u0.shape[0], 1), float(ctx.ps["sigma"]))
        ctx.state[kk] = s
    return ctx.state[kk]


def _core(base, mode, ctx, cur, prev, F, Fp):
    """-> (nxt, act) ; act (R,) in [0,1] = fraction of nodes (or 0/1 per realisation) where the stabiliser is active."""
    p = ctx.ps
    c2 = ctx.c ** 2
    v = cur - prev
    R = cur.shape[0]
    if base in ("vel_shrink", "drag_damp_x"):
        s = 1 + p["alpha"] * np.abs(v) / _sigma(ctx, mode)
        nxt = cur + (1 - p["k"]) * v + c2 * F / s
        act = (1 / s < 1 - ACT_THR).mean(1)
    elif base == "drag":
        k = np.minimum(1.0, p["alpha"] * np.abs(v) / _sigma(ctx, mode))
        nxt = cur + (1 - k) * v + c2 * F
        act = (k > ACT_THR).mean(1)
    elif base == "drag_damp":
        k = np.minimum(1.0, p["alpha"] * np.abs(v) / _sigma(ctx, mode))
        k = 1 - (1 - k) * (1 - p["k"])
        nxt = cur + (1 - k) * v + c2 * F
        act = (k > ACT_THR).mean(1)
    elif base in ("limiter", "limiter_damp"):
        vm = p["vmax"] * _sigma(ctx, mode)
        kk = p.get("k", 0.0) if base == "limiter_damp" else 0.0
        lim = vm * np.tanh(v / vm)
        nxt = cur + (1 - kk) * lim + c2 * F
        x = np.abs(v) / vm
        ratio = np.where(x > 1e-9, np.tanh(np.minimum(x, 40)) / np.maximum(x, 1e-300), 1.0)
        act = (ratio < 1 - ACT_THR).mean(1)
    elif base == "cfl":
        if mode == "ac":
            nv = ac_rms_rows(v)
            lam = ac_rms_rows(F - Fp) / np.maximum(nv, EPS)
        else:
            nv = rms_rows(v)
            lam = rms_rows(F - Fp) / np.maximum(nv, EPS)
        c2t = np.where(nv > 1e-9, np.minimum(c2, p["s"] * 4 / np.maximum(lam, EPS)), c2)
        nxt = 2 * cur - prev + c2t * F
        act = (c2t[:, 0] < (1 - ACT_THR) * c2).astype(float)
    elif base == "energy":
        if mode == "ac":
            E = ac_rms_rows(cur) ** 2 / _sigma(ctx, "ac") ** 2
        elif mode == "abs":
            E = rms_rows(cur) ** 2 / _sigma(ctx, "abs") ** 2
        else:
            E = rms_rows(cur) ** 2 / rms_rows(ctx.u0) ** 2
        k = p["kmax"] * np.clip(E - 1, 0, 1)
        nxt = cur + (1 - k) * v + c2 * F
        act = (k[:, 0] > ACT_THR).astype(float)
    elif base in ("saturate", "saturate_visc"):
        if "phi" not in ctx.state:
            if mode == "ac":
                r = ac_rms_rows(F)
            elif mode == "abs":
                r = np.full((R, 1), float(p["sigma"]))
            else:
                r = rms_rows(F)
            ctx.state["phi"] = p["kappa"] * r + EPS
        phi = ctx.state["phi"]
        S = phi * np.tanh(F / phi)
        nxt = 2 * cur - prev + c2 * S
        if base == "saturate_visc":
            nxt = nxt + p["nu"] * (S - phi * np.tanh(Fp / phi))
        y = np.abs(F) / phi
        ratio = np.where(y > 1e-9, np.tanh(np.minimum(y, 40)) / np.maximum(y, 1e-300), 1.0)
        act = (ratio < 1 - ACT_THR).mean(1)
    else:
        raise ValueError(base)
    return nxt, act


def _reg(name, base, mode, **defaults):
    def fn(ctx, cur, prev, F, Fp):
        return _core(base, mode, ctx, cur, prev, F, Fp)[0]
    fn.__doc__ = f"idea2 {base} normalised by {mode}"
    wt.scheme(name, **defaults)(fn)


_DEF = {"vel_shrink": dict(alpha=1.0, k=0.0), "drag": dict(alpha=0.5), "limiter": dict(vmax=0.3), "cfl": dict(s=0.9),
        "energy": dict(kmax=0.5), "saturate": dict(kappa=1.0)}
for _b, _d in _DEF.items():
    _reg(f"{_b}_ac", _b, "ac", **_d)
    if _b != "cfl":
        _reg(f"{_b}_abs", _b, "abs", **_d, sigma=SIG_ABS)
_reg("limiter_damp", "limiter_damp", "rms", vmax=0.3, k=0.1)
_reg("saturate_visc", "saturate_visc", "rms", kappa=1.0, nu=0.1)
_reg("drag_damp", "drag_damp", "rms", alpha=0.5, k=0.05)
_reg("limiter_damp_ac", "limiter_damp", "ac", vmax=0.3, k=0.1)
_reg("saturate_visc_ac", "saturate_visc", "ac", kappa=1.0, nu=0.1)


def parse_scheme(name):
    """-> (base, mode) for any scheme this module can instrument ('wave' etc. -> (None,None))."""
    mode = "rms"
    n = name
    if n.endswith("_ac"):
        mode, n = "ac", n[:-3]
    elif n.endswith("_abs"):
        mode, n = "abs", n[:-4]
    if n in _DEF or n in ("limiter_damp", "saturate_visc", "drag_damp"):
        return n, mode
    return None, None


def instr_run(g, scheme, force, u0, c, steps, ps=None, pf=None, fail_amp=None, extra=None):
    """Run with scheme-specific activation recording.  Returns Run, act (steps, R) (NaN once failed).
    `extra(t, ctx, cur, prev, F, Fp, nxt, alive)` is an optional second observer."""
    u0 = np.atleast_2d(np.asarray(u0, float))
    ps = C.resolve_params(wt.SCHEMES, scheme, ps)
    pf = C.resolve_params(wt.FORCES, force, pf)
    base, mode = parse_scheme(scheme)
    act = np.full((steps, u0.shape[0]), np.nan)

    def obs(t, ctx, cur, prev, F, Fp, nxt, alive):
        if base is not None:
            a = _core(base, mode, ctx, cur, prev, F, Fp)[1].astype(float)
            a[~alive] = np.nan
            act[t] = a
        if extra is not None:
            extra(t, ctx, cur, prev, F, Fp, nxt, alive)

    amp = wt.FAIL_AMP if fail_amp is None else float(fail_amp)
    X, fs = C._loop(g, scheme, force, u0, c, steps, ps, pf, amp, observer=obs)
    return C.Run(X, fs, g, u0, float(c), scheme, force, ps, pf, steps, amp), act


# --------------------------------------------------------------------------------------------------
def strength_params(base, strength_idx, grid=GRID10):
    return {PARAM[base][0]: grid[base][strength_idx]}


def scaled_strength(base, val, factor):
    """parameter value with `factor` times the strength (2x, 4x): larger-is-stronger multiplies, else divides."""
    return val * factor if PARAM[base][1] > 0 else val / factor


def cfg_label(scheme, ps):
    return scheme + ("[" + ",".join(f"{k}={v:g}" for k, v in ps.items()) + "]" if ps else "")


def graphs_dev(kinds=("cycle", "clique", "sbm"), bridges=(2,), seeds=(0, 1, 2, 3), sizes=(24, 24), wb=1.0):
    return [dict(kind=k, sizes=sizes, bridges=b, wb=wb, seed=s) for k in kinds for b in bridges for s in seeds]


def make_u0(spec, g, R=8, kind="uniform", truth=None, **kw):
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    return C.init(kind, g, R, rng, truth=truth, **kw)


def lifetime(X, g, fail_step, frac=0.05):
    """first step where AC-RMS < frac * AC-RMS(u0); T if never; NaN if failed."""
    a = C.ac_rms(X, g)                # (T+1, R)
    thr = frac * np.maximum(a[0], 1e-300)
    below = a < thr[None]
    T1 = X.shape[0]
    out = np.where(below.any(0), below.argmax(0), T1 - 1).astype(float)
    out[fail_step >= 0] = np.nan
    return out


def nanmed(a):
    a = np.asarray(a, float)
    return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")


def nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def selftest():
    """my core == original wave_test schemes, bit-exact, on a couple of graphs/forces."""
    import itertools
    spec = dict(kind="sbm", sizes=(24, 24), bridges=2, wb=1.0, seed=0)
    g, truth, meta = C.make_graph(spec)
    u0 = make_u0(spec, g)
    # the originals are registered under their own names; mine under composites. Compare via a re-registered 'rms' twin.
    for b in ORIG:
        _reg(f"{b}_twin", b, "rms", **_DEF[b])
    ok = True
    for b, f in itertools.product(ORIG, ("mean", "owa")):
        r1 = C.run(g, b, f, u0, 1.0, 300)
        r2 = C.run(g, f"{b}_twin", f, u0, 1.0, 300)
        same = np.array_equal(r1.X, r2.X) and np.array_equal(r1.fail_step, r2.fail_step)
        ok &= same
        print(f"  twin check {b:10s}+{f:5s}: {'bit-exact' if same else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    print("selftest ok:", selftest())
