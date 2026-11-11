"""Shared helpers for idea 3 tests (forces of wave_test, own extra forces)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
import numpy as np
wt = C.wt
RES = Path(__file__).parent / "results"

FORCES19 = ["trim", "owa", "vw", "odd_owa", "odd_vw", "huber", "tanh", "amp", "gauss", "absvw",
            "symowa_ext", "symowa_med", "median", "huber_local", "huber_global", "amp_gated",
            "blend_agree", "huber_anneal", "edge_memory"]
# parameter-free-of-state "Form A" (potential) / "Form B" classification (design doc T3.05)
FORM_A = ["huber", "tanh", "amp", "edge_memory"]
FORM_B = ["gauss", "absvw", "symowa_ext", "symowa_med", "median", "trim"]


# ---- own forces -----------------------------------------------------------
def _dw_mean_removed(ctx, F):
    d = np.asarray(C.wdeg(ctx.g))
    return F - (F * d).sum(-1, keepdims=True) / d.sum()


@wt.force("owa_mc")
def _owa_mc(ctx, u):
    """owa minus its degree-weighted mean: sum_i d_i F_i = 0 (momentum corrected)."""
    return _dw_mean_removed(ctx, wt.rx.aggregate(ctx.g, u, "owa") - u)


@wt.force("vw_mc")
def _vw_mc(ctx, u):
    return _dw_mean_removed(ctx, wt.rx.aggregate(ctx.g, u, "vw") - u)


@wt.force("trim_mc")
def _trim_mc(ctx, u):
    return _dw_mean_removed(ctx, wt.rx.aggregate(ctx.g, u, "trim") - u)


# core graph specs used in many tests (dev seeds)
def core_specs(seeds=C.DEV_SEEDS):
    return C.zoo("core", seeds=seeds)


def graph_kind_specs(seed=0, bridges=2):
    return [dict(kind=k, bridges=bridges, seed=seed) for k in ("cycle", "clique", "sbm")]
