#!/usr/bin/env python3
"""003-bridge-selectivity: plan section 2 development mechanism test (development graphs only).

Part 1  given-block preservation on 24+24 clique/cycle graphs with 1 or 4 bridges of weight 0.25 (graph seed 0):
        first bridge force hand check, linear single-mode references, energy curves, then a small shared uniform
        noise addition with block-mean gap / within-block RMS tracking.  If a hard reference check fails, a
        BLOCKER.md is written and part 2 is not started.
Part 2  random-start discovery on 48 development graphs: Mean / fixed Huber / legacy endpoint-local Huber, each in
        damped wave and first-order diffusion, at h = 0.5 and 0.25, plus normalized spectral.  Bridge vs interior
        effective stiffness, windowed raw / low-pass edge-RMS scores, AUC, q=2 Kruskal ARI and the Kruskal gap.

See README.md for the exact command, settings, tolerances, censoring rules and output layout.
Exits 1 if a hard validation check fails.
"""
import sys

sys.dont_write_bytecode = True  # keep the shared reference-code directories free of __pycache__

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.environ.setdefault("GNN_REPO", str(REPO))          # wave_test.py locates rank_experiment.py through this
sys.path.insert(0, str(REPO / "GNN_testing" / "battery"))
import common as C  # noqa: E402  (imports wave_test as C.wt and rank_experiment as C.rx; pins BLAS to 1 thread)
import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

wt, rx = C.wt, C.rx

# ---- fixed design (from the request; nothing here was chosen from results) ---------------------------------
AMP = 20.0             # failure: not all(|u| < AMP) on any updated substep (nonfinite included)
EMERGENCY = 1e8        # diagnostic safety censor: stop integrating an init once max|u| exceeds this or nonfinite
GAMMA = 0.1            # wave damping (centered), one fixed value for this pilot
HS = (0.5, 0.25)
T_PHYS = 250.0
DT_SAVE = 0.5          # saved frames t = 0, 0.5, ..., 250 (h=0.25 keeps every second substep)
N_SAVE = int(round(T_PHYS / DT_SAVE)) + 1
R = 8
FC = 0.02              # low-pass cutoff, cycles per saved frame, applied within each window
WINDOWS = (("early", 0.0, 50.0), ("late", 125.0, 250.0))
READOUTS = ("raw", "lp")
FORCES = ("mean", "huber", "huber_local")
DYNS = ("wave", "diffusion")
TAU_FIXED = 0.5
KAPPA = 1.0
EPS = 1e-12
FAMILIES = ("clique", "cycle", "sbm")
P1_FAMILIES = ("clique", "cycle")
P1_BRIDGES = (1, 4)
P1_WB = 0.25
P1_SEED = 0
P1_NOISE = 0.05        # part 1 noise: U(-0.05, 0.05), one vector per graph, shared by all methods
P2_SIZES = ((24, 24), (16, 32))
P2_BRIDGES = (1, 4)
P2_SEEDS = (0, 1, 2, 3)
SEED_ROOT = 3          # streams: default_rng([3, part, family_index, n0, n1, bridges, graph_seed])
QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)
S_BINS = (0.0, 0.25, 0.5, 1.0, math.inf)
CHECKPOINT_T = (0.0, 50.0, 125.0, 250.0)
GATE = 0.05            # development graph-failure gate (fraction of graphs with any failed init)
SENS_DARI = 0.1        # |ARI(h=.5) - ARI(h=.25)| above this flags h-sensitivity (descriptive)
HIGH_AUC = 0.99        # descriptive label for "near-perfect AUC"
INTERIOR_FLOOR = 1e-8  # interior median score below this: interior converged to round-off, ratio not summarized
GAP_NEAR_TIE = 1e-12   # |gap| <= this x max score -> near tie (exact zero reported separately)
TOL_IMPL = 1e-13       # wave_test forces vs independent dense implementation, / max(1, max|F|)
TOL_HAND_ABS = 1e-15   # first bridge force: absolute error
TOL_HAND_REL = 1e-12   # first bridge force: relative error (needed for the EPS-sized endpoint-local force)
TOL_MODE = 1e-9        # single-mode amplitude vs closed form, and off-mode leakage
TOL_CONSERVE = 1e-12   # |sum d F| / sum d|F| for the edge-antisymmetric Mean / fixed Huber (reported)


# ---- graphs, initial states, forces ------------------------------------------------------------------------
def p1_specs():
    return [dict(kind="two_block", block=b, sizes=(24, 24), bridges=nb, wb=P1_WB, seed=P1_SEED)
            for b in P1_FAMILIES for nb in P1_BRIDGES]


def p2_specs():
    return [dict(kind="two_block", block=b, sizes=sz, bridges=nb, wb=1.0, seed=s)
            for b in FAMILIES for sz in P2_SIZES for nb in P2_BRIDGES for s in P2_SEEDS]


def stream(part, spec):
    """Deterministic RNG for part 1 noise / part 2 initial states (distinct from t310 seed*1000+310)."""
    n0, n1 = spec["sizes"]
    return np.random.default_rng([SEED_ROOT, part, FAMILIES.index(spec["block"]), n0, n1, spec["bridges"],
                                  spec["seed"]])


def gmeta(spec, meta):
    return dict(graph=meta["id"], family=spec["block"], sizes=f"{spec['sizes'][0]}+{spec['sizes'][1]}",
                bridges=spec["bridges"], wb=spec["wb"], seed=spec["seed"])


def make_force(g, name):
    """Existing wave_test force (degree-normalized acceleration F(u), (R,n) -> (R,n))."""
    pf = C.resolve_params(wt.FORCES, name)
    ctx = wt.Ctx(g=g, c=float("nan"), pf=pf, ps={}, u0=None)
    fn = wt.FORCES[name][0]
    return lambda u: fn(ctx, u)


def dense_force(g, name, u):
    """Independent dense implementation of the three forces for the reference checks (u is 1-D)."""
    A = g.A
    d = A.sum(1)
    X = u[None, :] - u[:, None]                      # X[i, j] = u_j - u_i
    if name == "mean":
        P = X
    elif name == "huber":
        P = np.clip(X, -TAU_FIXED, TAU_FIXED)
    else:
        tau = np.array([KAPPA * np.median(np.abs(X[i, A[i] > 0])) + EPS for i in range(g.n)])
        P = np.clip(X, -tau[:, None], tau[:, None])
    return (A * P).sum(1) / d


def local_tau(g, U):
    """Endpoint thresholds tau_i = kappa * median_j |u_j - u_i| + EPS (unweighted, valid neighbours), U (..., n)."""
    sh = U.shape
    flat = U.reshape(-1, sh[-1])
    return (KAPPA * wt.nb_median(g, np.abs(wt.diffs(g, flat))) + EPS).reshape(sh)


# ---- integration -------------------------------------------------------------------------------------------
def integrate(force, u0, h, dyn, d, gamma=GAMMA, velocity=False):
    """Integrate u'' + gamma u' = F(u) (wave, centered damping, zero initial velocity:
    u1 = u0 + h^2/2 F(u0); u+ = (2u - (1 - gamma h/2) u- + h^2 F(u)) / (1 + gamma h/2)) or u' = F(u)
    (diffusion, forward Euler) to T_PHYS.  Saved frames every DT_SAVE.  Failure = first updated substep with
    not all(|u| < AMP); an init is censored (NaN from then on) at its first nonfinite substep or once
    max|u| > EMERGENCY.  velocity=True also returns V at the saved frames: wave (u_{k+1} - u_{k-1}) / 2h
    (exactly 0 at t=0; the final frame uses one extra substep that is not saved or failure-checked);
    diffusion F(u)."""
    steps = int(round(T_PHYS / h))
    every = int(round(DT_SAVE / h))
    assert steps % every == 0 and steps // every + 1 == N_SAVE
    Rn, n = u0.shape
    X = np.full((N_SAVE, Rn, n), np.nan)
    V = np.full((N_SAVE, Rn, n), np.nan) if velocity else None
    X[0] = u0
    first_fail = np.full(Rn, -1)
    censor = np.full(Rn, -1)
    alive = np.ones(Rn, bool)
    prev, cur = None, u0.astype(float).copy()
    a, b, h2 = 1 + gamma * h / 2, 1 - gamma * h / 2, h * h
    with np.errstate(all="ignore"):
        for k in range(steps + 1):
            F = np.full((Rn, n), np.nan)
            if alive.any():
                F[alive] = force(cur[alive])
            if dyn == "diffusion":
                nxt = cur + h * F
                if velocity and k % every == 0:
                    V[k // every] = F
            else:
                nxt = cur + 0.5 * h2 * F if k == 0 else (2 * cur - b * prev + h2 * F) / a
                if velocity and k % every == 0:
                    V[k // every] = 0.0 * cur if k == 0 else (nxt - prev) / (2 * h)
            if k == steps:
                break
            bad = alive & ~(np.abs(nxt) < AMP).all(1) & (first_fail < 0)
            first_fail[bad] = k + 1
            nf = alive & ~np.isfinite(nxt).all(1)
            em = alive & ~nf & (np.abs(nxt).max(1) > EMERGENCY)
            censor[nf | em] = k + 1
            alive &= ~(nf | em)
            nxt[~alive] = np.nan
            if (k + 1) % every == 0:
                X[(k + 1) // every] = nxt
            prev, cur = cur, nxt
    return dict(X=X, V=V, first_fail=first_fail, censor=censor, h=h, steps=steps, every=every, dyn=dyn)


# ---- part 1 ------------------------------------------------------------------------------------------------
def potential(g, name, U):
    """Edge potential sum_e w_e Phi(s_e), Phi' = phi, for Mean and fixed Huber; U (..., n) -> (...)."""
    s = U[..., g.ej] - U[..., g.ei]
    if name == "mean":
        P = 0.5 * s * s
    else:
        a = np.abs(s)
        P = np.where(a <= TAU_FIXED, 0.5 * s * s, TAU_FIXED * a - 0.5 * TAU_FIXED ** 2)
    return P @ g.ew


def block_stats(X, d, truth):
    """Degree-weighted block means, gap m0 - m1, within-block RMS about each block mean and pooled; X (T, n)."""
    out = {}
    pooled = np.zeros(X.shape[0])
    for b in (0, 1):
        sel = truth == b
        db = d[sel]
        m = X[:, sel] @ db / db.sum()
        v = ((X[:, sel] - m[:, None]) ** 2) @ db
        out[f"m{b}"], out[f"rms_in{b}"] = m, np.sqrt(v / db.sum())
        pooled += v
    out["gap"] = out["m0"] - out["m1"]
    out["rms_in"] = np.sqrt(pooled / d.sum())
    return out


def first_true(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def hand_bridge_force(g, truth, meta, name):
    """Hand value of the first force at each bridge endpoint for +1 / -1 blocks.  Each endpoint i has intra degree
    k_i (unit weights), nb_i bridges of weight wb and weighted degree d_i = k_i + wb nb_i; every bridge difference
    is -2 seen from the +1 side.  Flux per bridge: Mean wb*2, fixed Huber wb*min(2, tau), endpoint-local
    wb*min(2, tau_i) with tau_i = median of {0 x k_i, 2 x nb_i} + EPS evaluated per endpoint."""
    rows = []
    wb = float(g.ew[meta["bridge_idx"][0]])
    ends = sorted(set(g.ei[meta["bridge_idx"]]) | set(g.ej[meta["bridge_idx"]]))
    for i in ends:
        i = int(i)
        nb = int(sum((g.ei[e] == i or g.ej[e] == i) for e in meta["bridge_idx"]))
        k = int(g.deg[i]) - nb
        di = k + wb * nb
        side = +1.0 if truth[i] == 0 else -1.0
        vals = np.sort(np.r_[np.zeros(k), 2.0 * np.ones(nb)])
        cnt = len(vals)
        med = 0.5 * (vals[(cnt - 1) // 2] + vals[cnt // 2])
        tau = {"mean": math.inf, "huber": TAU_FIXED, "huber_local": KAPPA * med + EPS}[name]
        flux = wb * min(2.0, tau)                    # magnitude per bridge
        rows.append(dict(node=i, side=int(side), intra_deg=k, n_bridges_at_node=nb, weighted_degree=di,
                         tau_hand=tau if math.isfinite(tau) else float("nan"), flux_per_bridge_hand=flux,
                         F_hand=-side * nb * flux / di))
    return rows


def mode_checks(spec, g, meta):
    """Mean force from a single D-orthonormal L_rw eigenvector (Fiedler and top mode) at both h:
    undamped wave vs cos(w t), w = 2 asin(h sqrt(lam)/2)/h (the 002 reference); and, as integrator checks for
    this test's schemes, damped wave vs the closed-form two-root solution of
    (1 + gh/2) a+ = (2 - h^2 lam) a - (1 - gh/2) a-, a1 = 1 - h^2 lam/2, and forward-Euler diffusion vs (1 - h lam)^n."""
    lam, Phi, d = C.spectrum(g)
    F = make_force(g, "mean")
    rows = []
    for h in HS:
        every = int(round(DT_SAVE / h))
        nsub = np.arange(N_SAVE) * every
        for k in (1, g.n - 1):
            phi = Phi[:, k]
            for kind, dyn, gam in (("wave_undamped", "wave", 0.0), ("wave_damped", "wave", GAMMA),
                                   ("diffusion", "diffusion", 0.0)):
                run = integrate(F, phi[None, :], h, dyn, d, gamma=gam)
                a = run["X"][:, 0, :] @ (d * phi)
                if kind == "wave_undamped":
                    w = 2 * math.asin(h * math.sqrt(lam[k]) / 2) / h
                    expect = np.cos(w * h * nsub)
                elif kind == "wave_damped":
                    A2, A1, A0 = 1 + gam * h / 2, -(2 - h * h * lam[k]), 1 - gam * h / 2
                    z1, z2 = np.roots([A2, A1, A0]).astype(complex)
                    a1 = 1 - 0.5 * h * h * lam[k]
                    c1 = (a1 - z2) / (z1 - z2)
                    expect = np.real(c1 * z1 ** nsub + (1 - c1) * z2 ** nsub)
                else:
                    expect = (1 - h * lam[k]) ** nsub
                off = np.abs(run["X"][:, 0, :] - a[:, None] * phi[None, :]).max()
                rows.append(dict(graph=meta["id"], h=h, mode=k, lam=float(lam[k]), h2lam=float(h * h * lam[k]),
                                 hlam=float(h * lam[k]), check=kind,
                                 reference=("002 undamped cos reference" if kind == "wave_undamped"
                                            else "integrator closed form (added check)"),
                                 max_amp_dev=float(np.abs(a - expect).max()), max_offmode=float(off),
                                 amp_T=float(a[-1]), expect_T=float(expect[-1])))
    return rows


def run_summary(st, t, first_fail, censor, h):
    """Half-gap crossing and internal-RMS summaries for one part 1 run (saved-frame resolution)."""
    gap, rin = st["gap"], st["rms_in"]
    r = gap / gap[0]
    k = first_true(r <= 0.5)
    out = dict(gap0=float(gap[0]), rms_in0=float(rin[0]), half_gap_t=t[k] if k >= 0 else float("nan"),
               half_gap_censored=int(k < 0), min_gap_ratio=float(np.nanmin(r)), gap_ratio_T=float(r[-1]))
    if k >= 0:
        after = r[k:]
        up = np.flatnonzero((after[1:] > 0.5) & (after[:-1] <= 0.5))
        out.update(n_returns_above_half=int(up.size), first_return_t=t[k + 1 + up[0]] if up.size else float("nan"),
                   max_gap_ratio_after_half=float(np.nanmax(after)), max_abs_gap_ratio_after_half=float(np.nanmax(np.abs(after))),
                   rms_in_at_half=float(rin[k]), rms_in_at_half_over_initial=float(rin[k] / rin[0]) if rin[0] > 0 else float("nan"),
                   rms_in_over_abs_gap_at_half=float(rin[k] / abs(gap[k])),
                   rms_in0_at_half=float(st["rms_in0"][k]), rms_in1_at_half=float(st["rms_in1"][k]))
    else:
        out.update(n_returns_above_half=0, first_return_t=float("nan"), max_gap_ratio_after_half=float("nan"),
                   max_abs_gap_ratio_after_half=float("nan"), rms_in_at_half=float("nan"),
                   rms_in_at_half_over_initial=float("nan"), rms_in_over_abs_gap_at_half=float("nan"),
                   rms_in0_at_half=float("nan"), rms_in1_at_half=float("nan"))
    ki = first_true(rin <= 0.5 * rin[0]) if rin[0] > 0 else -1
    out.update(internal_half_t=t[ki] if ki >= 0 else float("nan"), internal_half_censored=int(rin[0] > 0 and ki < 0),
               max_rms_in=float(np.nanmax(rin)), rms_in_T=float(rin[-1]),
               half_gap_over_internal_half=(out["half_gap_t"] / t[ki]) if (k >= 0 and ki > 0) else float("nan"),
               failed=int(first_fail >= 0), first_fail_t=first_fail * h if first_fail >= 0 else float("nan"),
               censored=int(censor >= 0))
    return out


def part1(checks):
    t = np.arange(N_SAVE) * DT_SAVE
    force_rows, mode_rows, curves, summ, hcmp, impl = [], [], [], [], [], []
    for spec in p1_specs():
        g, truth, meta = C.make_graph(spec)
        base = gmeta(spec, meta)
        d = C.wdeg(g)
        ublock = np.where(truth == 0, 1.0, -1.0)
        noise = stream(1, spec).uniform(-P1_NOISE, P1_NOISE, g.n)
        starts = {"noiseless": ublock, "noisy": ublock + noise}
        bm = meta["bridge_mask"]
        endpoints = np.unique(np.r_[g.ei[bm], g.ej[bm]])
        lam_max = float(C.spectrum(g)[0][-1])
        # first bridge force, noiseless +1 / -1
        for name in FORCES:
            Fc = make_force(g, name)(ublock[None, :])[0]
            hand = hand_bridge_force(g, truth, meta, name)
            off = np.delete(Fc, endpoints)
            # observed per-bridge action/reaction |d_a F_a + d_b F_b| (reported, not assumed)
            imb = max(abs(d[g.ei[e]] * Fc[g.ei[e]] + d[g.ej[e]] * Fc[g.ej[e]]) for e in meta["bridge_idx"])
            for hr in hand:
                i = hr["node"]
                Fi = float(Fc[i])
                err = abs(Fi - hr["F_hand"])
                force_rows.append(dict(base, force=name, **hr, F_code=Fi, flux_per_bridge_code=abs(Fi) * hr["weighted_degree"] / hr["n_bridges_at_node"],
                                       abs_err=err, rel_err=err / abs(hr["F_hand"]),
                                       sign_ok=int(np.sign(Fi) == -hr["side"]),
                                       max_abs_F_off_endpoints=float(np.abs(off).max()),
                                       bridge_reaction_imbalance_max=float(imb), sum_dF=float(Fc @ d),
                                       wave_u1_h05=float(ublock[i] + 0.5 * 0.25 * Fi), diffusion_u1_h05=float(ublock[i] + 0.5 * Fi),
                                       label="sanity check only (block start, not a partition score)"))
            for u in starts.values():           # wave_test force vs independent dense implementation
                Fd = dense_force(g, name, u)
                Fw = make_force(g, name)(u[None, :])[0]
                impl.append(float(np.abs(Fw - Fd).max() / max(1.0, float(np.abs(Fd).max()))))
        mode_rows += mode_checks(spec, g, meta)
        # block-start runs
        for sname, u in starts.items():
            for name in FORCES:
                Fn = make_force(g, name)
                res = {}
                for dyn in DYNS:
                    for h in HS:
                        run = integrate(Fn, u[None, :], h, dyn, d, velocity=True)
                        X, V = run["X"][:, 0], run["V"][:, 0]
                        st = block_stats(X, d, truth)
                        kin = 0.5 * (V ** 2) @ d
                        pot = potential(g, name, X) if name != "huber_local" else np.full(N_SAVE, np.nan)
                        rate = 2 * kin if dyn == "diffusion" else 2 * GAMMA * kin     # continuous dE/dt = -rate
                        diss = np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * DT_SAVE)]
                        if dyn == "wave":
                            E = kin + pot if name != "huber_local" else np.full(N_SAVE, np.nan)
                        else:
                            E = pot
                        bal = (E + diss - E[0]) / E[0] if name != "huber_local" else np.full(N_SAVE, np.nan)
                        maxabs = np.abs(X).max(1)
                        cen = X @ d / d.sum()
                        for j in range(N_SAVE):
                            curves.append(dict(base, start=sname, force=name, dyn=dyn, h=h, t=t[j], m0=st["m0"][j],
                                               m1=st["m1"][j], gap=st["gap"][j], gap_ratio=st["gap"][j] / st["gap"][0],
                                               rms_in0=st["rms_in0"][j], rms_in1=st["rms_in1"][j], rms_in=st["rms_in"][j],
                                               center=cen[j], maxabs=maxabs[j],
                                               kinetic=kin[j] if dyn == "wave" else float("nan"),
                                               potential=pot[j], energy=E[j],
                                               energy_ratio=E[j] / E[0] if name != "huber_local" else float("nan"),
                                               dissipated_trapz=diss[j] if name != "huber_local" else float("nan"),
                                               balance_resid=bal[j]))
                        ff, cs = int(run["first_fail"][0]), int(run["censor"][0])
                        sm = dict(base, start=sname, force=name, dyn=dyn, h=h, h2_lam_max=h * h * lam_max,
                                  h_lam_max=h * lam_max, **run_summary(st, t, ff, cs, h))
                        if name != "huber_local":
                            dE = np.diff(E)
                            sm.update(energy_T_over_0=float(E[-1] / E[0]), n_energy_increases=int((dE > 0).sum()),
                                      max_energy_increase_rel=float(max(0.0, dE.max()) / E[0]),
                                      max_abs_balance_resid=float(np.nanmax(np.abs(bal))))
                        summ.append(sm)
                        res[(dyn, h)] = (st, E, sm)
                for dyn in DYNS:
                    (sa, Ea, ma), (sb, Eb, mb) = res[(dyn, 0.5)], res[(dyn, 0.25)]
                    hcmp.append(dict(base, start=sname, force=name, dyn=dyn,
                                     max_abs_dgap_ratio=float(np.max(np.abs(sa["gap"] / sa["gap"][0] - sb["gap"] / sb["gap"][0]))),
                                     max_abs_drms_in_rel=float(np.max(np.abs(sa["rms_in"] - sb["rms_in"])) / max(sb["rms_in"].max(), 1e-300)),
                                     half_gap_t_h05=ma["half_gap_t"], half_gap_t_h025=mb["half_gap_t"],
                                     d_half_gap_t=ma["half_gap_t"] - mb["half_gap_t"],
                                     internal_half_t_h05=ma["internal_half_t"], internal_half_t_h025=mb["internal_half_t"],
                                     n_returns_h05=ma["n_returns_above_half"], n_returns_h025=mb["n_returns_above_half"],
                                     max_abs_denergy_rel=float(np.max(np.abs(Ea - Eb)) / Eb[0]) if name != "huber_local" else float("nan"),
                                     energy_T_over_0_h05=ma.get("energy_T_over_0", float("nan")),
                                     energy_T_over_0_h025=mb.get("energy_T_over_0", float("nan"))))
    # ---- checks (hard unless noted) ----
    def add(name, value, tol, passed, hard=1):
        checks.append(dict(part=1, check=name, value=value, tol=tol, passed=int(bool(passed)), hard=hard))
    add("p1_force_impl_vs_dense", max(impl), TOL_IMPL, max(impl) <= TOL_IMPL)
    hand_ok = [fr["abs_err"] <= TOL_HAND_ABS and fr["rel_err"] <= TOL_HAND_REL for fr in force_rows]
    add("p1_bridge_force_hand", max(fr["rel_err"] for fr in force_rows), f"abs<={TOL_HAND_ABS} and rel<={TOL_HAND_REL}", all(hand_ok))
    add("p1_bridge_force_sign", sum(fr["sign_ok"] for fr in force_rows), f"{len(force_rows)}/{len(force_rows)} endpoints",
        all(fr["sign_ok"] for fr in force_rows))
    flux_err = max(abs(fr["flux_per_bridge_code"] - {"mean": 0.5, "huber": 0.125}[fr["force"]])
                   for fr in force_rows if fr["force"] != "huber_local")
    add("p1_bridge_flux_mean0.5_huber0.125", flux_err, TOL_HAND_ABS, flux_err <= TOL_HAND_ABS)
    off = max(fr["max_abs_F_off_endpoints"] for fr in force_rows)
    add("p1_zero_force_off_endpoints", off, 0.0, off == 0.0)
    for kind in ("wave_undamped", "wave_damped", "diffusion"):
        rs = [r for r in mode_rows if r["check"] == kind]
        v = max(max(r["max_amp_dev"], r["max_offmode"]) for r in rs)
        add(f"p1_linear_mode_{kind}", v, TOL_MODE, v <= TOL_MODE)
    v = max(abs(fr["sum_dF"]) for fr in force_rows if fr["force"] != "huber_local")
    add("p1_mean_huber_sum_dF_block_start", v, TOL_CONSERVE, v <= TOL_CONSERVE, hard=0)
    nfail = sum(s["failed"] for s in summ)
    add("p1_runs_failed", nfail, "reported", True, hard=0)
    return dict(force=force_rows, modes=mode_rows, curves=curves, summary=summ, hcmp=hcmp)


# ---- part 2 helpers ----------------------------------------------------------------------------------------
def frames(t0, t1):
    return int(round(t0 / DT_SAVE)), int(round(t1 / DT_SAVE)) + 1


def kruskal_gap(g, score, truth):
    """min bridge score - max over blocks of the induced-block minimum-spanning-tree bottleneck (the score at
    which stable ascending Kruskal on that block's own edges first connects it).  Also returns both bottlenecks."""
    bm = truth[g.ei] != truth[g.ej]
    bott = []
    for b in (0, 1):
        sel = np.flatnonzero(~bm & (truth[g.ei] == b))
        nodes = np.flatnonzero(truth == b)
        parent = {int(v): int(v) for v in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        comps, top = len(nodes), float("nan")
        for e in sel[np.argsort(score[sel], kind="stable")]:
            a, c = find(int(g.ei[e])), find(int(g.ej[e]))
            if a != c:
                parent[a] = c
                comps -= 1
                if comps == 1:
                    top = float(score[e])
                    break
        assert comps == 1, "induced block not connected"
        bott.append(top)
    return float(score[bm].min() - max(bott)), bott[0], bott[1]


def score_metrics(g, truth, sc, bm):
    """Class summaries, AUC, Kruskal gap and the q=2 cut of one score vector."""
    out = {}
    for cls, sel in (("bridge", bm), ("interior", ~bm)):
        v = sc[sel]
        out[f"{cls}_median"], out[f"{cls}_mean"] = float(np.median(v)), float(v.mean())
        out[f"{cls}_rms"] = float(np.sqrt((v ** 2).mean()))
    out["median_ratio"] = out["bridge_median"] / out["interior_median"] if out["interior_median"] > 0 else float("nan")
    out["interior_below_floor"] = int(out["interior_median"] < INTERIOR_FLOOR)
    out["auc"] = C.bridge_auc(sc, bm)
    gap, b0, b1 = kruskal_gap(g, sc, truth)
    out.update(gap=gap, bottleneck0=b0, bottleneck1=b1, min_bridge=float(sc[bm].min()),
               gap_over_interior_median=gap / out["interior_median"] if out["interior_median"] > 0 else float("nan"),
               gap_exact_tie=int(gap == 0.0), gap_near_tie=int(abs(gap) <= GAP_NEAR_TIE * float(sc.max())),
               n_duplicate_scores=int(len(sc) - len(np.unique(sc))))
    lab = rx.cut_partition(g, sc, 2)
    if lab is None:
        out.update(cut_ok=0, ari_survivor=float("nan"), exact_survivor=0)
    else:
        out.update(cut_ok=1, ari_survivor=float(adjusted_rand_score(truth, lab)),
                   exact_survivor=int(np.array_equal(lab, truth) or np.array_equal(lab, 1 - truth)))
    return out


def stiffness_values(g, name, U):
    """Edge stiffness phi(s)/s on frames U (T, R, n), s = u_ej - u_ei, limit 1 at s = 0.
    Returns (s, k_i, k_j): k_i is the ratio applied in node ei's force, k_j in node ej's (equal unless local)."""
    s = U[..., g.ej] - U[..., g.ei]
    a = np.abs(s)
    if name == "mean":
        k = np.ones_like(s)
        return s, k, k
    if name == "huber":
        k = np.where(a > TAU_FIXED, TAU_FIXED / np.where(a > 0, a, 1.0), 1.0)
        return s, k, k
    tau = local_tau(g, U)
    ti, tj = tau[..., g.ei], tau[..., g.ej]
    safe = np.where(a > 0, a, 1.0)
    return s, np.where(a > ti, ti / safe, 1.0), np.where(a > tj, tj / safe, 1.0)


def qstats(prefix, v, clipped):
    out = {f"{prefix}_n": int(v.size)}
    if v.size:
        qs = np.quantile(v, QUANTS)
        out.update({f"{prefix}_q{int(q * 100):02d}": float(x) for q, x in zip(QUANTS, qs)})
        out[f"{prefix}_mean"] = float(v.mean())
        out[f"{prefix}_clip_frac"] = float(clipped.mean())
    else:
        out.update({f"{prefix}_q{int(q * 100):02d}": float("nan") for q in QUANTS})
        out[f"{prefix}_mean"] = out[f"{prefix}_clip_frac"] = float("nan")
    return out


def stiffness_rows(g, name, Uw, bm, base):
    """Per window: bridge vs interior stiffness distributions over edge x saved time x init and |s|-bin table.
    Measures: 'edge' (Mean / fixed Huber), 'directional' (local: both endpoint ratios pooled) and
    'endpoint_mean' (local: (k_i + k_j)/2, a summary, not a conservative force)."""
    s, ki, kj = stiffness_values(g, name, Uw)
    a = np.abs(s)
    B = np.broadcast_to(bm, s.shape)
    if name == "huber_local":
        meas = {"directional": (np.stack([ki, kj]), np.stack([ki < 1, kj < 1]), np.stack([B, B]), np.stack([a, a])),
                "endpoint_mean": (0.5 * (ki + kj), (ki < 1) | (kj < 1), B, a)}
    else:
        meas = {"edge": (ki, ki < 1, B, a)}
    rows, bins = [], []
    for mname, (k, clip, Bm, aa) in meas.items():
        row = dict(base, measure=mname)
        row.update(qstats("bridge", k[Bm], clip[Bm]))
        row.update(qstats("interior", k[~Bm], clip[~Bm]))
        for st in ["mean", "clip_frac"] + [f"q{int(q * 100):02d}" for q in QUANTS]:
            row[f"contrast_{st}"] = row[f"bridge_{st}"] - row[f"interior_{st}"]
        rows.append(row)
        if name == "mean":
            continue
        for lo, hi in zip(S_BINS[:-1], S_BINS[1:]):
            inb = (aa >= lo) & (aa < hi)
            br, it = k[inb & Bm], k[inb & ~Bm]
            bins.append(dict(base, measure=mname, s_bin=f"[{lo:g},{hi:g})", n_bridge=int(br.size), n_interior=int(it.size),
                             bridge_mean=float(br.mean()) if br.size else float("nan"),
                             interior_mean=float(it.mean()) if it.size else float("nan"),
                             residual_contrast=float(br.mean() - it.mean()) if br.size and it.size else float("nan"),
                             bridge_abs_s_mean=float(aa[inb & Bm].mean()) if br.size else float("nan"),
                             interior_abs_s_mean=float(aa[inb & ~Bm].mean()) if it.size else float("nan")))
    edge_mean_i = ki.mean(axis=(0, 1))                      # per-edge summaries for the NPZ
    edge_mean_j = kj.mean(axis=(0, 1))
    edge_clip_i = (ki < 1).mean(axis=(0, 1))
    edge_clip_j = (kj < 1).mean(axis=(0, 1))
    return rows, bins, np.stack([edge_mean_i, edge_mean_j, edge_clip_i, edge_clip_j])


# ---- part 2 ------------------------------------------------------------------------------------------------
def part2(checks):
    specs = p2_specs()
    graphs, scores, stiff, sbins, fails = [], [], [], [], []
    inits = []
    mmax = max(C.make_graph(s)[2]["m"] for s in specs)
    run_keys, edge_scores, edge_stiff, ckpt = [], [], [], []
    edge_ptr, edge_i, edge_j, edge_w, edge_b = [0], [], [], [], []
    t310_overlap = 0
    ck_idx = [int(round(tt / DT_SAVE)) for tt in CHECKPOINT_T]
    for gi, spec in enumerate(specs):
        g, truth, meta = C.make_graph(spec)
        base_g = gmeta(spec, meta)
        d = C.wdeg(g)
        bm = meta["bridge_mask"]
        u0 = stream(2, spec).uniform(-1.0, 1.0, (R, g.n))
        t310 = C.init("uniform", g, R, np.random.default_rng(spec["seed"] * 1000 + 310))
        t310_overlap += int(np.any(np.isclose(u0, t310, rtol=0, atol=0)))
        inits.append(u0)
        lam = C.spectrum(g)[0]
        edge_ptr.append(edge_ptr[-1] + meta["m"])
        edge_i += list(g.ei)
        edge_j += list(g.ej)
        edge_w += list(g.ew)
        edge_b += list(bm)
        spec_lab = rx.spectral_labels(g)
        graphs.append(dict(base_g, graph_index=gi, n=g.n, m=meta["m"], n_bridge_edges=int(bm.sum()), lam2=float(lam[1]),
                           lam_max=float(lam[-1]), h2_lam_max_h05=0.25 * lam[-1], h_lam_max_h05=0.5 * lam[-1],
                           spectral_ari=float(adjusted_rand_score(truth, spec_lab)),
                           spectral_exact=int(np.array_equal(spec_lab, truth) or np.array_equal(spec_lab, 1 - truth))))
        for name in FORCES:
            Fn = make_force(g, name)
            for dyn in DYNS:
                for h in HS:
                    run = integrate(Fn, u0, h, dyn, d)
                    X = run["X"]
                    assert X.shape == (N_SAVE, R, g.n)
                    ff, cs = run["first_fail"], run["censor"]
                    failed = ff >= 0
                    surv = ~failed
                    key = dict(base_g, force=name, dyn=dyn, h=h)
                    for r in range(R):
                        f = int(ff[r])
                        row = dict(key, init=r, failed=int(f >= 0), first_fail_substep=f,
                                   first_fail_t=f * h if f >= 0 else float("nan"), censored=int(cs[r] >= 0),
                                   censor_substep=int(cs[r]))
                        if f >= 0:
                            j = min(N_SAVE - 1, int(math.ceil(f * h / DT_SAVE)))
                            u = X[j, r]
                            m = float(u @ d / d.sum()) if np.isfinite(u).all() else float("nan")
                            row.update(saved_t_after_fail=j * DT_SAVE, maxabs_at_saved=float(np.nanmax(np.abs(u))),
                                       center_at_saved=m, centered_max_at_saved=float(np.nanmax(np.abs(u - m))))
                        fails.append(row)
                    n_fail, n_surv = int(failed.sum()), int(surv.sum())
                    run_keys.append(dict(run_index=len(run_keys), graph_index=gi, **key, n_fail=n_fail))
                    ckpt.append(X[ck_idx])
                    esc = np.full((len(WINDOWS), len(READOUTS), mmax), np.nan)
                    est = np.full((len(WINDOWS), 4, mmax), np.nan, dtype=np.float32)
                    for wi, (wname, t0, t1) in enumerate(WINDOWS):
                        a, b = frames(t0, t1)
                        Uw = X[a:b][:, surv]
                        wkey = dict(key, window=wname, n_init=R, n_fail=n_fail, n_censored=int((cs >= 0).sum()),
                                    n_scored=n_surv, scored_on="all8" if n_fail == 0 else f"survivors{n_surv}")
                        if n_surv:
                            rws, bns, per_edge = stiffness_rows(g, name, Uw, bm, wkey)
                            stiff += rws
                            sbins += bns
                            est[wi, :, :meta["m"]] = per_edge
                        for ri, ro in enumerate(READOUTS):
                            row = dict(wkey, readout=ro)
                            if n_surv:
                                per = C.edge_score(Uw, g, None, frac=1.0, lowpass_fc=FC if ro == "lp" else None, per_run=True)
                                sc = np.median(per, axis=0)
                                esc[wi, ri, :meta["m"]] = sc
                                ok = np.isfinite(sc).all()
                            else:
                                ok = False
                            if ok:
                                row.update(score_metrics(g, truth, sc, bm))
                            else:
                                row.update(cut_ok=0, ari_survivor=float("nan"), exact_survivor=0)
                            row["unscorable"] = int(not ok or not row["cut_ok"])
                            row["ari_plan"] = 0.0 if n_fail > 0 else row["ari_survivor"]
                            row["exact_plan"] = 0 if n_fail > 0 else row["exact_survivor"]
                            scores.append(row)
                    edge_scores.append(esc)
                    edge_stiff.append(est)
    for gr in graphs:                                   # spectral as a score row for pairing
        scores.append(dict({k: gr[k] for k in ("graph", "family", "sizes", "bridges", "wb", "seed")}, force="spectral",
                           dyn="none", h="none", window="none", readout="none", n_init=1, n_fail=0, n_scored=1,
                           scored_on="deterministic", cut_ok=1, unscorable=0, ari_survivor=gr["spectral_ari"],
                           ari_plan=gr["spectral_ari"], exact_survivor=gr["spectral_exact"], exact_plan=gr["spectral_exact"]))

    def add(name, value, tol, passed, hard=1):
        checks.append(dict(part=2, check=name, value=value, tol=tol, passed=int(bool(passed)), hard=hard))
    ids = [gr["graph"] for gr in graphs]
    add("p2_graph_count_unique", len(set(ids)), "==48", len(set(ids)) == len(ids) == 48)
    add("p2_inits_share_no_value_with_t310", t310_overlap, "==0 graphs", t310_overlap == 0)
    scored = [s for s in scores if s["force"] != "spectral" and "gap" in s]
    pos_bad = sum(1 for s in scored if s["gap"] > 0 and not s["exact_survivor"])
    neg_bad = sum(1 for s in scored if s["gap"] < 0 and s["exact_survivor"])
    add("p2_strict_positive_gap_implies_exact", pos_bad, "==0 violations", pos_bad == 0)
    add("p2_strict_negative_gap_implies_not_exact", neg_bad, "==0 violations", neg_bad == 0)
    add("p2_exact_ties", sum(s["gap_exact_tie"] for s in scored), "reported", True, hard=0)
    add("p2_unscorable_rows", sum(s["unscorable"] for s in scores), "reported", True, hard=0)
    npz_edges = dict(graph_ids=np.array(ids), edge_ptr=np.array(edge_ptr), edge_i=np.array(edge_i), edge_j=np.array(edge_j),
                     edge_w=np.array(edge_w), edge_bridge=np.array(edge_b, bool),
                     run_index=np.array([k["run_index"] for k in run_keys]),
                     run_graph_index=np.array([k["graph_index"] for k in run_keys]),
                     run_force=np.array([k["force"] for k in run_keys]), run_dyn=np.array([k["dyn"] for k in run_keys]),
                     run_h=np.array([k["h"] for k in run_keys]), run_n_fail=np.array([k["n_fail"] for k in run_keys]),
                     windows=np.array([w[0] for w in WINDOWS]), readouts=np.array(READOUTS),
                     scores=np.stack(edge_scores),
                     stiffness_names=np.array(["mean_k_ei_side", "mean_k_ej_side", "clip_frac_ei_side", "clip_frac_ej_side"]),
                     stiffness=np.stack(edge_stiff))
    npz_ckpt = dict(graph_ids=np.array(ids), truth=np.stack([C.make_graph(s)[1] for s in specs]),
                    inits=np.stack(inits), checkpoint_t=np.array(CHECKPOINT_T),
                    run_graph_index=npz_edges["run_graph_index"], run_force=npz_edges["run_force"],
                    run_dyn=npz_edges["run_dyn"], run_h=npz_edges["run_h"], states=np.stack(ckpt),
                    first_fail_substep=np.array([[r["first_fail_substep"] for r in fails[i * R:(i + 1) * R]]
                                                 for i in range(len(run_keys))]))
    return dict(graphs=graphs, scores=scores, stiff=stiff, sbins=sbins, fails=fails, npz_edges=npz_edges, npz_ckpt=npz_ckpt)


# ---- part 2 summaries --------------------------------------------------------------------------------------
CANDIDATES = (("huber", "wave", "candidate"), ("huber_local", "wave", "candidate"), ("mean", "wave", "reference"))


def strata(rows):
    yield ("all", "all"), rows
    for k in ("family", "sizes", "bridges"):
        for v in sorted({r[k] for r in rows}, key=str):
            yield (k, v), [r for r in rows if r[k] == v]


def paired(scores):
    """Graph-paired differences: each wave method vs Mean wave (same h/window/readout), its same-force diffusion
    and spectral.  Metrics: ari_plan (primary), ari_survivor and auc (supplemental, survivor-based)."""
    idx = {(s["graph"], s["force"], s["dyn"], s["h"], s["window"], s["readout"]): s for s in scores if s["force"] != "spectral"}
    spec = {s["graph"]: s for s in scores if s["force"] == "spectral"}
    out = []
    for (gid, force, dyn, h, w, ro), s in idx.items():
        cand = next((c for c in CANDIDATES if c[0] == force and c[1] == dyn), None)
        if cand is None:
            continue
        comps = [("mean_wave", idx.get((gid, "mean", "wave", h, w, ro))), ("same_force_diffusion", idx.get((gid, force, "diffusion", h, w, ro))),
                 ("spectral", spec[gid])]
        for cname, c in comps:
            if cname == "mean_wave" and force == "mean":
                continue
            for metric in ("ari_plan", "ari_survivor", "auc"):
                if cname == "spectral" and metric == "auc":
                    continue
                a, b = s.get(metric, float("nan")), c.get(metric, float("nan"))
                out.append(dict(graph=gid, family=s["family"], sizes=s["sizes"], bridges=s["bridges"], seed=s["seed"],
                                candidate=f"{force}_wave", role=cand[2], comparator=cname, h=h, window=w, readout=ro,
                                metric=metric, cand_value=a, comp_value=b, diff=a - b,
                                cand_failed=int(s["n_fail"] > 0), comp_failed=int(c["n_fail"] > 0)))
    return out


def paired_summary(pairs):
    out = []
    keys = sorted({(p["candidate"], p["role"], p["comparator"], p["h"], p["window"], p["readout"], p["metric"]) for p in pairs}, key=str)
    groups = {}
    for p in pairs:
        groups.setdefault((p["candidate"], p["role"], p["comparator"], p["h"], p["window"], p["readout"], p["metric"]), []).append(p)
    for key in keys:
        for (sk, sv), rs in strata(groups[key]):
            v = np.array([r["diff"] for r in rs], float)
            fin = np.isfinite(v)
            est, lo, hi = C.boot_ci(v[fin]) if fin.any() else (float("nan"),) * 3
            out.append(dict(candidate=key[0], role=key[1], comparator=key[2], h=key[3], window=key[4], readout=key[5],
                            metric=key[6], stratum=sk, level=sv, n_graphs=len(rs), n_pairs=int(fin.sum()),
                            cand_mean=float(np.nanmean([r["cand_value"] for r in rs])) if fin.any() else float("nan"),
                            comp_mean=float(np.nanmean([r["comp_value"] for r in rs])) if fin.any() else float("nan"),
                            mean_diff=est, boot95_lo=lo, boot95_hi=hi,
                            median_diff=float(np.median(v[fin])) if fin.any() else float("nan"),
                            n_pos=int((v[fin] > 0).sum()), n_neg=int((v[fin] < 0).sum()), n_zero=int((v[fin] == 0).sum()),
                            cand_graphs_failed=sum(r["cand_failed"] for r in rs), comp_graphs_failed=sum(r["comp_failed"] for r in rs)))
    # h agreement of the sign of the mean difference
    idx = {(r["candidate"], r["comparator"], r["h"], r["window"], r["readout"], r["metric"], r["stratum"], r["level"]): r for r in out}
    for r in out:
        o = idx.get((r["candidate"], r["comparator"], 0.25 if r["h"] == 0.5 else 0.5, r["window"], r["readout"], r["metric"],
                     r["stratum"], r["level"]))
        r["mean_diff_other_h"] = o["mean_diff"] if o else float("nan")
        r["sign_agrees_across_h"] = int(o is not None and np.sign(o["mean_diff"]) == np.sign(r["mean_diff"]))
    return out


def h_halving(scores):
    idx = {(s["graph"], s["force"], s["dyn"], s["h"], s["window"], s["readout"]): s for s in scores if s["force"] != "spectral"}
    out = []
    for (gid, force, dyn, h, w, ro), a in idx.items():
        if h != 0.5:
            continue
        b = idx[(gid, force, dyn, 0.25, w, ro)]
        g = lambda s, k: s.get(k, float("nan"))  # noqa: E731
        sga, sgb = np.sign(g(a, "gap")), np.sign(g(b, "gap"))
        row = dict(graph=gid, family=a["family"], sizes=a["sizes"], bridges=a["bridges"], seed=a["seed"], force=force, dyn=dyn,
                   window=w, readout=ro, n_fail_h05=a["n_fail"], n_fail_h025=b["n_fail"],
                   ari_plan_h05=a["ari_plan"], ari_plan_h025=b["ari_plan"], d_ari_plan=a["ari_plan"] - b["ari_plan"],
                   auc_h05=g(a, "auc"), auc_h025=g(b, "auc"), d_auc=g(a, "auc") - g(b, "auc"),
                   gap_h05=g(a, "gap"), gap_h025=g(b, "gap"), exact_h05=a["exact_survivor"], exact_h025=b["exact_survivor"],
                   bridge_median_h05=g(a, "bridge_median"), bridge_median_h025=g(b, "bridge_median"),
                   interior_median_h05=g(a, "interior_median"), interior_median_h025=g(b, "interior_median"))
        row["gap_sign_changes"] = int(sga != sgb)
        row["h_sensitive"] = int(a["n_fail"] != b["n_fail"] or a["exact_survivor"] != b["exact_survivor"]
                                 or row["gap_sign_changes"] or (np.isfinite(row["d_ari_plan"]) and abs(row["d_ari_plan"]) > SENS_DARI))
        out.append(row)
    return out


def gate_rows(fails, n_graphs):
    out = []
    keys = sorted({(f["force"], f["dyn"], f["h"]) for f in fails}, key=str)
    for force, dyn, h in keys:
        rs = [f for f in fails if (f["force"], f["dyn"], f["h"]) == (force, dyn, h)]
        for (sk, sv), sub in strata(rs):
            gset = {f["graph"] for f in sub}
            gfail = {f["graph"] for f in sub if f["failed"]}
            out.append(dict(force=force, dyn=dyn, h=h, stratum=sk, level=sv, runs=len(sub), run_failures=sum(f["failed"] for f in sub),
                            run_censored=sum(f["censored"] for f in sub), graphs=len(gset), graphs_any_failure=len(gfail),
                            graph_failure_frac=len(gfail) / len(gset),
                            gate_exceeded=int(sk == "all" and len(gfail) / len(gset) > GATE)))
    return out


def method_summary(scores, stiff):
    out = []
    keys = sorted({(s["force"], s["dyn"], s["h"], s["window"], s["readout"]) for s in scores}, key=str)
    for key in keys:
        rs = [s for s in scores if (s["force"], s["dyn"], s["h"], s["window"], s["readout"]) == key]
        for (sk, sv), sub in strata(rs):
            def col(k):
                return np.array([s.get(k, float("nan")) for s in sub], float)
            gap, auc = col("gap"), col("auc")
            row = dict(force=key[0], dyn=key[1], h=key[2], window=key[3], readout=key[4], stratum=sk, level=sv,
                       n_graphs=len(sub), graphs_failed=sum(s["n_fail"] > 0 for s in sub),
                       unscorable=sum(s["unscorable"] for s in sub),
                       ari_plan_mean=float(np.nanmean(col("ari_plan"))), ari_survivor_mean=float(np.nanmean(col("ari_survivor"))),
                       ari_survivor_n=int(np.isfinite(col("ari_survivor")).sum()),
                       exact_plan=int(np.nansum(col("exact_plan"))), exact_survivor=int(np.nansum(col("exact_survivor"))))
            if key[0] != "spectral":
                fin = np.isfinite(auc)
                row.update(auc_median=float(np.median(auc[fin])) if fin.any() else float("nan"),
                           auc_mean=float(auc[fin].mean()) if fin.any() else float("nan"),
                           n_auc_ge_high=int((auc[fin] >= HIGH_AUC).sum()),
                           n_gap_pos=int((gap > 0).sum()), n_gap_neg=int((gap < 0).sum()), n_gap_zero=int((gap == 0).sum()),
                           n_high_auc_neg_gap=int(((auc >= HIGH_AUC) & (gap < 0)).sum()),
                           bridge_score_median=float(np.nanmedian(col("bridge_median"))),
                           interior_score_median=float(np.nanmedian(col("interior_median"))),
                           n_interior_below_floor=int(np.nansum(col("interior_below_floor"))),
                           score_ratio_median_above_floor=float(np.nanmedian(np.where(col("interior_below_floor") == 1, np.nan,
                                                                                      col("median_ratio"))))
                           if (col("interior_below_floor") == 0).any() else float("nan"))
                st = [x for x in stiff if (x["force"], x["dyn"], x["h"], x["window"]) == key[:4]
                      and x["measure"] in ("edge", "directional") and (sk == "all" or x[sk] == sv)]
                if st:
                    row.update(stiff_bridge_median=float(np.median([x["bridge_q50"] for x in st])),
                               stiff_interior_median=float(np.median([x["interior_q50"] for x in st])),
                               stiff_bridge_mean=float(np.mean([x["bridge_mean"] for x in st])),
                               stiff_interior_mean=float(np.mean([x["interior_mean"] for x in st])),
                               stiff_contrast_mean=float(np.mean([x["contrast_mean"] for x in st])),
                               clip_bridge_mean=float(np.mean([x["bridge_clip_frac"] for x in st])),
                               clip_interior_mean=float(np.mean([x["interior_clip_frac"] for x in st])))
            out.append(row)
    return out


def sbin_summary(sbins):
    """Pooled-over-graphs |s|-bin table (counts summed; means weighted by counts).  Descriptive only."""
    out = []
    keys = sorted({(b["force"], b["dyn"], b["h"], b["window"], b["measure"], b["s_bin"]) for b in sbins}, key=str)
    for key in keys:
        rs = [b for b in sbins if (b["force"], b["dyn"], b["h"], b["window"], b["measure"], b["s_bin"]) == key]
        for (sk, sv), sub in strata(rs):
            nb, ni = sum(b["n_bridge"] for b in sub), sum(b["n_interior"] for b in sub)
            bm = sum(b["bridge_mean"] * b["n_bridge"] for b in sub if b["n_bridge"]) / nb if nb else float("nan")
            im = sum(b["interior_mean"] * b["n_interior"] for b in sub if b["n_interior"]) / ni if ni else float("nan")
            gc = [b["residual_contrast"] for b in sub if np.isfinite(b["residual_contrast"])]
            out.append(dict(force=key[0], dyn=key[1], h=key[2], window=key[3], measure=key[4], s_bin=key[5], stratum=sk,
                            level=sv, graphs=len(sub), graphs_both_classes=len(gc), n_bridge=nb, n_interior=ni,
                            bridge_mean=bm, interior_mean=im, pooled_residual_contrast=bm - im,
                            graph_median_residual_contrast=float(np.median(gc)) if gc else float("nan")))
    return out


# ---- output helpers ----------------------------------------------------------------------------------------
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    if str(path).endswith(".gz"):   # mtime=0 keeps reruns byte-identical
        f = io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0), newline="")
    else:
        f = open(path, "w", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()})


def write_npz(path, arrays):
    """Compressed NPZ with fixed member timestamps so reruns are byte-identical."""
    import zipfile
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(v), allow_pickle=False)
            info = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buf.getvalue())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: the script's directory)")
    ap.add_argument("--part", choices=("1", "all"), default="all", help="'1' runs only part 1 and its checks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert wt.FAIL_AMP == AMP and wt.EPS == EPS
    assert C.resolve_params(wt.FORCES, "huber") == {"tau": TAU_FIXED}
    assert C.resolve_params(wt.FORCES, "huber_local") == {"kappa": KAPPA}
    checks = []
    p1 = part1(checks)
    write_csv(args.out / "part1_bridge_force.csv", p1["force"])
    write_csv(args.out / "part1_linear_modes.csv", p1["modes"])
    write_csv(args.out / "part1_block_curves.csv.gz", p1["curves"])
    write_csv(args.out / "part1_block_summary.csv", p1["summary"])
    write_csv(args.out / "part1_h_compare.csv", p1["hcmp"])
    env = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__,
               sklearn=__import__("sklearn").__version__)
    hard1 = [c["check"] for c in checks if c["hard"] and not c["passed"]]
    if hard1:
        write_csv(args.out / "checks.csv", checks)
        (args.out / "BLOCKER.md").write_text(
            "# 003 BLOCKER: part 1 reference check(s) failed; part 2 was not started\n\n"
            + "\n".join(f"- `{c['check']}`: value {c['value']} vs tol {c['tol']}" for c in checks
                        if c["hard"] and not c["passed"]) + "\n")
        (args.out / "summary.json").write_text(json.dumps(dict(environment=env, checks=checks, hard_failures=hard1,
                                                               part2="not started"), indent=2) + "\n")
        print("BLOCKER:", hard1)
        sys.exit(1)
    if args.part == "1":
        write_csv(args.out / "checks.csv", checks)
        (args.out / "summary.json").write_text(json.dumps(dict(environment=env, checks=checks, hard_failures=[],
                                                               part2="not run (--part 1)"), indent=2) + "\n")
        print(json.dumps(checks, indent=1))
        return
    p2 = part2(checks)
    pairs = paired(p2["scores"])
    write_csv(args.out / "part2_graphs.csv", p2["graphs"])
    write_csv(args.out / "part2_scores.csv", p2["scores"])
    write_csv(args.out / "part2_stiffness.csv", p2["stiff"])
    write_csv(args.out / "part2_sbins.csv", p2["sbins"])
    write_csv(args.out / "part2_sbins_summary.csv", sbin_summary(p2["sbins"]))
    write_csv(args.out / "part2_init_failures.csv", p2["fails"])
    write_csv(args.out / "part2_gate.csv", gate_rows(p2["fails"], len(p2["graphs"])))
    write_csv(args.out / "part2_paired_graphs.csv", pairs)
    write_csv(args.out / "part2_paired_summary.csv", paired_summary(pairs))
    write_csv(args.out / "part2_h_halving.csv", h_halving(p2["scores"]))
    write_csv(args.out / "part2_method_summary.csv", method_summary(p2["scores"], p2["stiff"]))
    write_npz(args.out / "part2_edges.npz", p2["npz_edges"])
    write_npz(args.out / "part2_checkpoints.npz", p2["npz_ckpt"])
    write_csv(args.out / "checks.csv", checks)
    gates = [r for r in gate_rows(p2["fails"], 48) if r["stratum"] == "all"]
    summary = dict(environment=env, checks=checks, hard_failures=[c["check"] for c in checks if c["hard"] and not c["passed"]],
                   gate={f"{r['force']}_{r['dyn']}_h{r['h']}": dict(run_failures=r["run_failures"], graphs_any_failure=r["graphs_any_failure"],
                                                                  gate_exceeded=r["gate_exceeded"]) for r in gates})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["hard_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
