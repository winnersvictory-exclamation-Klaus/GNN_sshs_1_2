"""Self-tests for battery/common.py.   Run:  python battery/test_common.py [-k substring]
Each test_* function is plain asserts (also collectable by pytest).  Exit code 1 if anything fails."""
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

wt, rx = C.wt, C.rx
warnings.filterwarnings("ignore")


def _g(block="clique", bridges=2, seed=0, sizes=(24, 24), wb=1.0):
    return C.make_graph(dict(kind="two_block", block=block, sizes=sizes, bridges=bridges, wb=wb, seed=seed))


def _u0(g, R=8, seed=1, kind="uniform", **kw):
    return C.init(kind, g, R, np.random.default_rng(seed), **kw)


def _pid(_):
    return os.getpid()


def _sq(x):
    return x * x


# ---------------------------------------------------------------------------------------------- spec tests
def test_mean_force_equals_minus_Lrw():
    for blk in ("cycle", "clique", "sbm"):
        g, t, m = _g(blk, 2)
        u = _u0(g, 5)
        F = C.eval_force(g, "mean", u)
        assert np.allclose(F, -(u @ C.L_rw(g).T), atol=1e-12), blk
    g, t, m = C.make_graph(dict(kind="two_block", block="cycle", bridges=2, wb=0.25))    # weighted edges
    u = _u0(g, 5)
    assert np.allclose(C.eval_force(g, "mean", u), -(u @ C.L_rw(g).T), atol=1e-12)


def test_energy_conserved_wave_mean():
    for blk, c in (("cycle", 0.5), ("sbm", 0.5), ("clique", 0.5), ("sbm", 1.3)):
        g, t, m = _g(blk, 2, wb=0.25)
        r = C.run(g, "wave", "mean", _u0(g), c, 500)
        E = C.energy(r.X, g, c, r.fail_step)
        rel = np.abs(E / E[0] - 1).max()
        assert rel < 1e-6, (blk, c, rel)
    # the naive form is NOT conserved (documents why 'staggered' is the default)
    En = C.energy(r.X, g, 1.3, r.fail_step, kind="naive")
    assert np.abs(En / En[0] - 1).max() > 1e-3


def test_disconnected_gives_ari_1():
    for blk in ("cycle", "clique", "sbm"):
        g, t, m = C.make_graph(dict(kind="two_block", block=blk, bridges=0, seed=0))
        assert not m["connected"]
        r = C.run_diff(g, "mean", _u0(g), 0.5, 500)
        labs = C.partitions(r.X, g, 2, r.fail_step, which=("edge_cut", "edge_cut_lp"))
        for nm, lab in labs.items():
            assert lab is not None and C.adjusted_rand_score(t, lab) == 1.0, (blk, nm)
        met = C.partition_metrics(g, labs["edge_cut"], t)
        assert met["ari"] == 1.0 and met["cond"] == 0.0


def test_zoo_graphs_connected_and_valid():
    for name in ("core", "core_full", "wide", "neg", "scale"):
        seeds = C.DEV_SEEDS if name in ("core", "wide", "neg") else (0, 1)
        if name == "core_full":
            seeds = (0, 100)
        for case in C.build_zoo(name, seeds):
            m = case.meta
            assert m["connected"], m["id"]
            assert (case.g.deg >= 1).all()
            if name == "neg":
                assert case.truth is None and m["bridge_mask"] is None
            else:
                assert case.truth is not None and len(case.truth) == case.g.n
                assert m["bridge_mask"].shape == (m["m"],) and m["n_bridges"] >= 1, m["id"]
                assert len(np.unique(case.truth)) == m["q"]
    for case in C.build_zoo("disc", (0, 1)):
        assert not case.meta["connected"] and case.meta["n_bridges"] == 0
    sizes = {c.meta["n"] for c in C.build_zoo("scale", (0,))}
    assert sizes == {48, 96, 192, 384}, sizes
    # held-out seeds build too
    assert len(C.build_zoo("core", C.HELD_SEEDS[:4])) == 48


def test_random_regular_and_degrees():
    for d in (3, 4, 6, 8):
        g, t, m = C.make_graph(dict(kind="random_regular", n=48, d=d, seed=3))
        assert (g.deg == d).all() and m["connected"]
    g, t, m = C.make_graph(dict(kind="watts_strogatz", n=48, k=4, p=0.3, seed=1))
    assert m["connected"] and g.deg.min() >= 1


def test_make_graph_matches_rx_prepare():
    for kind in ("cycle", "clique", "sbm", "er", "shuffled"):
        spec = dict(kind=kind, sizes=(24, 24), bridges=0 if kind in ("er", "shuffled") else 2, wb=1.0, seed=1, role="x")
        g0, t0 = rx.prepare(spec)
        g, t, m = C.make_graph(dict(kind=kind, sizes=(24, 24), bridges=spec["bridges"], wb=1.0, seed=1)
                               if kind not in ("er", "shuffled") else dict(kind=kind, sizes=(24, 24), seed=1))
        assert np.array_equal(g.A, g0.A), kind
        assert (t is None and t0 is None) or np.array_equal(t, t0)
    # caching returns the same object; different spec formulations normalise to the same graph
    a = C.make_graph(dict(kind="cycle", seed=2))[0]
    b = C.make_graph(dict(kind="two_block", block="cycle", sizes=(24, 24), bridges=2, wb=1.0, seed=2))[0]
    assert a is b
    try:
        C.make_graph(dict(kind="cycle", bridgez=2))
        raise AssertionError("typo accepted")
    except KeyError:
        pass


def test_planted_structure_truth_and_bridges():
    g, t, m = C.make_graph(dict(kind="barbell_path", k=10, L=5))
    assert m["n_bridges"] == 1 and g.n == 25 and np.bincount(t).tolist() == [13, 12]
    g, t, m = C.make_graph(dict(kind="grid_neck", a=5, w=3))
    assert m["n_bridges"] == 3 and g.n == 50
    g, t, m = C.make_graph(dict(kind="ring_of_cliques", q=5, s=6))
    assert m["n_bridges"] == 5 and m["q"] == 5
    g, t, m = C.make_graph(dict(kind="binary_tree", depth=4))
    assert g.n == 31 and m["n_bridges"] == 1 and np.bincount(t).tolist() == [16, 15]
    g, t, m = C.make_graph(dict(kind="lollipop", k=8, L=6))
    assert m["n_bridges"] == 1
    g, t, m = C.make_graph(dict(kind="two_block", block="sbm", sizes=(8, 40), bridges=8, wb=0.25))
    assert m["n_bridges"] == 8 and np.allclose(g.ew[m["bridge_idx"]], 0.25)
    g, t, m = C.make_graph(dict(kind="sbm_q", q=4, size=12))
    assert m["q"] == 4 and m["n_bridges"] > 0
    for sg in (0.0, 0.3, 0.6, 1.0):
        g, t, m = C.make_graph(dict(kind="sbm_het", sigma=sg, seed=2))
        assert m["connected"]


def test_init_kinds():
    g, t, m = _g("sbm", 2)
    r = lambda s=0: np.random.default_rng(s)  # noqa: E731
    # uniform reproduces the rx/wt recipe exactly
    assert np.array_equal(C.init("uniform", g, 8, r(7)), r(7).uniform(-1, 1, (8, g.n)))
    for kind in ("gauss", "rademacher", "student_t3"):
        u = C.init(kind, g, 400, r(1))
        assert u.shape == (400, g.n)
        assert abs(np.sqrt((u ** 2).mean()) / C.RMS_STD - 1) < 0.25, kind     # matched RMS (heavy tail: loose)
    u = C.init("rademacher", g, 4, r(0))
    assert set(np.unique(np.abs(u))) == {C.RMS_STD}
    u = C.init("impulse", g, 6, r(0), node=3, amp=2.0)
    assert (u[:, 3] == 2).all() and (np.abs(u).sum(1) == 2).all()
    u = C.init("uniform", g, 4, r(0), scale=10, dc=3)
    assert np.allclose(u, C.init("uniform", g, 4, r(0)) * 10 + 3)
    # low_mode aligned with the Fiedler vector in the D inner product, high_mode with the top one
    lam, Phi, d = C.spectrum(g)
    u = C.init("low_mode", g, 6, r(0))
    cs = np.abs((u * d) @ Phi) / np.sqrt(((u ** 2) * d).sum(1))[:, None] / np.sqrt(d.sum() / g.n) ** 0
    a = ((u * d) @ Phi)
    frac = a[:, 1] ** 2 / (a[:, 1:] ** 2).sum(1)
    assert (frac > 0.8).all(), frac
    u = C.init("high_mode", g, 6, r(0))
    a = ((u * d) @ Phi)
    assert (a[:, -1] ** 2 / (a[:, 1:] ** 2).sum(1) > 0.8).all()
    # block offset: block means differ by ~delta
    u = C.init("block_offset", g, 300, r(0), truth=t, delta=2.0)
    assert abs((u[:, t == 1].mean() - u[:, t == 0].mean()) - 2.0) < 0.1
    u0 = C.init("block_offset", g, 5, r(0), truth=t, delta=0.0)
    assert np.array_equal(u0, C.init("uniform", g, 5, r(0)))


def test_spectrum_D_orthonormal():
    for blk in ("cycle", "sbm"):
        g, t, m = _g(blk, 2, wb=0.25)
        lam, Phi, d = C.spectrum(g)
        assert abs(lam[0]) < 1e-12 and lam[1] > 1e-6 and lam[-1] <= 2 + 1e-9
        assert np.allclose(Phi.T @ (d[:, None] * Phi), np.eye(g.n), atol=1e-10)
        assert np.allclose(C.L_rw(g) @ Phi, Phi * lam, atol=1e-10)


def test_run_matches_rx_and_diff_matches_rx():
    g, t, m = _g("cycle", 2)
    u0 = _u0(g)
    r = C.run(g, "wave", "mean", u0, 1.0, 500)
    X, failed = rx.simulate(g, "mean", "wave", u0)
    assert np.array_equal(r.X, X) and r.fail.tolist() == failed.tolist()
    for agg in ("mean", "trim", "owa", "vw"):
        r = C.run_diff(g, agg, u0, 0.5, 500)
        X, failed = rx.simulate(g, agg, "diff", u0)
        assert np.array_equal(r.X, X), agg
        assert (r.fail == failed).all()
        r = C.run(g, "wave", agg, u0, 1.0, 500)
        X, failed = rx.simulate(g, agg, "wave", u0)
        assert np.array_equal(r.X, X) and (r.fail == failed).all(), agg
    # partial overrides, typo guard, fail_amp
    r = C.run(g, "damp", "huber", u0, 0.5, 20, ps=dict(k=0.3), pf=dict(tau=0.2))
    assert r.ps == {"k": 0.3} and r.pf == {"tau": 0.2}
    for bad in (dict(ps=dict(kk=1)), dict(pf=dict(tau=1, zzz=2))):
        try:
            C.run(g, "damp", "huber", u0, 0.5, 20, **bad)
            raise AssertionError("typo accepted")
        except KeyError:
            pass
    r1 = C.run(g, "wave", "owa", u0, 0.5, 200)
    r2 = C.run(g, "wave", "owa", u0, 0.5, 200, fail_amp=1e3)
    assert (r2.fail_step >= r1.fail_step).all() and r2.fail_step.max() > r1.fail_step.max()
    assert wt.FAIL_AMP == 20.0                                             # restored


def test_loop_equals_wt_simulate_everywhere():
    """The generic integrator (used by run_diff, hooks, early exit) replicates wt.simulate bit for bit,
    including state-carrying forces/schemes and failed realisations."""
    g, t, m = _g("sbm", 2)
    u0 = _u0(g, 4)
    forces = ["mean", "owa", "trim", "edge_memory", "huber_global", "amp_gated", "hi_geo:owa", "quad_l:vw",
              "symowa_ext", "huber_anneal"]
    for sname in wt.SCHEMES:
        for f in forces:
            ps = C.resolve_params(wt.SCHEMES, sname)
            pf = C.resolve_params(wt.FORCES, f)
            X1, fs1 = wt.simulate(g, sname, f, ps, pf, u0, 0.7, 120)
            X2, fs2 = C._loop(g, sname, f, u0, 0.7, 120, ps, pf, None)
            assert np.array_equal(X1, X2, equal_nan=True) and np.array_equal(fs1, fs2), (sname, f)
    # early exit leaves the same fail_steps
    ps, pf = {}, C.resolve_params(wt.FORCES, "owa")
    _, fs1 = wt.simulate(g, "wave", "owa", C.resolve_params(wt.SCHEMES, "wave"), pf, u0, 1.0, 300)
    _, fs3 = C._loop(g, "wave", "owa", u0, 1.0, 300, C.resolve_params(wt.SCHEMES, "wave"), pf, None,
                     early_stop=True, record=False)
    assert np.array_equal(fs1, fs3)


def test_force_batching_is_rowwise_for_all_forces():
    """Jacobian probes batch 2n perturbed states into one force call: valid only if every force treats the R
    axis independently.  Check all registered forces."""
    g, t, m = _g("sbm", 2, sizes=(10, 14))
    U = _u0(g, 7, seed=3)
    for f in wt.FORCES:
        full = C.eval_force(g, f, U)
        rows = np.concatenate([C.eval_force(g, f, U[i]) for i in range(len(U))])
        assert np.allclose(full, rows, atol=1e-11, equal_nan=True), f


def test_probes_zero_defects_for_mean():
    for blk in ("cycle", "sbm"):
        g, t, m = _g(blk, 2, wb=0.25)
        for u in (_u0(g, 3)[0], _u0(g, 3, kind="gauss", seed=5)[1]):
            p = C.force_props(g, "mean", u)
            for k in ("oddness", "shift_abs", "homog_abs", "const_defect", "momentum_rel", "dsym_rel", "jac_kink"):
                assert p[k] < 1e-7, (blk, k, p[k])
            assert abs(p["eig_sym_max"]) < 1e-8                           # no anti-restoring direction
            lam, Phi, d = C.spectrum(g)
            assert abs(p["lam_max"] - lam[-1]) < 1e-6
            assert abs(p["c_crit_pred"] - 2 / np.sqrt(lam[-1])) < 1e-5
            assert abs(p["c_crit_eig"] - 2 / np.sqrt(lam[-1])) < 1e-4
    g, t, m = _g("sbm", 2)
    J = C.jacobian(g, "mean", _u0(g, 1)[0])
    assert np.allclose(J, -C.L_rw(g), atol=1e-7)


def test_probes_detect_known_nonlinearities():
    g, t, m = _g("sbm", 2)
    u = _u0(g, 1, kind="gauss", seed=2)[0]
    # form A forces (huber, tanh, amp): odd, D-symmetric Jacobian, zero momentum, shift invariant, NOT homogeneous
    for f in ("huber", "tanh", "amp"):
        p = C.force_props(g, f, u)
        assert p["oddness"] < 1e-9 and p["dsym_rel"] < 1e-6 and p["momentum_rel"] < 1e-9, (f, p)
        assert p["shift_abs"] < 1e-9 and p["homog_rel"] > 1e-3, f
    # rank forces: shift invariant but not odd / not D-symmetric / momentum != 0
    p = C.force_props(g, "owa", u)
    assert p["shift_abs"] < 1e-9 and p["oddness"] > 1e-2 and p["dsym_rel"] > 1e-2 and p["momentum_rel"] > 1e-3
    p = C.force_props(g, "odd_owa", u)
    assert p["oddness"] < 1e-9
    # absvw is odd; huber homogeneous when tau huge (limit consistency)
    assert C.force_props(g, "absvw", u)["oddness"] < 1e-9
    assert C.force_props(g, "huber", u, pf=dict(tau=1e6))["homog_rel"] < 1e-9
    # kink diagnostic: ~0 for smooth points, O(1) for tied states on rank forces / exactly saturated clips
    ties = np.where(np.arange(g.n) % 2 == 0, 0.5, -0.5)
    for f in ("trim", "median", "owa"):
        assert C.force_props(g, f, ties)["jac_kink"] > 0.05, f
    assert C.force_props(g, "mean", ties)["jac_kink"] < 1e-6 and C.force_props(g, "tanh", u)["jac_kink"] < 1e-4
    assert C.force_props(g, "huber", np.where(np.arange(g.n) % 2 == 0, 0.25, -0.25))["jac_kink"] > 0.05   # |x| = tau


def test_modal_roots_match_simulation():
    """Per-mode characteristic polynomial (T1.01 formula) vs the actual damp / viscous / damp_visc updates."""
    g, t, m = _g("sbm", 2, wb=0.25)
    lam, Phi, d = C.spectrum(g)
    c = 0.8
    for sname, ps, kk, nu in (("damp", dict(k=0.1), 0.1, 0.0), ("viscous", dict(nu=0.2), 0.0, 0.2),
                              ("damp_visc", dict(k=0.05, nu=0.1), 0.05, 0.1), ("wave", {}, 0.0, 0.0)):
        for mode in (1, 7, g.n - 1):
            u0 = (Phi[:, mode] * 0.3)[None, :]
            r = C.run(g, sname, "mean", u0, c, 60, ps=ps)
            a = (r.X[:, 0, :] * d) @ Phi[:, mode]
            l = lam[mode]
            for tt in range(1, 58):
                lhs = a[tt + 1] - (2 - kk - (c * c + nu) * l) * a[tt] + (1 - kk - nu * l) * a[tt - 1]
                assert abs(lhs) < 1e-11, (sname, mode, tt, lhs)
            z = C.modal_roots(-l, c, kk, nu)
            assert np.allclose(z ** 2 - (2 - kk - (c * c + nu) * l) * z + (1 - kk - nu * l), 0, atol=1e-12)
    # wave mean: |z| = 1 below the threshold, > 1 above
    assert abs(C.modal_rho(-1.5, 1.0) - 1) < 1e-12 and C.modal_rho(-1.5, 1.7) > 1.01
    assert abs(C.c_crit_linear(np.array([-2.0, -1.0, 0.0])) - np.sqrt(2)) < 1e-6
    # Jury: damp (k) is stable iff 0 < c^2 lambda < 4 - 2k, so damping LOWERS the top-mode limit (it does not raise it)
    for kk in (0.1, 0.3):
        assert abs(C.c_crit_linear(np.array([-2.0]), k=kk) - np.sqrt((4 - 2 * kk) / 2)) < 1e-6


def test_empirical_c_crit_matches_theory_for_mean():
    for blk in ("cycle", "sbm"):
        g, t, m = _g(blk, 2)
        lam = C.spectrum(g)[0]
        pred = 2 / np.sqrt(lam[-1])
        e = C.c_crit_empirical(g, "wave", "mean", steps=1500, c_lo=0.3, c_hi=3.0, n_scan=12, rtol=0.005)
        assert e["status"] == "ok" and abs(e["c_crit"] / pred - 1) < 0.02, (blk, e["c_crit"], pred)
        assert not e["non_monotone"]
    e = C.c_crit_empirical(g, "damp", "mean", ps=dict(k=0.3), steps=600, c_lo=0.3, c_hi=1.2, n_scan=4)
    assert e["status"] == "stable_to_hi" or e["c_crit"] > 1.0
    e = C.c_crit_empirical(g, "wave", "mean", steps=300, c_lo=1.8, c_hi=3.0, n_scan=3)
    assert e["status"] == "unstable_at_lo" and np.isnan(e["c_crit"])


# ---------------------------------------------------------------------------------------------- metrics tests
def test_failed_realisations_are_masked():
    g, t, m = _g("cycle", 2)
    r = C.run(g, "wave", "owa", _u0(g), 1.0, 300)
    assert r.fail.all() and r.fail_step.max() < 20
    for i in range(r.R):
        s = r.fail_step[i]
        assert (r.X[s:, i] == 0).all() and np.abs(r.X[s - 1, i]).max() < 20
    tm = C.traj_metrics(r)
    assert np.isnan(tm["ac_final"]).all() and not tm["dead"].any() and np.isnan(tm["rms_final"]).all()
    gr = C.growth_rate(r.X, r.fail_step)
    assert np.isfinite(gr).all() and (gr > 0.1).all()                          # blow-up within ~10 steps: steep
    E = C.energy(r.X, g, 1.0, r.fail_step)
    for i in range(r.R):
        assert np.isnan(E[r.fail_step[i] - 1:, i]).all() and np.isfinite(E[:r.fail_step[i] - 1, i]).all()
    me = C.mode_energy(r.X, g, r.fail_step)
    assert np.isnan(me["low"][r.fail_step[0]:, 0]).all() and np.isfinite(me["low"][:r.fail_step[0], 0]).all()
    # edge score / partitions skip failed runs; all failed -> None
    assert C.edge_score(r.X, g, r.fail_step) is None
    assert all(v is None for v in C.partitions(r.X, g, 2, r.fail_step).values())
    h = C.hold(r.X, g, t, fail_step=r.fail_step)
    assert np.isnan(h["hold"]).all()
    # mixed: some fail, some not -> only ok ones used
    r = C.run(g, "wave", "trim", _u0(g), 0.5, 500)
    assert 0 < r.fail.sum() < r.R
    gr = C.growth_rate(r.X, r.fail_step)
    assert np.isfinite(gr[r.fail]).all() and (gr[r.fail] > 0).all()                   # pre-blow-up slope of failing runs
    sc = C.edge_score(r.X, g, r.fail_step)
    sc2 = C.edge_score(r.X[:, r.ok], g)
    assert np.allclose(sc, sc2)
    s = C.traj_summary(r)
    assert abs(s["fail_frac"] - r.fail.mean()) < 1e-12 and s["n_ok"] == r.ok.sum()


def test_fail_modes_consistent_with_fail_step():
    g, t, m = _g("cycle", 2)
    u0 = _u0(g)
    for f, c in (("owa", 0.5), ("trim", 0.5), ("vw", 0.5), ("mean", 1.6)):
        r = C.run(g, "wave", f, u0, c, 300)
        fm = C.fail_modes(g, "wave", f, u0, c, 300)
        assert np.array_equal(fm["raw_step"], r.fail_step), f
        assert fm["raw_fail"].tolist() == r.fail.tolist(), f
        assert (fm["dc_only"] == (fm["raw_fail"] & ~fm["ac_fail"])).all()
    # the mean force at c=1.6 blows up through an oscillating (AC) mode, never through drift
    assert fm["ac_fail"].all() and not fm["dc_only"].any()


def test_growth_rate_dc_drift_synthetic():
    g, t, m = _g("clique", 2)
    T, R = 400, 3
    tt = np.arange(T + 1)[:, None, None]
    pat = np.random.default_rng(0).normal(size=(1, R, g.n))
    X = pat * np.exp(0.01 * tt)
    assert np.allclose(C.growth_rate(X), 0.01, atol=1e-6)
    X = pat * np.exp(-0.02 * tt)
    assert np.allclose(C.growth_rate(X), -0.02, atol=1e-6)
    X = pat * 0.0 + 1e-300                                                  # decayed to nothing: NaN not garbage
    assert np.isnan(C.growth_rate(X)).all()
    # dc drift: a pure constant offset ramp
    X = np.zeros((T + 1, R, g.n)) + (np.arange(T + 1) * 0.01)[:, None, None]
    d = C.dc_drift(X, g, W=10)
    assert np.allclose(d["plain"], 0.01 * (T - 4.5)) and np.allclose(d["dw"], d["plain"])
    # mean force conserves the degree-weighted mean, not the plain one
    g, t, m = _g("sbm", 2)
    r = C.run(g, "wave", "mean", _u0(g) + 3.0, 0.5, 500)
    dd = C.dc_drift(r.X, g, r.fail_step)
    assert np.abs(dd["dw"]).max() < 1e-9 and np.abs(dd["plain"]).max() > 1e-4


def test_ac_final_dead_and_traj_metrics():
    g, t, m = _g("sbm", 2)
    r = C.run(g, "damp", "mean", _u0(g), 0.5, 500, ps=dict(k=0.2))
    tm = C.traj_metrics(r)
    assert tm["dead"].all() and (tm["ac_final"] < 1e-3).all()
    r = C.run(g, "wave", "mean", _u0(g), 0.5, 500)
    tm = C.traj_metrics(r)
    assert not tm["dead"].any() and (np.abs(tm["growth_rate"]) < 1e-3).all()
    assert abs(np.nanmedian(tm["ac_peak"]) - 1) < 0.5


def test_mode_energy_parseval_and_low_init():
    g, t, m = _g("sbm", 2, wb=0.25)
    lam, Phi, d = C.spectrum(g)
    r = C.run(g, "wave", "mean", _u0(g, kind="low_mode"), 0.5, 100)
    me = C.mode_energy(r.X, g, r.fail_step, full=True)
    a = me["a"]
    tot = (a[..., 1:] ** 2).sum(-1)
    assert np.allclose(me["total"], tot)
    assert np.allclose((a ** 2).sum(-1), (r.X ** 2 * d).sum(-1), rtol=1e-9)           # Parseval in D-norm
    assert (me["low"][0] > 0.8).all() and (me["high"][0] < 0.05).all()
    # high-mode init dominated by the top modes; wave mean keeps modal energies constant
    r = C.run(g, "wave", "mean", _u0(g, kind="high_mode"), 0.5, 100)
    me = C.mode_energy(r.X, g, r.fail_step)
    assert (me["high"][0] > 0.8).all()


def test_temporal_spectrum_synthetic():
    g, t, m = _g("clique", 2)
    T, R = 256, 2
    s = np.random.default_rng(0).normal(size=g.n)
    tt = np.arange(T + 1)
    X = np.tile((((-1.0) ** tt)[:, None, None] * s[None, None, :]), (1, R, 1))
    ts = C.temporal_spectrum(X)
    assert (ts["nyquist"] > 0.95).all() and (ts["centroid"] > 0.45).all()
    X = np.cos(2 * np.pi * 0.1 * tt)[:, None, None] * s[None, None, :] * np.ones((1, R, 1))
    ts = C.temporal_spectrum(X)
    assert np.allclose(ts["centroid"], 0.1, atol=0.01) and (ts["nyquist"] < 1e-3).all()
    r = C.run(g, "wave", "mean", _u0(g), 0.3, 500)                                  # slow wave: low centroid
    assert (C.temporal_spectrum(r.X, r.fail_step)["centroid"] < 0.1).all()


def test_lyapunov_and_activation():
    g, t, m = _g("sbm", 2)
    u0 = _u0(g, 4)
    ly = C.lyapunov(g, "wave", "mean", u0, 0.5, 400)
    assert np.all(np.abs(ly["rate"]) < 0.01), ly["rate"]            # linear neutral wave: no exponential divergence
    ly = C.lyapunov(g, "wave", "mean", u0, 1.8, 400)                   # exponentially unstable linear: fails -> NaN
    assert np.isnan(ly["rate"]).all()
    a = C.activation(g, "wave", "mean", u0, 0.5, 100)
    assert np.nanmax(a["frac"]) == 0.0
    a = C.activation(g, "limiter", "mean", u0, 0.5, 100, ps=dict(vmax=1e6))
    assert np.nanmax(a["frac"]) == 0.0
    a = C.activation(g, "limiter", "mean", u0, 0.5, 100, ps=dict(vmax=1e-3))
    assert (a["frac"][1:20] > 0.9).all() and (a["mean_frac"] > 0.3).all()     # tiny vmax: limiter always bites until consensus
    a = C.activation(g, "vel_shrink", "mean", u0, 0.5, 100, ps=dict(alpha=50.0))
    assert (a["mean_frac"] > 0.3).all()


# ---------------------------------------------------------------------------------------------- partition tests
def test_edge_score_and_readouts_match_rx():
    g, t, m = _g("clique", 2)
    r = C.run_diff(g, "mean", _u0(g), 0.5, 500)
    assert np.allclose(C.edge_score(r.X, g), np.median(rx.edge_scores(r.X, g), 0))
    lab = C.partitions(r.X, g, 2, which=("edge_cut",))["edge_cut"]
    assert np.array_equal(lab, rx.cut_partition(g, np.median(rx.edge_scores(r.X, g), 0), 2))
    lp = C.partitions(r.X, g, 2, which=("edge_cut_lp",))["edge_cut_lp"]
    assert np.array_equal(lp, rx.cut_partition(g, np.median(rx.edge_scores(rx.lowpass(r.X, 0.02), g), 0), 2))
    # fft_lowmode with one realisation == rx.wave_fft_labels
    for s in range(4):
        gg, tt, mm = _g("sbm", 2, seed=s)
        u0 = np.random.default_rng(5).uniform(-1, 1, (1, gg.n))
        rr = C.run(gg, "wave", "mean", u0, 1.0, 500)
        mine = C.partitions(rr.X, gg, 2, which=("fft_lowmode",))["fft_lowmode"]
        ref = rx.wave_fft_labels(gg, np.random.default_rng(5))
        assert np.array_equal(mine, ref), s
    # rx.edge_scores window bug documented: with steps != 500 rx uses X[250:], ours uses the last half
    r2 = C.run_diff(g, "mean", _u0(g), 0.5, 200)
    assert rx.edge_scores(r2.X, g).shape[0] == r2.R
    h = r2.X.shape[0] // 2
    ours = C.edge_score(r2.X, g, per_run=True)
    manual = np.sqrt(((r2.X[h:, :, g.ei] - r2.X[h:, :, g.ej]) ** 2).mean(0))
    assert np.allclose(ours, manual)


def test_all_readouts_recover_clean_structure():
    for blk in ("clique", "sbm"):
        g, t, m = _g(blk, 1, seed=1)
        r = C.run_diff(g, "mean", _u0(g), 0.5, 500)
        row = C.evaluate_partitions(r.X, g, t, r.fail_step)
        for nm in C.PARTITION_NAMES:
            assert row[f"ari_{nm}"] > 0.9, (blk, nm, row[f"ari_{nm}"])
        assert row["bridge_auc"] == 1.0
    # q > 2 with a block structure
    g, t, m = C.make_graph(dict(kind="ring_of_cliques", q=4, s=12))
    r = C.run_diff(g, "mean", _u0(g), 0.5, 200)
    labs = C.partitions(r.X, g, 4, which=("edge_cut", "timeavg_2means", "traj_corr_spectral"))
    assert C.adjusted_rand_score(t, labs["edge_cut"]) == 1.0
    assert C.adjusted_rand_score(t, labs["timeavg_2means"]) > 0.9
    # structureless graph: no readout should be confidently right (just must not crash and ARI undefined)
    g, t, m = C.make_graph(dict(kind="er", n=48))
    r = C.run_diff(g, "mean", _u0(g), 0.5, 200)
    row = C.evaluate_partitions(r.X, g, None, r.fail_step)
    assert np.isnan(row["ari_edge_cut"]) and np.isnan(row["edge_margin"])


def test_edge_margin_and_bridge_auc_synthetic():
    g, t, m = _g("clique", 4)
    bm = m["bridge_mask"]
    sc = np.where(bm, 5.0, 1.0)
    assert C.edge_margin(sc, g, t) == 5.0 and C.bridge_auc(sc, bm) == 1.0
    assert C.bridge_auc(-sc, bm) == 0.0 and C.bridge_auc(np.ones(len(sc)), bm) == 0.5
    assert np.isnan(C.edge_margin(sc, g, None)) and np.isnan(C.bridge_auc(sc, np.zeros(len(sc), bool)))


def test_q_select():
    for spec in (dict(kind="ring_of_cliques", q=4, s=12), dict(kind="ring_of_cliques", q=6, s=8),
                 dict(kind="two_block", block="clique", bridges=2)):
        g, t, m = C.make_graph(dict(spec, seed=0))
        r = C.run_diff(g, "mean", _u0(g), 0.5, 60)
        q = C.q_select(C.edge_score(r.X, g), g)["q"]
        assert q == m["q"], (m["id"], q)
    g, t, m = C.make_graph(dict(kind="ring_of_cliques", q=4, s=12))
    sc = np.where(m["bridge_mask"], 1.0, 0.1)
    assert C.q_select(sc, g)["q"] == 4
    assert C.q_select(np.full(len(sc), 0.1) + 1e-3 * np.random.default_rng(0).random(len(sc)), g, min_rel_gap=1.0)["q"] == 1
    top = C.mst_edge_scores(g, sc)
    assert len(top) == g.n - 1 and (np.diff(top) <= 0).all()


def test_ari_over_time_on_disconnected():
    g, t, m = C.make_graph(dict(kind="two_block", block="sbm", bridges=0, seed=0))
    r = C.run(g, "wave", "mean", _u0(g), 0.5, 400)
    o = C.ari_over_time(r.X, g, t, window=100, stride=50, fail_step=r.fail_step)
    assert len(o["ari"]) == 7 and np.nanmin(o["ari"]) == 1.0 and (o["t_end"] - o["t_start"] == 100).all()
    r = C.run(g, "wave", "trim", _u0(g), 1.0, 400)
    o = C.ari_over_time(r.X, g, t, window=100, stride=50, fail_step=r.fail_step)
    if r.fail.all():
        assert np.isnan(o["ari"][-1])


# ---------------------------------------------------------------------------------------------- hold tests
def test_hold_matches_rx_and_min_frac_fix():
    g, t, m = _g("clique", 2)
    r = C.run(g, "damp", "mean", _u0(g, 8), 0.5, 500, ps=dict(k=0.01))
    for lab in (t, np.random.default_rng(0).integers(0, 2, g.n)):
        st_rx = rx.window_stats(r.X, g, lab)
        st_me = C.window_stats(r.X, g, lab)
        for a, b in zip(st_rx, st_me):
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12)
        for par in (dict(eps=0.3, delta=0.1, rho=2.0), dict(eps=1.0, delta=0.05, rho=1.0)):
            assert np.array_equal(rx.hold_steps(st_rx, **par), C.hold_steps(st_me, **par))
    # diffusion on a clean two-block graph holds under the fixed params; random labels do not
    g, t, m = _g("clique", 1, seed=2)
    r = C.run_diff(g, "mean", _u0(g), 0.5, 500)
    h = C.hold(r.X, g, t)
    assert h["rejected"] is False and h["minfrac"] == 0.5 and (h["hold"] > 0).all() and h["hold_pos"].min() == 1
    assert h["din"].shape == h["dout"].shape == (r.R, len(h["t_start"]))
    # the known defect: one cut-off node passes the rx criterion; hold() rejects it
    T1, R, n = 501, 4, g.n
    rng = np.random.default_rng(0)
    X = 0.01 * rng.normal(size=(T1, R, n))
    X[:, :, 0] = 1.0
    lab = np.zeros(n, int)
    lab[0] = 1
    st = rx.window_stats(X, g, lab)
    assert (rx.hold_steps(st, 0.3, 0.1, 2.0) > 0).all()                      # rx: spurious hold
    hh = C.hold(X, g, lab, min_frac=0.1)
    assert hh["rejected"] and (hh["hold"] == 0).all()
    assert (C.hold(X, g, lab, min_frac=0.0)["hold"] > 0).all()               # fix switched off reproduces rx
    assert C.hold(X, g, None)["rejected"]


# ---------------------------------------------------------------------------------------------- stimulus / confounders
def test_stimulus_linear_mean_wave():
    g, t, m = _g("clique", 2)
    rows = C.stimulus(g, 24, C.make_runner("wave", "mean", 0.5, steps=200), amps=(0.2, 1, 5), steps=200, R=8)
    assert len(rows) == 6
    by = {(r["amp"], r["sign"]): r for r in rows}
    for s in (1, -1):
        assert abs(by[(5.0, s)]["far_peak"] / by[(1.0, s)]["far_peak"] - 5) < 1e-6      # exactly linear response
        assert abs(by[(1.0, s)]["far_peak"] / by[(0.2, s)]["far_peak"] - 5) < 1e-6
    assert abs(by[(1.0, 1)]["far_peak"] - by[(1.0, -1)]["far_peak"]) < 1e-9             # sign symmetric
    assert by[(5.0, 1)]["det_auc"] > 0.95 and by[(5.0, 1)]["arrived_frac"] == 1.0
    assert by[(5.0, 1)]["det_auc"] >= by[(0.2, 1)]["det_auc"]
    assert 0 < by[(1.0, 1)]["transmit_ratio"] < 1 and by[(1.0, 1)]["fail_frac"] == 0
    # diffusion runner through the same API; asymmetric force: owa wave fails or is sign-asymmetric
    rows = C.stimulus(g, 24, C.make_diff_runner("owa", 0.5), amps=(1,), steps=100, R=4, det_trials=8)
    assert len(rows) == 2 and all(np.isfinite(r["far_peak"]) for r in rows)
    rows = C.stimulus(g, 24, C.make_runner("wave", "owa", 1.0), amps=(1,), steps=100, R=4, det_trials=8)
    assert all(r["fail_frac"] == 1 and np.isnan(r["far_peak"]) and np.isnan(r["det_auc"]) for r in rows)


def test_confounders():
    g, t, m = _g("clique", 2)
    base = C.make_runner("wave", "mean", 0.5, steps=100)
    u0 = _u0(g, 4)
    r = C.with_outliers(base, 0.1, "stuck", seed=3)(g, u0)
    idx = r.outlier_idx
    assert len(idx) == 5 and (r.X[:, :, idx] == u0[None, :, idx]).all()
    rest = np.setdiff1d(np.arange(g.n), idx)
    assert not np.allclose(r.X[-1][:, rest], u0[:, rest])
    wr = C.with_outliers(base, 0.1, "stuck", seed=3, value=2.0)(g, u0)
    assert (wr.X[1:, :, wr.outlier_idx] == 2.0).all()
    nz = C.with_outliers(base, 0.1, "noisy", seed=3, scale=0.1)
    a, b = nz(g, u0), nz(g, u0)
    assert np.array_equal(a.outlier_idx, b.outlier_idx) and not np.allclose(a.X, b.X)      # fresh noise per call
    assert np.abs(a.X - C.make_runner("wave", "mean", 0.5, steps=100)(g, u0).X).max() > 0.01
    bn = C.bridge_noise(base, m["bridge_mask"], sigma=0.1)
    r = bn(g, np.zeros((2, g.n)))
    ends = np.unique(np.r_[g.ei[m["bridge_mask"]], g.ej[m["bridge_mask"]]])
    assert np.abs(r.X[:, :, ends]).max() > 0
    # noise spreads from the bridge endpoints: bridge endpoints are noisier at the start
    assert np.abs(r.X[1]).max() == np.abs(r.X[1][:, ends]).max() or np.abs(r.X[1][:, ends]).max() > 0
    # stimulus accepts wrapped runners and a confounded run still reports failures
    rows = C.stimulus(g, 24, C.with_outliers(base, 0.05, "stuck"), amps=(5,), steps=100, R=4, det_trials=8)
    assert len(rows) == 2 and rows[0]["n_ok"] == 4
    # picklable (needed for pmap)
    import pickle
    pickle.loads(pickle.dumps(nz))


# ---------------------------------------------------------------------------------------------- stats / io
def test_stats_helpers():
    p = np.array([0.01, 0.04, 0.03])
    assert np.allclose(C.holm(p), [0.03, 0.06, 0.06])
    assert np.allclose(C.holm([0.5, np.nan, 0.001]), [0.5, np.nan, 0.002], equal_nan=True)
    assert (C.holm(np.array([0.9, 0.95])) <= 1).all()
    x = np.random.default_rng(0).normal(5, 2, 100)
    est, lo, hi = C.boot_ci(x)
    assert abs(est - x.mean()) < 1e-12 and lo < est < hi and 0.5 < hi - lo < 1.3 and C.boot_ci(x) == C.boot_ci(x)
    assert np.isnan(C.boot_ci([])[0]) and np.isnan(C.boot_ci([1.0])[1])
    assert C.boot_ci([1, 2, np.nan, 3])[0] == 2.0
    a = np.random.default_rng(1).normal(size=30)
    w = C.paired_wilcoxon(a + 1.0, a)
    assert w["p"] < 1e-4 and w["n"] == 30 and abs(w["mean_diff"] - 1) < 1e-12
    w = C.paired_wilcoxon(a, a)
    assert w["p"] == 1.0 and w["n_nonzero"] == 0
    w = C.paired_wilcoxon(np.r_[a, np.nan], np.r_[a + 0.5, 1.0])
    assert w["n"] == 30
    import pandas as pd
    rows = []
    for gi in range(12):
        for seed in range(2):
            b = np.random.default_rng(gi * 10 + seed).normal()
            rows += [dict(graph=gi, seed=seed, cfg="base", v=b), dict(graph=gi, seed=seed, cfg="up", v=b + 1.0),
                     dict(graph=gi, seed=seed, cfg="same", v=b)]
    res = C.paired_vs_baseline(pd.DataFrame(rows), "v", "cfg", "base")
    assert set(res["config"]) == {"up", "same"} and res.set_index("config").loc["up", "p_holm"] < 1e-3
    assert res.set_index("config").loc["same", "p"] == 1.0 and (res["n"] == 24).all()


def test_write_csv_roundtrip():
    import csv
    with tempfile.TemporaryDirectory() as d:
        rows = [dict(a=np.float64(1.5), b=np.int64(2), c="x"), dict(a=3.0, d=np.array([1, 2]), b=float("nan"))]
        p = C.write_csv(Path(d) / "sub" / "dir" / "t.csv", rows)
        out = list(csv.DictReader(p.open()))
        assert list(out[0]) == ["a", "b", "c", "d"] and out[0]["a"] == "1.5" and out[1]["d"] == "[1, 2]"
        import pandas as pd
        p = C.write_csv(Path(d) / "df.csv", pd.DataFrame(rows))
        assert p.exists()


def test_pmap_order_and_cap():
    assert C.pmap(_sq, range(10)) == [i * i for i in range(10)]
    pids = set(C.pmap(_pid, range(16), procs=8))
    assert 1 <= len(pids) <= 2 and os.getpid() not in pids, pids
    assert C.pmap(_sq, [3], procs=2) == [9] and C.pmap(_sq, [1, 2], procs=1) == [1, 4]


def test_import_hygiene():
    assert wt.FAIL_AMP == 20.0 and C.FAIL_AMP == 20.0
    assert C.rx is wt.rx and C.MAX_PROCS == 2


def main():
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(k, v) for k, v in sorted(globals().items(), key=lambda kv: kv[1].__code__.co_firstlineno if callable(kv[1]) and hasattr(kv[1], "__code__") else 0)
             if k.startswith("test_") and sel in k]
    bad = 0
    t00 = time.time()
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            bad += 1
            print(f"FAIL {name} ({time.time() - t0:.1f}s)\n{traceback.format_exc()}", flush=True)
    print(f"\n{len(tests) - bad}/{len(tests)} passed in {time.time() - t00:.0f}s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
