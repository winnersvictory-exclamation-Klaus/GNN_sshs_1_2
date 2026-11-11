# `battery/common.py` — API reference

Read this instead of the source. Everything below was run; `python battery/test_common.py` (35 self-tests, ~12 s) must pass.
Do not edit `common.py` (work around bugs locally and report them).

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # battery/ (for scripts in battery/ideaN/)
import common as C        # import FIRST (pins BLAS to 1 thread). C.wt = wave_test, C.rx = rank_experiment
```

## Conventions (read once)

* Trajectory `X`: `(steps+1, R, n)`, `X[0]=u0`. `fail_step`: `(R,)` int, `-1` = never failed, else `s`: `X[:s]` valid, **`X[s:]==0`**.
* Failure threshold `C.FAIL_AMP = 20`. Per call: `fail_amp=...` (temporarily sets `wt.FAIL_AMP`). It is an *absolute* threshold, so for `u0*10` pass `fail_amp=200` if you mean "relative".
* `ps`/`pf` are **partial** dicts merged over registry defaults; unknown keys raise `KeyError` (typo guard).
* Graphs are `rx.G`; `make_graph` caches them per process — never mutate. Weighted degree `C.wdeg(g)` is the `D` of `L_rw = I - D⁻¹A`; `L_rw` is non-symmetric, so every "eigen / energy / symmetry" quantity here uses the **D-inner product** `<a,b>_D = aᵀDb`.
* Mean force: `F(u) = A(u) − u = −L_rw u`. Diffusion update: `u+ = u + eta·F`. Wave: `u+ = 2u − u_prev + c²F` (start-up `u_prev = u0`).
* Metrics take `fail_step` and mask failed runs. What failed realisations give:

| metric | failed realisation |
|---|---|
| `ac_final`, `rms_final` | NaN; `dead` → False |
| `growth_rate`, `dc_drift`, `ac_peak` | computed on the valid window **before** failure (NaN if < 6 valid steps) |
| `energy`, `mode_energy` | NaN from the failure step on |
| `edge_score`, `partitions`, `hold`, `ari_over_time` | failed runs are excluded (all failed → `None` / NaN) |
| `stimulus` | trials where either run failed are dropped (`n_ok`, `fail_frac` reported; all dropped → NaN row) |

## Graphs

| function | notes |
|---|---|
| `make_graph(spec, cache=True) -> (g, truth\|None, meta)` | `spec['kind']` ∈ `two_block`(block=`cycle\|clique\|sbm`, sizes, bridges, wb) · `barbell_path`(k, L) · `grid_neck`(a, w, wb) · `ring_of_cliques`(q≥3, s) · `sbm_q`(q, size, p_in, p_out) · `binary_tree`(depth; truth = two root subtrees, root labelled 0) · `lollipop`(k, L; truth = clique vs tail) · `watts_strogatz`(n, k, p) · `random_regular`(n, d) · `er`(n or sizes, deg) · `shuffled`(sizes) · `sbm_het`(sizes, bridges, wb, sigma, p_in). `seed` always accepted. rx kinds `cycle/clique/sbm` are aliases for `two_block`; `two_block`/`er`/`shuffled` reproduce `rx.prepare` exactly (tested). Missing keys → defaults (`normalize_spec`). |
| `meta` | `id, kind, n, m, seed, params, q` (#true blocks/None), `connected, bridge_mask` (bool `(m,)` over `g.ei/g.ej`, or None), `bridge_idx, n_bridges, dmax, mean_deg` |
| `zoo(name, seeds=(0,1,2)) -> [spec]`, `build_zoo(name, seeds) -> [Case(spec,g,truth,meta)]` | names: `core` (cycle/clique/sbm × bridges 1,2,4,8, 24+24, wb 1: 12/seed) · `core_full` (× sizes 24+24,16+32,8+40 × wb 1,0.25: 72/seed) · `wide` (15/seed) · `neg` (er, shuffled, random_regular d=3,4,6,8, watts_strogatz; truth None: 7/seed) · `disc` (bridges=0, **disconnected** control, 3/seed) · `scale` (n=48,96,192,384 × cycle/clique/sbm, bridges 2: 12/seed) |
| `spec_id(spec)`, `normalize_spec(spec)`, `bridge_mask(g, truth)`, `is_connected(g)` | |
| `wdeg(g)`, `L_rw(g)`, `spectrum(g) -> (lam, Phi, d)` | `lam` ascending (`lam[0]≈0`), `Phi` columns D-orthonormal eigenvectors of `L_rw` (`Phiᵀ diag(d) Phi = I`) |
| `DEV_SEEDS=0..7`, `HELD_SEEDS=100..115` | |

## Initial conditions

`init(kind, g, R, rng, truth=None, scale=1.0, dc=0.0, rms=0.577, **kw) -> (R,n)`; `rng = np.random.default_rng(seed)`.
Kinds: `uniform` (== `rng.uniform(-1,1,(R,n))`, the rx recipe), `gauss`, `rademacher`, `student_t3` (clipped at 5σ; `clip=None` to disable — unclipped tails put `u0` itself above `FAIL_AMP` now and then) — all matched to RMS `rms` = RMS of U(-1,1) so thresholds mean the same thing; `impulse` (`node=`, `amp=1`; RMS(u0) is tiny!), `low_mode`/`high_mode` (±D-orthonormal eigenvector `k=1`/`-1` + `noise_frac=0.1` gaussian, RMS `rms`), `block_offset` (needs `truth`; `delta=`Δ, uniform noise × `noise=1`; block b gets `(b−(q−1)/2)·Δ/(q−1)`). Modifiers: `u*scale + dc`.

## Simulation

| function | notes |
|---|---|
| `run(g, scheme, force, u0, c=1.0, steps=500, ps=None, pf=None, fail_amp=None) -> Run` | wraps `wt.simulate`. `Run`: `.X, .fail_step, .fail (bool R), .ok, .R, .valid_len, .g, .u0, .c, .scheme, .force, .ps, .pf, .steps, .fail_amp` |
| `run_diff(g, force, u0, eta=0.5, steps=500, pf=None, fail_amp=None) -> Run` | `u+ = u + eta·F` for **any** registered `wt` force via a `wt.Ctx` (`force='mean', eta=0.5` ≡ `rx.simulate(...,'diff')`, tested bit-exact). `Run.scheme='diffusion'`, `.eta` set, `.c` NaN |
| `make_runner(scheme, force, c, ps, pf, steps, fail_amp) -> Runner`, `make_diff_runner(force, eta, pf, steps, fail_amp)` | picklable callable `runner(g, u0, steps=None) -> Run`; used by `stimulus`/confounders/`pmap` jobs; `.label` |
| `fail_frac(g, scheme, force, u0, c, steps=2000, ps, pf, fail_amp) -> float` | early-exit (stops when all failed), keeps no trajectory — cheap for sweeps |
| `resolve_params(wt.FORCES\|wt.SCHEMES, name, over)` | full param dict |

`wt.simulate` is replicated step-for-step by the internal integrator (used for diffusion, hooks, early exit): bit-identical on all 17 schemes × 10 forces incl. failures (tested).

## Trajectory metrics (per realisation → `(R,)` unless noted)

| function | meaning |
|---|---|
| `traj_metrics(run, W=50, dead_thr=0.05) -> dict` | all of: `fail, fail_step, growth_rate, dc_plain, dc_dw, ac_final, rms_final, dead, ac_peak` |
| `traj_summary(run, ...) -> dict` (scalars for a CSV row) | `R, n_ok, fail_frac, mean_fail_step, dead_frac, ac_final, rms_final` (medians over ok), `growth_rate, dc_dw, dc_plain, ac_peak` (medians) |
| `growth_rate(X, fail_step, frac=0.5, ...)` | LS slope of log AC-RMS per step over the last half of the valid window (>0 growing). Points < 1e-12×initial are dropped; NaN if < 4 left. **Meaningless for decayed (dead) runs.** |
| `dc_drift(X, g, fail_step, W=50) -> {plain, dw}` | final-window node-mean (plain / degree-weighted) − initial. `dw` is conserved by the mean force (tested to 1e-9), `plain` is not on irregular graphs |
| `ac_final`, `rms_final`, `dead`, `ac_peak`, `ac_rms(X, g=None, weighted=False) -> (T+1,R)` | `ac_final` = AC-RMS of last W steps / AC-RMS(u0) (wt divides by RMS incl. DC; differs only with `dc≠0`) |
| `fail_modes(g, scheme, force, u0, c, steps, ps, pf, fail_amp, big=1e6) -> dict` | re-runs with threshold `big`; `raw_step` (== normal `fail_step`), `ac_step` (first step where `max|u−mean|≥A`), `raw_fail, ac_fail, dc_only` — drift vs oscillation |
| `energy(X, g, c, fail_step, kind='staggered') -> (T,R)` | `½‖u_{t+1}−u_t‖²_D + ½c² u_tᵀ(D−A)u_{t+1}`: **exactly** conserved by wave+mean (rel. error < 1e-6 tested; 1.3e-15 measured). `kind='naive'` uses `u_tᵀ(D−A)u_t` and oscillates. Not conserved by damp/viscous/nonlinear schemes. `energy_drift(E) -> {ratio_end, ratio_max, ratio_min}` relative to `E_0` |
| `mode_energy(X, g, fail_step, n_low=2, high_frac=0.25, weight='disp', full=False) -> dict` | `low, high` `(T+1,R)` energy fractions of modes 1..n_low / top quartile (D-orthonormal eigenmodes of `L_rw`, consensus mode excluded), `total`; `weight='pot'` weights by λ; `full` adds modal coordinates `a (T+1,R,n)` |
| `temporal_spectrum(X, fail_step, start_frac=0.0, nyq_band=0.45) -> {centroid, nyquist}` | AC state, Hann window; centroid in cycles/step; `nyquist` = power fraction at f ≥ 0.45 (period-2 flicker) |
| `lyapunov(g, scheme, force, u0, c, steps, ps, pf, eps=1e-8, seed=0, ...) -> {rate, t_sat, d}` | twin run `eps` away; rate = slope of log d until saturation. Twin is not renormalised → slow exponents underestimated; NaN for failed/decayed |
| `activation(g, scheme, force, u0, c, steps, ps, pf, thr=0.1) -> {frac (steps,R), mean_frac, late_frac, first_quiet, fail_step}` | fraction of node-steps where `|u_next − u_wave| > thr·|u_wave − u|` (stabiliser actually bites). 0 once the state has converged to exact consensus |

## Force property probes

| function | notes |
|---|---|
| `eval_force(g, force, U, pf=None, t=0, c=1.0) -> (R,n)` | **fresh `wt.Ctx` per call** (stateful forces start from initial state: `edge_memory` ≡ mean). All 110 registered forces are verified row-independent (batched evaluation is valid) |
| `jacobian(g, force, u, pf=None, h=1e-6, t=0, return_kink=False) -> (n,n)` | **central** difference, step `h·max(1,rms u)`, one batched call. **Warning:** trim/owa/median/symowa/huber (clips)/ties are discontinuous — J is an arbitrary one-sided mix at ties/kinks (Rademacher or constant-block states!). `return_kink=True` gives `‖J_fwd−J_bwd‖/‖J‖` (≈1e-6 smooth, O(1) at kinks) |
| `force_props(g, force, u=None, pf=None, ..., return_jac=False) -> dict` (floats; `u` is 1-D) | `oddness`, `shift_abs/rel`, `homog_abs/rel` (s=0.1,10), `const_defect`, `momentum`, `momentum_rel` (Σd_iF_i), `dsym_abs/rel` (‖DJ−(DJ)ᵀ‖), `lam_max` (top of −sym(DJ) in D metric = stiffness), `eig_sym_max` (>1e-8 ⇒ anti-restoring direction), `c_crit_pred = 2/√lam_max`, `rho_J`, `c_crit_eig` (from true, possibly complex, eigenvalues of J), `jac_kink` |
| `force_props_rows(g, force, U, pf, **kw) -> [dict]` | one dict per row of `U` |
| `modal_roots(mu, c, k=0, nu=0)`, `modal_rho(...)`, `c_crit_linear(mu, k=0, nu=0)` | per-mode roots of `z² − (2−k+(c²+ν)μ)z + (1−k+νμ)`, `μ=−λ` (checked against actual damp/viscous/damp_visc runs to 1e-11). **Damp is stable iff `c²λ < 4−2k`: damping *lowers* the top-mode limit** |
| `c_crit_empirical(g, scheme, force, u0=None, ps, pf, steps=2000, c_lo=0.02, c_hi=3, n_scan=16, rtol=0.01, R=8, seed=0) -> dict` | geometric scan for the FIRST failing c, then bisection. `c_crit` (nan if unstable at `c_lo`, inf if stable to `c_hi`), `status`, `non_monotone` (stable again above), `scan_c`, `scan_fail`. Depends on `steps` (near-threshold growth is slow) and on `u0` |

Expected for the mean force: `defects ≈ 0`, `c_crit_pred = c_crit_eig = 2/√λ_max(L_rw)`, empirical within 2% (tested).

## Partition readouts and metrics

| function | notes |
|---|---|
| `edge_score(X, g, fail_step, frac=0.5, lowpass_fc=None, per_run=False) -> (m,)` | RMS over the last `frac` of `u_i−u_j`, median over ok runs (rx procedure; chunked, memory-safe at n=384). `None` if all failed |
| `partitions(X, g, q=2, fail_step, which=None, fc=0.02, frac=0.5) -> {name: labels\|None}` | names `C.PARTITION_NAMES`: `edge_cut` (rx single-linkage cut), `edge_cut_lp` (after `rx.lowpass(fc=0.02)`), `timeavg_2means` (k-means on time-averaged AC state), `traj_corr_spectral` (NJW on node-trajectory correlation), `fft_lowmode` (Sahai-style; ≡ `rx.wave_fft_labels` for one realisation, tested). Pooled over ok runs |
| `partition_metrics(g, labels, truth) -> {ari, nmi, cond, minfrac, ok, n_parts}` | |
| `edge_margin(score, g, truth)` | median bridge-edge score / median intra-edge score. NaN if intra median is 0; **meaningless (1e15) when the state decayed to round-off** — check `dead`/amplitude first |
| `bridge_auc(score, bridge_mask)` | AUC of edge score vs true bridges (0.5 = chance; <0.5 = bridges are *quieter*) |
| `evaluate_partitions(X, g, truth, fail_step, q=None, which=None, fc, frac) -> dict` | flat CSV row: `n_ok`, `ari_/nmi_/cond_/minfrac_/ok_<readout>`, `edge_margin`, `bridge_auc`, `edge_margin_lp`, `bridge_auc_lp` (q default = #true blocks) |
| `ari_over_time(X, g, truth, window=100, stride=None, q=2, fail_step) -> {t_start, t_end, ari}` | edge_cut readout in sliding windows |
| `q_select(score, g, qmax=8, min_rel_gap=None) -> {q, gaps, top}`, `mst_edge_scores(g, score)` | largest gap of the descending MST edge scores; `min_rel_gap` enables q=1. Exact on `ring_of_cliques`; **poor on `sbm_q`** in my spot check (3/18 correct) |

## Quasi-equilibrium

`hold(X, g, labels, params=None, min_frac=0.1, fail_step=None, norm='rms') -> dict`
rx criterion (windows W=50, stride 25: `D_in ≤ 0.3`, `D_out ≥ 0.1`, `D_out/D_in ≥ 2`, rms ≥ 1e-3; hold = longest run, in steps) **plus** rejection (hold=0, `rejected=True`) of labels with any part < `min_frac` of the nodes. Returns `hold, hold_pos` (R,; NaN for failed), `din, dout, rms` `(R, n_windows)`, `t_start`, `rejected`, `minfrac`, `params`. `min_frac=0` reproduces rx exactly (tested; the defect is reproduced with a single cut-off node). `norm='ac'` normalises by the AC-RMS instead of the DC-inclusive RMS (rx's normaliser lets a drifting mean flatter `D_in`). Lower-level: `window_stats`, `hold_steps` (equal to rx at defaults, tested).

## Stimulus and confounders

`stimulus(g, n0, runner, amps=(0.2,1,5), signs=(+1,-1), sigma=0.05, R=8, steps=200, seed=0, det_trials=32, far=None, src_pool=None) -> [row dict]`
Near = nodes `< n0`, far = `≥ n0` (override with `far` mask / `src_pool`). Per (amp, sign) row: `amp, sign, R, n_ok, fail_frac, far_peak, far_peak_signed, near_peak, transmit_ratio, arrival, arrived_frac, snr, noise_far_rms, noise_gain, det_auc, det_n`. Response metrics use paired stimulus/noise-only runs (same noise); `det_auc` uses `det_trials` independent stimulus and noise-only trials and the raw far-side peak `max|u|` (what an observer sees). For wave+mean the response is exactly linear in amp and sign-symmetric (tested).
`with_outliers(runner, frac=0.05, mode='stuck'|'noisy', seed=0, scale=1.0, value='init')`, `bridge_noise(runner, bridge_mask, sigma=0.05, seed=0)` → wrapped runners (same call signature, work in `stimulus`; result has `.outlier_idx`). Outlier set is fixed per `seed`; per-step noise differs per call.

## Stats / IO

`boot_ci(x, n_boot=2000, alpha=0.05, seed=0, func=np.mean) -> (est, lo, hi)` (pass one value per graph) · `paired_wilcoxon(a, b) -> {n, n_nonzero, mean_diff, median_diff, stat, p}` · `holm(pvals)` · `paired_vs_baseline(df, value, config, baseline, pair=('graph','seed')) -> DataFrame[config, n, mean_diff, ci_lo, ci_hi, median_diff, p, p_holm]` · `write_csv(path, rows|DataFrame) -> Path` (makes dirs, numpy-safe) · `pmap(fn, jobs, procs=2, chunksize=1, progress=False) -> list` (ordered; **hard cap 2 workers** via `C.MAX_PROCS` / env `BATTERY_MAX_PROCS`; `fn` must be top-level and the script needs `if __name__ == "__main__":`; spawn start, ~1–2 s startup).

## Minimal complete test script (run verbatim, 6 s)

```python
"""battery/ideaN/t00_demo.py: zoo graphs -> configs -> metrics -> CSV"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # battery/
import common as C
import numpy as np

CONFIGS = [  # label, runner
    ("wave+mean c=0.5", C.make_runner("wave", "mean", c=0.5, steps=500)),
    ("damp(k=.1)+mean c=1", C.make_runner("damp", "mean", c=1.0, ps=dict(k=0.1), steps=500)),
    ("damp(k=.1)+owa c=1", C.make_runner("damp", "owa", c=1.0, ps=dict(k=0.1), steps=500)),
    ("diffusion+mean", C.make_diff_runner("mean", eta=0.5, steps=500)),
]

def job(args):                                   # top-level, picklable; ONE graph x all configs
    spec, R = args
    g, truth, meta = C.make_graph(spec)
    rng = np.random.default_rng(spec["seed"] * 1000 + 17)
    u0 = C.init("uniform", g, R, rng)
    rows = []
    for label, runner in CONFIGS:
        run = runner(g, u0)                                  # -> C.Run (X, fail_step, ...)
        row = dict(graph=meta["id"], seed=spec["seed"], config=label, n=meta["n"], bridges=meta["n_bridges"])
        row.update(C.traj_summary(run))                      # fail_frac, dead_frac, ac_final, growth_rate, ...
        row.update(C.evaluate_partitions(run.X, g, truth, run.fail_step))   # ari_*, edge_margin, bridge_auc
        labs = C.partitions(run.X, g, 2, run.fail_step, which=("edge_cut",))
        h = C.hold(run.X, g, labs["edge_cut"], fail_step=run.fail_step)     # min_frac fix applied
        row["hold_rec"] = float(np.nanmean(h["hold"])) if run.ok.any() else np.nan
        rows.append(row)
    return rows

if __name__ == "__main__":                       # guard is REQUIRED (pmap spawns workers)
    specs = C.zoo("core", seeds=C.DEV_SEEDS[:2])
    out = C.pmap(job, [(s, 8) for s in specs], procs=2)          # list of per-graph row lists
    rows = [r for rs in out for r in rs]
    path = C.write_csv(Path(__file__).parent / "results" / "t00_demo.csv", rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    for cfg, sub in df.groupby("config", sort=False):            # mean + bootstrap CI over graphs
        print(f"{cfg:22s} fail={C.boot_ci(sub.fail_frac)[0]:.2f} ari_edge_cut=%.3f [%.3f, %.3f] n=%d"
              % (*C.boot_ci(sub.ari_edge_cut), sub.ari_edge_cut.notna().sum()))
    print(C.paired_vs_baseline(df, "ari_edge_cut", "config", "diffusion+mean", pair=("graph", "seed")).round(3))
    print("wrote", path, len(rows), "rows")
```
Output (24 graph×seed pairs per config): wave+mean ARI 0.15, damp(.1)+mean 0.20, diffusion+mean 0.76, damp+owa fail 100%.

## Things that will bite

* **Statistics unit = graph (×seed).** `traj_summary`/`evaluate_partitions` already collapse the R realisations; bootstrap and Wilcoxon over graphs, paired by `(graph, seed)`.
* `ac_final`/`dead` need ok runs; a config that fails everywhere has NaN ARI — report `n` with every mean.
* Rank forces (trim, owa, median, vw ties) are **discontinuous**: Jacobians are only a.e. derivatives (see `jac_kink`).
* `run()` ≈ 10–60 ms at n=48, R=8, T=500 (mean 14 ms, trim/owa/median ≈ 55–65 ms, `hi_geo:owa` 210 ms); n=384 clique blocks: mean 0.9 s, owa 4.8 s. `evaluate_partitions` ≈ 26 ms, `stimulus` ≈ 140 ms, `force_props` ≈ 3–4 ms, `c_crit_empirical` ≈ 1–3 s.
* Multi-aggregation forces cost K aggregations per step (series); equal-cost tests must account for it yourself.
