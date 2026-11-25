# 002 — Center drift vs. relative growth in legacy `huber_local` wave (development stability diagnostic)

Plan §3 E diagnostic for the configuration that failed the gate in 001. Development graphs only: no confirmation seeds, no new graphs, no parameter sweep, no performance claim. All forces, graphs, initial states and readouts come from the existing reference code (`GNN_testing/wave_test.py`, `GNN_testing/battery/common.py`, `rank_experiment.py`). The only new code is the integrator without zeroing, the projection wrapper and the edge-symmetric candidate force.

## Reproduction

From the repository root (`GNN_sshs_1_2/`):

```
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/002-center-drift/center_drift.py --out /tmp/002-rerun
diff -r -x center_drift.py -x README.md /tmp/002-rerun agents-collab/results/002-center-drift
```

Published outputs used Python 3.15.2. Any Python 3.15 interpreter can replace `~/.venvs/math/bin/python`. Without `--out`, the script writes next to itself. Two independent runs gave byte-identical outputs (gzip mtime fixed at 0). The script sets `GNN_REPO` itself and writes no bytecode into the shared code directories. It exits with status 1 if any hard check fails.

## Key outcomes (descriptive)

1. **The replay reproduces the archive exactly.** Legacy replay with the t310 RNG (`seed*1000+310`, eight U(−1,1) draws, `c=0.5`, `prev=cur`, 500 steps): 69/384 failed initializations on 30/48 graphs. Each graph's `n_ok` and mean fail step match `t310_partition.csv`, as do survivor-based raw/LP ARI and bridge AUC. There are no discrepancies. Without zeroing, the first absolute hit equals the legacy `fail_step` for every initialization. The pre-failure prefix is bit-identical.
2. **Most absolute crossings are driven by the center, but relative motion also grows.** At the first absolute crossing, |m| exceeded the centered max in 61/69 failures (median |m| 16.5, median centered max 4.05). Classes of the 69 failures:
   - Center-only: 6. The centered max stayed below 20 and peak centered RMS stayed below 2× its initial value.
   - Mixed: 59. Centered max stayed below 20, but peak centered RMS was at least 2× initial.
   - Relative crossing: 4. The centered max itself crossed 20.
   
   Relative growth is not confined to failures. Of the 315 non-failures, 131 have a centered-RMS ratio of at least 2, including all 81 surviving cycle initializations. Cycle failures and cycle survivors have similar median ratios (3.5 vs 3.2). In cycles, the center drift decides which runs cross 20. The three clique failures are genuine relative blow-ups (ratios 45, 126, 490). The centered max crossed 20 within 500 steps for only 4/384 initializations (on 4 graphs). Because the centered dynamics do not depend on m, a center-free legacy run would show the same 4 crossings, up to rounding.

   | Family (128 inits each) | Center-only | Mixed | Relative crossing | No crossing, ratio ≥2 | No crossing |
   |---|---:|---:|---:|---:|---:|
   | Cycle | 0 | 47 | 0 | 81 | 0 |
   | Clique | 0 | 0 | 3 | 8 | 117 |
   | SBM | 6 | 12 | 1 | 42 | 67 |

3. **Projection is an algebraic control, not an accuracy gain.** On the 12-graph slice (Taylor startup, T=250), `huber_local` failed 17/96 initializations on 8/12 graphs at h=0.5, and 12/96 on 6/12 at h=0.25. With the degree-weighted mean force removed, failures drop to 1/96 at h=0.5 and 0/96 at h=0.25. The remaining failure is a relative crossing with centered-RMS ratio 247 (clique, 4 bridges, seed 1, init 0). It does not occur at h=0.25. The projected trajectory equals the re-centered original. Edge scores, AUC and descriptive ARI are therefore identical. Only the plan score changes: LP 0.333 → 0.794 at h=0.5, 0.493 → 0.847 at h=0.25. That difference comes purely from the failure denominator.
4. **The edge-symmetric candidate has an exactly fixed center but loses most of the bridge magnitude contrast.** It had 0/96 failures at both h. Its Σ d F is zero to rounding (≤ 3e-16 relative). Two initializations still show a centered-RMS ratio of 2.1–2.4, both on clique graphs with 4 bridges. A conserved center says nothing about energy or relative stability. Median LP bridge edge RMS is 0.067 against 0.016 internal, versus 0.50 against 0.024 for the original. Mean wave gives 0.053 against 0.011. LP ARI is 0.672 at h=0.5 and 0.850 at h=0.25: the candidate's readout changes with step size. It is also flagged as discretization-sensitive on 5/12 graphs, versus 1/12 for Mean.
5. **OWA / odd_OWA controls show the same mechanisms separately.** On the cycle graph, OWA had all eight initializations censored at 1e8 in both original and projected form, at both h. The centered max crossed 20 by t ≈ 12–19, so projection does not rescue this relative-mode instability. On the clique graph, original OWA failed 8/8 through center drift (|m| ≈ 18–20 at crossing, centered max ≤ 2.8). The projected form failed 3/8 at h=0.5 and 2/8 at h=0.25, all through relative crossings near t ≈ 200. `odd_owa` did not fail. Its center acceleration is still nonzero (max 0.012 on the clique), and m moves by up to 0.72× max|u|. Oddness under sign reversal does not conserve the weighted center.
6. **No stabilization is shown to keep the signal.** The plan's §2 matrix, including same-force diffusion, is still required before any discovery or performance statement. Nothing here supports a partition method.

### Slice summary (12 graphs, 96 initializations per row)

Descriptive ARI uses every uncensored initialization, including those that crossed 20. Plan ARI is zero for a graph if any of its eight initializations crossed. Values are means over graphs; AUC and edge RMS are medians over graphs. Edge RMS is shown as bridge / internal.

| Force | h | Fail runs / graphs | Centered fail | Ratio ≥2 inits | Raw ARI desc / plan | LP ARI desc / plan | LP AUC | LP edge RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `huber_local` | 0.5 | 17 / 8 | 1 | 55 | 0.334 / 0.250 | 0.877 / 0.333 | 1.000 | 0.498 / 0.024 |
| `huber_local` | 0.25 | 12 / 6 | 0 | 44 | 0.336 / 0.250 | 0.847 / 0.493 | 1.000 | 0.412 / 0.023 |
| projected `huber_local` | 0.5 | 1 / 1 | 1 | 55 | 0.334 / 0.334 | 0.877 / 0.794 | 1.000 | 0.498 / 0.024 |
| projected `huber_local` | 0.25 | 0 / 0 | 0 | 44 | 0.336 / 0.336 | 0.847 / 0.847 | 1.000 | 0.412 / 0.023 |
| edge-symmetric candidate | 0.5 | 0 / 0 | 0 | 2 | 0.172 / 0.172 | 0.672 / 0.672 | 0.998 | 0.067 / 0.016 |
| edge-symmetric candidate | 0.25 | 0 / 0 | 0 | 2 | 0.286 / 0.286 | 0.850 / 0.850 | 1.000 | 0.070 / 0.015 |
| Mean | 0.5 | 0 / 0 | 0 | 0 | 0.260 / 0.260 | 0.848 / 0.848 | 1.000 | 0.053 / 0.011 |
| Mean | 0.25 | 0 / 0 | 0 | 0 | 0.260 / 0.260 | 0.835 / 0.835 | 1.000 | 0.051 / 0.009 |
| fixed Huber | 0.5 | 0 / 0 | 0 | 0 | 0.000 / 0.000 | 0.518 / 0.518 | 0.989 | 0.079 / 0.018 |
| fixed Huber | 0.25 | 0 / 0 | 0 | 0 | 0.000 / 0.000 | 0.503 / 0.503 | 0.997 | 0.083 / 0.018 |

Graph classes at h=0.5 / 0.25:
- `huber_local` (original and projected): 11/10 relative growth, 1/2 stable with ARI ≥ 0.5.
- Candidate: 2/2 relative growth, 4/2 stable non-informative, 6/8 stable with ARI ≥ 0.5.
- Mean: 2 stable non-informative at both h; both are cycles with 4 bridges.
- Fixed Huber: 7/6 stable non-informative.

No readout fell to the noise floor, and every cut was scorable. Graphs flagged as discretization-sensitive: `huber_local` 6/12, projected 4/12, candidate 5/12, Mean 1/12, fixed Huber 5/12. In cycles, the original's failures are concentrated: 13 of 17 at h=0.5 and 10 of 12 at h=0.25. Full detail is in `slice_summary.csv`, `slice_graphs.csv` and `slice_h_sensitivity.csv`.

## Methods

**Graphs and initial states.** `C.zoo('core', DEV_SEEDS[:4])`: 48 graphs covering cycle/clique/SBM, 24+24 nodes, 1/2/4/8 unit-weight bridges and seeds 0–3. Initial states are `C.init('uniform', g, 8, default_rng(seed*1000+310))`, as in `t310_partition.py`. The slice is the 12 graphs with 1 or 4 bridges and seed 0 or 1, with the same eight vectors. The OWA controls use the cycle and clique graphs with seed 0 and 1 bridge.

**Part 1 (replay).**
- *Legacy run:* `C.run(g,'wave','huber_local',u0,c=0.5,steps=500)`, using `wt.simulate` with zeroing.
- *Diagnostic run:* identical arithmetic (`2u−u₋+0.25F`, `prev=cur`), without zeroing, to step 500.
- Tracked for each initialization and frame:
  - m = Σd u/Σd;
  - its acceleration a = Σd F(u)/Σd;
  - max|u|;
  - centered max max|u−m|;
  - degree-weighted centered RMS √(Σd (u−m)²/Σd).

  The discrete identity m₊−2m+m₋ = h²a is checked. Because `huber_local` depends only on differences, the centered dynamics do not depend on m.
- *Archive comparison:* survivor-based readouts of the legacy run are compared with the archive.

**Parts 2–3 (slice).**
- *Scheme:* undamped central differences, u₁ = u₀ + ½h²F(u₀) (zero-velocity Taylor startup) at both h=0.5 (500 steps) and h=0.25 (1000 steps), to T=250. This startup is *not* the legacy replay; it is labelled `taylor_zero_velocity`.
- *Configurations:* original, projected (`F − Σd F/Σd`), edge-symmetric candidate, Mean and fixed Huber (τ=0.5). OWA and odd_OWA, original and projected, run on the two control graphs only.
- *Candidate:* τᵢ = κ·median_j|u_j−u_i| + 1e−12, with κ=1 and the unweighted neighbor median `wt.nb_median` (the original rule). For an oriented edge with s = u_j−u_i, the flux w/2·(clip(s,±τᵢ)+clip(s,±τⱼ)) is added to i and subtracted from j, then divided by the weighted degree.
- *Projection vs. re-centered original:* compared over the frames where both runs are finite. This check is run for `huber_local`, OWA and odd_OWA.

**Part 4 (readout).**
- *Frames:* h=0.5 uses every step; h=0.25 uses every second substep. Both give the same 501 frames at t = 0, 0.5, …, 250.
- *Edge score:* `C.edge_score` on frames 250–500, raw and after `rx.lowpass` at 0.02 cycles per saved frame. Each score is the median over uncensored initializations.
- *Cut and metrics:* `rx.cut_partition` with q=2, sklearn ARI, `C.bridge_auc` and `C.edge_margin` (bridge median ÷ internal median).

**Failure, censoring and classification rules**
- *Absolute failure:* the first updated frame with `not all(|u| < 20)`, nonfinite included, checked at every substep. The run continues without zeroing.
- *Centered crossing:* the first finite frame with max|u−m| ≥ 20.
- *Censoring:* an initialization is censored from its first nonfinite frame, or once max|u| > 1e8 (the emergency limit). Its state is NaN from that frame on; it is excluded from the descriptive readout and counts as a failure. Censoring occurred only on the OWA cycle control (16 runs per variant across the two h; never nonfinite).
- *Relative-growth evidence:* peak centered RMS ≥ 2× its t=0 value. Mean wave gives exactly 1. Fixed Huber reaches at most 1.32.
- *Per-initialization class,* in priority order: `censored`; `relative_crossing`; `mixed_center_crossing_relative_growth` (absolute crossing with ratio ≥2); `center_only_crossing`; `no_crossing_relative_growth`; `no_crossing`.
- *Per-graph class,* in priority order:
  - `censored`: any initialization censored;
  - `relative_growth`: any initialization in a relative class;
  - `center_only_crossing`;
  - `stable_noise_floor`: bridge and internal medians both < 1e−6, raw and LP;
  - `stable_noninformative`: raw and LP ARI both < 0.5;
  - otherwise `stable_ari_ge_0.5`.
- *Discretization-sensitive:* the graph class differs between h, the failure count differs, or |ΔARI| > 0.1 in either readout. The relative difference of the centered state between h at common times is reported but is O(1) by t=250 even for linear Mean (dispersion). Do not use it as a stability criterion.

These thresholds give descriptive labels, not decision rules. The raw ratios, peaks and first-hit times are in the tables.

## Validation (`checks.csv`, `summary.json`)

| Check | Tolerance | Result |
|---|---|---|
| 48 graphs, IDs equal the archive; slice has 12 | exact | pass |
| Diagnostic first absolute hit = legacy `fail_step`; prefix bit-identical | exact, 384 inits | pass |
| Archive `n_ok`, mean fail step, survivor raw/LP ARI and AUC | ≤1e−12 (reported, not fatal) | 48/48 match |
| Discrete center recursion residual / max(1, max\|u\|) | ≤1e−10 | 9.2e−16 |
| Mean-free forces (projected, candidate, Mean, fixed Huber, projected OWA/odd_OWA): \|Σd F\|/Σd\|F\| | ≤1e−12 | 6.6e−15 |
| Same forces: \|m(t)−m(0)\| / max(1, max\|u\|) | ≤1e−10 | 1.4e−13 |
| Projected = re-centered original, t ≤ 50 | ≤1e−12 relative | 4.6e−14 |
| Projected = re-centered original, full horizon, vs a twin started 1 ulp above u₀ | ≤10× twin deviation | 4.8× (max) |
| Projected = re-centered original, full horizon | 1e−8 (reported, not fatal) | **2.3e−8, exceeded** |
| Linear single-mode amplitude vs cos(ωnh), ω = 2 asin(h√λ/2)/h; Fiedler and top mode, 12 graphs, both h | ≤1e−9 | 1.0e−12 (off-mode 2.2e−14) |
| First bridge force, ±1 blocks, two 24-cliques, one 0.25 bridge | ≤1e−15 | Mean ∓0.5/23.25, fixed Huber ∓0.125/23.25, others 0 |

The single full-horizon tolerance of 1e−8 was fixed before the run. It was exceeded on late clique `huber_local` frames (maximum 2.3e−8). The deviation is ≤ 5e−14 up to t=50 and grows later. A run started one ulp above u₀ deviates from the original by the same order (up to 2.7e−8). The twin comparison and the early-horizon check were added after this was observed. The late deviation is rounding amplified by the dynamics, not an algebraic difference. The bridge-force startup is a sanity check only, not a partition score. The frequency check uses the Taylor startup, which is exact for a single linear mode.

## Files

| File | Contents |
|---|---|
| `center_drift.py` | the complete script |
| `replay_graphs.csv` | 48 graphs: archive vs replay `n_ok`, mean fail step, survivor ARI/AUC, match flags, failure / centered / growth / class counts |
| `replay_inits.csv` | 384 initializations: legacy and diagnostic first-hit steps, \|m\| and centered amplitudes at the crossings, peaks with their steps, values at T, centered-RMS ratio, class |
| `replay_summary.csv` | run and graph counts by family, bridge count and overall |
| `replay_trace.csv.gz` | m, a, max\|u\|, centered max and RMS at t = 0, 5, …, 250 and within ±10 steps of each first crossing |
| `slice_graphs.csv` | for each graph × configuration × h: failure and class counts, center checks, raw/LP readouts, plan ARI, graph class |
| `slice_inits.csv`, `slice_trace.csv.gz` | the same per-initialization diagnostics and traces for the slice and controls (±10 substeps around crossings) |
| `slice_summary.csv` | configuration-level aggregates |
| `slice_h_sensitivity.csv` | h=0.5 vs 0.25 for each graph × configuration |
| `slice_recenter_check.csv` | projection vs re-centered original and the 1-ulp twin, per initialization |
| `check_linear_frequency.csv`, `check_bridge_force.csv`, `checks.csv`, `summary.json` | reference checks and validation results |

## Limits

- Small development sample (48 graphs; the slice has 12). The eight initializations are repeated measurements on one graph, not independent samples.
- The class thresholds (2× RMS, 20, 1e8) are descriptive.
- Projection and the edge-symmetric candidate were each tested at the registry constants only.
- Bridge/internal contrast and ARI come from development graphs; they are not evidence of discovery and not a performance comparison.
- Same-force diffusion and the full §2 matrix were not run here and remain mandatory.
