# 003 — Bridge selectivity: preservation vs. random-start discovery (plan §2, development only)

Request: `test-maker/003-bridge-selectivity/request.md`. Does clipping preserve a given block contrast, discover the topology from random starts, or neither? Does any random-start signal need inertia rather than the same force in diffusion? Part 1 is a labelled-start mechanism check; part 2 is the random-start test. No confirmation seeds, no force variants, no tuning or selection of step, window or readout. Both h values are reported. The archived 001/002 outputs were not touched.

## Reproduction

From the repository root (`GNN_sshs_1_2/`):

```
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/003-bridge-selectivity/bridge_selectivity.py --out /tmp/003-rerun
diff -r -x bridge_selectivity.py -x README.md /tmp/003-rerun agents-collab/results/003-bridge-selectivity
```

- Environment: Python 3.15.2, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1.
- Runtime: about 1 min on one core. No timing claims are made, so no machine lease was used.
- Without `--out`, outputs are written next to the script. `--part 1` runs only part 1 and its checks.
- Gzip and NPZ members have fixed timestamps, and two runs gave byte-identical outputs.
- Gate on part 1: if any hard part 1 check fails, the script writes `BLOCKER.md` and exits 1 without starting part 2. Part 2 also exits 1 if a hard part 2 check fails. In this run, part 1 was run and inspected on its own before the full run.
- Code dependencies: graphs come from `C.make_graph` (= `rank_experiment.two_block`). The forces are the registered `wave_test` forces:
  - `mean`;
  - `huber`, with τ = 0.5;
  - `huber_local`, with κ = 1, an unweighted median over valid neighbours, and EPS = 1e-12.
- Readout dependencies: `C.edge_score`, `rx.lowpass`, `rx.cut_partition` (stable ascending Kruskal, q = 2), `rx.spectral_labels` and `C.bridge_auc`. The only new code is the integrator, the diagnostics, and an independent dense force implementation used for checking.

## Fixed design

| Item | Setting |
|---|---|
| Force F | Degree-normalized acceleration `F_i = (1/d_i) Σ_j w_ij φ_i(u_j − u_i)`. The same F drives both dynamics. |
| Wave | γ = 0.1 with centered damping and zero initial velocity: `u1 = u0 + ½h²F(u0)`, `u+ = (2u − (1−γh/2)u− + h²F(u)) / (1+γh/2)`. |
| Diffusion | Forward Euler: `u+ = u + hF(u)`. |
| Step size and horizon | h = 0.5 and h = 0.25, both to T = 250. Saved frames are t = 0, 0.5, …, 250 (501 frames); at h = 0.25 every second substep is kept. Stability margins: max h²λ_max(L_rw) = 0.50 and max hλ_max = 1.0 (at h = 0.5). |
| Failure | Checked on every updated substep: not all \|u\| < 20, nonfinite values included. Integration continues without zeroing. An init is censored at its first nonfinite state or once max\|u\| > 1e8. In this run there were no failures and no censoring anywhere, in either part. |
| Plan score | If any of the 8 inits fails, ARI is 0 for that graph/method/h. The survivor-only ARI (`ari_survivor`) is supplemental. Because nothing failed, the two are identical here. Unscorable cuts (no survivors, or `cut_partition` returns None) are flagged in `unscorable`; there are none. |
| Part 1 graphs | 24+24 clique and cycle, with 1 or 4 bridges of weight 0.25, graph seed 0 (4 graphs). |
| Part 1 noise | One U(−0.05, 0.05) vector per graph, `default_rng([3, 1, family_index, 24, 24, bridges, 0])` (family_index: clique 0, cycle 1, sbm 2). The same vector is shared by all methods and both h. |
| Part 2 graphs | 48: clique/cycle/SBM × 24+24 and 16+32 × 1 or 4 unit-weight bridges × graph seeds 0–3. |
| Part 2 inits | 8 × U(−1, 1) per graph, `default_rng([3, 2, family_index, n0, n1, bridges, seed])`. These differ from the t310 stream (`seed*1000+310`; a check confirms no shared values) and from the rank stream (`seed*1000+17`). One array per graph is reused by all 12 dynamic variants. Labels are used only for evaluation. |
| Windows | Early [0, 50] = frames 0–100 (101 frames). Late [125, 250] = frames 250–500 (251 frames). |
| Edge score | Per init, the RMS of `u_j − u_i` over the window, then the median over inits. Raw, and low-pass with the `rx.lowpass` FFT filter at fc = 0.02 cycles per saved frame, applied **inside each window**. This keeps bins 0–2 (early) and 0–5 (late), so the early score never sees late data. It differs from the archived filter, which ran over the whole trajectory. |
| Stiffness | φ(s)/s on saved frames, pooled over edge × frame × init, with limit 1 at s = 0. Mean and fixed Huber have one value per edge. `huber_local` keeps both directional ratios, `clip(s, ±τ_i)/s` and `clip(s, ±τ_j)/s`. Their endpoint mean is reported only as a summary; it is not a conservative force. Clip fraction = share of values below 1. |
| Kruskal gap | `min bridge score − max(bottleneck_0, bottleneck_1)`. The bottleneck of a block is the score at which stable ascending Kruskal on that induced block's own edges first connects it. The gap uses the same score vector as the cut. It is a truth-derived diagnostic and never touches the cut. |

## Key outcomes

### Part 1 — preservation of a given contrast (diagnostic only; no ARI, no discovery claim)

All reference checks pass (table below).
- **First bridge force.** At the +1 endpoint the force is −wb·min(2, τ)/d_i, and it is exactly zero off the endpoints. Mean flux per bridge is 0.5 and fixed Huber flux is 0.125. The degree factor is 1/23.25 on cliques and 1/2.25 on cycles.
- **Endpoint-local start is degenerate.** Every bridge endpoint has neighbour-difference median 0, so τ_i = EPS. The actual per-endpoint clip gives flux 0.25·EPS: F = −1.08e−14 (clique) and −1.11e−13 (cycle). The two endpoints happen to balance here (d_aF_a + d_bF_b = 0) because both have τ = EPS; that balance was observed, not assumed. The noiseless endpoint-local block start therefore barely moves (gap ratio ≥ 0.9999999999 throughout).

With noise added, the half-gap times (first saved time with gap ≤ ½·gap(0)) are identical at both h (h = 0.5 / 0.25):

| Graph | Mean wave | Mean diffusion | Fixed Huber wave | Fixed Huber diffusion | Local Huber wave | Local Huber diffusion |
|---|---:|---:|---:|---:|---:|---:|
| clique, 1 bridge | 82.5 | censored (r_T 0.80) | 231 | censored (0.94) | censored (1.00) | censored (1.00) |
| clique, 4 bridges | 26 | 195 | 65.5 | censored (0.77) | censored (0.99) | censored (1.00) |
| cycle, 1 bridge | 19.5 | 128.5 | 29 | 193 | censored (0.97) | censored (1.00) |
| cycle, 4 bridges | 6.5 | 24 | 12 | 49 | censored (0.88) | censored (1.00) |

*r_T* = gap(T)/gap(0). "Censored" means the gap never fell to half by T = 250.

- **Endpoint-local Huber preserves the given contrast.** Its minimum gap ratio is 0.88–1.00 in all 16 noisy runs. Mean and fixed Huber lose half the gap within the horizon in 14/16 and 12/16 runs, respectively. Fixed Huber slows the gap decay (231 vs 82.5 on the clique with 1 bridge), consistent with its flux cap.
- **Returns above half.** No crossing returned above half. In waves the gap does swing through zero; `max_abs_gap_ratio_after_half` reaches 0.5.
- **Internal RMS at the crossing.** On cliques it is 0.001–0.004, against 0.030 initially; internal mixing halves the noise by t ≈ 1–1.5, long before the gap halves. On cycles it is 0.11–0.32, i.e. 11–32% of the gap: in the cycle waves the gap halves before the internal RMS does. Within-block RMS includes the profile driven across the bridge, so on cycles under diffusion it can grow.
- **Energy.** Mean/fixed Huber wave E = K + V and diffusion V decay. At h = 0.25 no saved frame shows an increase. At h = 0.5, 4/16 wave runs show increases of ≤ 0.23% of E(0) between saved frames. Exact discrete monotonicity is not claimed.
- **Energy balance.** The continuous balance `E(t) + ∫ rate dt − E(0)` (trapezoid on saved frames) has a residual of at most 0.12·E(0) at h = 0.5. It shrinks about 4× at h = 0.25 (≤ 0.024). Curve-wise energy differences between h are ≤ 5% of E(0).

### Part 2 — random-start discovery

**Failures and gate.** There were 0/384 failed inits and 0/48 failed graphs in each of the 12 method × h cells, including endpoint-local wave (002's undamped version failed). The ≤5% gate is met. Zero failures is not a positive result on its own.

**Magnitude suppression vs relative selectivity (mechanism).** Mean stiffness of bridge / interior edges over the 48 graphs (h = 0.25; h = 0.5 agrees within 0.01):

| | Early, bridge / interior | Late, bridge / interior | Clip fraction, bridge / interior (early) |
|---|---:|---:|---:|
| Fixed Huber wave | 0.983 / 0.982 | 1.000 / 1.000 | 0.06 / 0.06 |
| Fixed Huber diffusion | 0.995 / 0.995 | 1.000 / 1.000 | 0.01 / 0.01 |
| Local Huber wave (directional) | 0.795 / 0.838 | 0.600 / 0.836 | 0.46 / 0.48 |
| Local Huber diffusion (directional) | 0.341 / 0.875 | 0.218 / 0.964 | 0.74 / 0.40 |

- **Fixed Huber is nearly linear here.** It suppresses neither magnitude nor selectivity: contrast is ≤ 0.005 in every family.
- **Local Huber suppresses all edges somewhat.** The median rule clips about half of each node's edges in each direction. That clip fraction of roughly 0.5 is structural, not specific to bridges.
- **Local Huber adds relative bridge selectivity, much more in diffusion than in the wave.** Bridge − interior contrast: diffusion −0.53 (early) and −0.75 (late); wave −0.04 (early) and −0.24 (late). By family, the late wave contrast is −0.35 (clique), −0.33 (SBM) and −0.02 (cycle); the late diffusion contrast is −1.00, −0.91 and −0.33.
- **|s|-bin check (`part2_sbins*.csv`).** Fixed Huber has near-zero observed contrast within the prespecified bins (≤ 0.007). Its stiffness is solely a function of |s|, but broad bins need not match |s| exactly. Local Huber diffusion keeps a large residual contrast inside bins, e.g. pooled −0.51 in [0, 0.25) early and −0.67 late. Local Huber wave keeps only −0.003 to −0.03 early and −0.11 late. Late-window |s| is almost always < 0.25, so the fixed bins resolve little there. Counts are pooled over edges, times and inits; they are not independent and not a causal test.

**Partitions (performance, descriptive).** Mean plan ARI over 48 graphs, as all / clique / cycle / SBM, at h = 0.5 → h = 0.25. Spectral is 0.81 / 1.00 / 0.46 / 0.98.

| Method | Early raw | Early LP | Late raw | Late LP |
|---|---|---|---|---|
| Mean wave | 0.31 / .88 / .06 / −.01 (same at both h) | 0.76 / 1 / .37 / .91 → 0.76 | 0.23 → 0.23 | 0.50 / 1 / .17 / .32 → 0.42 |
| Mean diffusion | 0.57 / 1 / .15 / .55 | 0.70 → 0.69 | 0.84 / 1 / .51 / .99 | 0.84 |
| Fixed Huber wave | 0.26 → 0.28 | 0.73 / 1 / .21 / .98 | 0.21 | 0.54 → 0.47 |
| Fixed Huber diffusion | 0.55 → 0.53 | 0.66 | 0.84 | 0.84 |
| Local Huber wave | 0.35 / .88 / .06 / .12 | 0.78 / 1 / .36 / .98 → 0.77 | 0.48 → 0.44 | 0.77 / 1 / .37 / .93 → 0.79 / 1 / .46 / .91 |
| Local Huber diffusion | 0.74 / 1 / .25 / .98 → 0.75 | 0.71 → 0.72 | 0.83 / 1 / .51 / .99 | 0.83 |

Graph-paired plan-ARI differences for local Huber wave (mean, descriptive bootstrap 95% CI over graphs). The sign agrees across h in every overall comparison:

| Comparator | Early LP (h = 0.5 / 0.25) | Late LP | Late raw |
|---|---|---|---|
| Mean wave | +0.02 [−0.06, 0.10] / +0.01 [−0.09, 0.09] | +0.27 [0.15, 0.40] / +0.37 [0.24, 0.50] | +0.25 / +0.22 |
| Same-force diffusion | +0.07 [−0.01, 0.16] / +0.05 [−0.03, 0.14] | −0.06 [−0.14, −0.00] / −0.04 [−0.11, 0.00] | −0.36 / −0.39 |
| Spectral | −0.03 [−0.09, 0.02] / −0.05 [−0.12, 0.01] | −0.05 / −0.03 (CIs cross 0) | −0.34 / −0.37 |

- **The local wave's late gain over Mean wave tracks Mean wave's late collapse.** Mean wave late LP is 0.17 (cycle) and 0.32 (SBM); the damped wave's late amplitudes are about 1e−4.
- **Same-force diffusion matches or beats the local wave late.** It also matches or beats it in late raw. The early LP edge of +0.05 to +0.07 has a CI that crosses zero. Cliques are 1.00 for every LP method, so all differences come from cycle and SBM.
- **Fixed Huber wave never beats Mean wave by more than 0.05.** Its CIs cross zero.
- **This pilot does not establish that inertia is needed.** Same-force diffusion does at least as well in the late-window averages and has stronger bridge selectivity.

**High AUC vs a usable cut.**
- **Gap sign decides exact recovery.** Across all 2,304 scored rows, a strictly positive Kruskal gap always gave exact recovery, and a strictly negative gap never did. There were no exact zero gaps, but 12 rows were flagged as near ties: three late local-Huber diffusion graphs, at both step sizes and in both readouts. Those gaps are not numerically robust.
- **Near-perfect AUC does not imply a usable cut.** Local Huber wave LP has median AUC 1.000, yet 13–16 of 48 graphs have a negative gap. In 2–4 of them AUC ≥ 0.99 despite the negative gap: at least one block has an internal connectivity bottleneck above the weakest bridge.
- **Same pattern elsewhere.** Every method's `n_high_auc_neg_gap` is in `part2_method_summary.csv`.

**h-halving.**
- **Graph level.** `part2_h_halving.csv` flags a graph as h-sensitive if failures, exact recovery or gap sign change, or if |ΔARI| > 0.1. Late-window LP wave readouts are h-sensitive on 11–14/48 graphs for every wave force, Mean included (late raw: 3–9/48). Diffusion is h-sensitive on ≤ 6/48 per readout (late window: 0).
- **Summary level.** The signs of the overall paired means agree across h. Six subgroup means flip sign (`sign_agrees_across_h = 0`). Treat late-window wave cuts as discretization-sensitive.

### Summary answer

Endpoint-local clipping **preserves** a given block contrast. It produces bridge-selective weakening from random starts, mostly in **diffusion**. Fixed Huber delays block-gap loss in some labeled starts but introduces little stiffness contrast from random starts. The tested local wave does not show a reliable advantage over same-force diffusion and spectral. It beats Mean wave in the late window, where Mean wave's readout deteriorates; the late wave readouts are step-sensitive. This is development evidence only. Confirmation stays closed.

## Validation (`checks.csv`, `summary.json`)

| Check | Tolerance | Result |
|---|---|---|
| `wave_test` forces vs independent dense implementation (block starts, all forces) | ≤ 1e−13 relative | 2.3e−16 |
| First bridge force vs hand value (60 endpoints × forces) | abs ≤ 1e−15 and rel ≤ 1e−12 | max rel 9.7e−16 |
| Sign at the ±1 endpoints | all | 60/60 |
| Flux per bridge: Mean 0.5, fixed Huber 0.125 | ≤ 1e−15 | 4.4e−16 |
| Force off the bridge endpoints | exactly 0 | 0 |
| Undamped single mode vs cos(ωt), ω = 2 asin(h√λ/2)/h (002 reference; Fiedler and top mode, 4 graphs, both h) | ≤ 1e−9 | 1.6e−12 |
| Added integrator checks: damped mode vs closed-form two-root solution; Euler mode vs (1−hλ)^n | ≤ 1e−9 | 9.2e−13 / 9.1e−14 |
| Part 2: 48 unique graphs; inits share no value with the t310 stream | exact | pass |
| Strict gap > 0 ⇒ exact recovery; strict gap < 0 ⇒ not exact (2,304 rows) | 0 violations | 0 / 0 |
| Σ d F at the block start for Mean / fixed Huber (reported only) | ≤ 1e−12 | 1.4e−17 |

## Files

| File | Contents |
|---|---|
| `bridge_selectivity.py` | the complete script |
| `part1_bridge_force.csv` | per endpoint × force: intra degree, bridges at the node, weighted degree, hand τ, hand/code force and flux, errors, sign, max \|F\| off the endpoints, observed bridge reaction imbalance, Σ dF |
| `part1_linear_modes.csv` | single-mode checks (graph, h, mode, λ, check type, deviations) |
| `part1_block_curves.csv.gz` | per graph × start (noiseless/noisy) × force × dyn × h × t (501 frames): block means, gap, gap ratio, within-block RMS (per block and pooled, degree-weighted), center, max\|u\|. Mean/fixed Huber also get kinetic, potential, energy, energy ratio, trapezoid dissipation and balance residual (wave velocity = central difference of substeps; the frame at T uses one extra, unsaved substep). |
| `part1_block_summary.csv` | per run: half-gap time and censoring, returns above half, internal RMS at the crossing, internal half time, energy summaries, failures |
| `part1_h_compare.csv` | h = 0.5 vs 0.25 at common times: gap ratio, internal RMS, half times, energy |
| `part2_graphs.csv` | graph metadata, λ₂, λ_max, h-stability numbers, spectral ARI |
| `part2_scores.csv` | per graph × force × dyn × h × window × readout (2,304 rows), plus 48 spectral rows: failure counts, scored set, class medians/means/RMS of the edge scores, ratio, AUC, Kruskal gap with both bottlenecks, tie flags, survivor and plan ARI, exact recovery |
| `part2_stiffness.csv` | per graph × force × dyn × h × window × measure: bridge/interior n, quantiles (5/25/50/75/95), mean, clip fraction, and bridge − interior contrasts |
| `part2_sbins.csv`, `part2_sbins_summary.csv` | \|s\|-bin tables (per graph; pooled by stratum, with counts) for fixed and local Huber |
| `part2_init_failures.csv` | all 4,608 inits: failure substep and time, censoring, center at failure (none failed) |
| `part2_gate.csv` | run and graph failure counts and the gate flag, by method × h and stratum |
| `part2_paired_graphs.csv`, `part2_paired_summary.csv` | graph-level paired differences and their summaries by stratum (all / family / sizes / bridges): mean, bootstrap CI, median, wins/losses/ties, and sign agreement across h. Metrics: `ari_plan` (primary), `ari_survivor`, `auc`. Rows with candidate `mean_wave` are a reference. |
| `part2_h_halving.csv` | per graph × method × window × readout: h = 0.5 vs 0.25 |
| `part2_method_summary.csv` | per method × h × window × readout × stratum: ARI, exact counts, AUC, gap counts, high-AUC/negative-gap counts, score magnitudes, stiffness summaries |
| `part2_edges.npz` | see the layout below |
| `part2_checkpoints.npz` | see the layout below |

**`part2_edges.npz` layout**
- Run axis: 576 runs = 48 graphs × 3 forces × 2 dyn × 2 h, described by `run_*` arrays (graph index, force, dyn, h, n_fail).
- Edges: graph g's edges are `edge_i/edge_j/edge_w/edge_bridge[edge_ptr[g]:edge_ptr[g+1]]`, oriented as `g.ei`/`g.ej`, with s = u_j − u_i.
- `scores`: float64, shape (576, window[early, late], readout[raw, lp], 620). These are the exact median edge scores used for the cut, AUC and gap; padding is NaN.
- `stiffness`: float32, shape (576, window, 4, 620) = per-edge window mean of the ratio on the `ei` side and on the `ej` side, and the clip fraction on each side.

**`part2_checkpoints.npz` layout**
- `inits`: shape (48, 8, 48).
- `truth`.
- `states`: shape (576, t ∈ {0, 50, 125, 250}, 8, 48), so stiffness and edge differences can be recomputed at those times.
- `first_fail_substep`.

## Limits

- **Small fixed design.** One damping value (γ = 0.1, a pilot value rather than an archive replay), one horizon, and two step sizes. The 48 development graphs are the units: the 8 inits are repeated measurements, not independent samples. The bootstrap CIs are descriptive.
- **Late wave amplitudes are tiny.** They have decayed to about 1e−4 (damping e^{−γt/2}), so late wave readouts sit at small but finite amplitudes and are h-sensitive.
- **Interior round-off in late diffusion.** Late diffusion interior scores reach round-off on 16 (Mean, fixed Huber) and 32 (local Huber) of 48 graphs. Ratios there are excluded from `score_ratio_median_above_floor` (floor 1e−8); AUC, gap and cut are unaffected.
- **Part 1 starts are diagnostic only.** The noiseless endpoint-local block start is EPS-dominated, so its preservation is mechanical. The noisy start sets τ from the noise level (about 0.03).
- **Scope of the comparisons.** No confirmation was opened. No force variant or readout was selected. Mechanism statements (stiffness, |s| bins) are separate from the performance comparisons, which are descriptive and development-only.
