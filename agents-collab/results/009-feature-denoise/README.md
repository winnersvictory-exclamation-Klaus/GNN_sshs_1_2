# 009 — Feature denoising: clipped/wave smoothing vs linear diffusion (development)

Question: does clipped or wave smoothing keep block boundaries better than linear diffusion when denoising a block-constant signal? Fixed development comparison, no tuning.

## Reproduce

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/009-feature-denoise/feature_denoise.py --out /tmp/009-rerun
```

About 1.5 min on one core. Output is built in `<out>.tmp` and renamed. Dependencies: `graph-maker/default-graphs/generators.py` (graphs), 003 (`make_force`: `mean`, `huber` with tau = 0.5, `huber_local` with kappa = 1), 008 (`operators`, `est_lam2`).

## Design

- Graphs (72, seeds 0-3): `sbm` (two_block, 24+24 and 16+32, bridges 1/4/16; 24 graphs), `sbm_het` (same grid; 24), `grid_neck` (a=5, w in {1,2,4}; 12), `barbell` (barbell_path k in {16,20}, L=4; 8), `ring4` (ring_of_cliques q=4, s=12; 4).
  - `sbm_het` uses sigma = 1.0. The generator's default sigma = 0.6 with p_in = 0.3 produces edge-for-edge the same graph as `two_block('sbm')`, which I checked, so I used 1.0 to make it a distinct family.
  - `ring_of_cliques` requires q >= 3, so q = 2 is not available. `ring4` signs cliques 0,1 as + and 2,3 as -, so the boundary is two ring edges.
  - grid_neck, barbell and ring4 are deterministic: seeds change only the noise draws. Bridge/boundary edges are the edges whose endpoints have different signs.
- Features: `x0 = m*s + sigma*z`, m = 1, sigma in {1,2,3}, `z = default_rng([9, graph_seed, draw]).standard_normal(n)`, 8 draws; the same z is used at all three sigmas. Not centered. As specified, graphs with equal seed and n share their draws.
- Methods: mean, fixed Huber (tau = 0.5, 003's value, which is not scaled to the noise) and endpoint-local Huber, each as diffusion (forward Euler) and damped wave (003 centered damping, zero initial velocity, g = 2 sqrt(lam2_est), 30-matvec Lanczos from 008). All use h = 0.25 (stable since lmax <= 2). Linear diffusion = `mean_diffusion`, the baseline.
- Times: steps {0, 1, ... 2048} on a rounded geometric grid (26 values + 0), i.e. t = 0.25 ... 512. Failure: nonfinite or |x| > 1e3 (0 occurrences).
- Metrics at every t, after subtracting the degree-weighted global mean: sign accuracy with threshold 0 (an exact 0 counts half); best-of-both-signs accuracy; MSE to m*s after the optimal scalar rescale (`m^2 (1 - cos^2)`, unweighted); contrast = mean |edge difference| over bridge edges / over interior edges (median over draws, medians over graphs when summarized; draws with interior mean < 1e-9 dropped). Per-graph values are means over the 8 draws.
- Selection: per (family, sigma, method), t* = argmax of mean threshold-0 accuracy over seeds 0-1 graphs (ties: earliest), frozen and applied to seeds 2-3 graphs (the test set). Test graphs: sbm 12, sbm_het 12, grid_neck 6, barbell 4, ring4 2 (36 total), 8 draws each. Both sign accuracy columns use the same t*.
- Baselines: x0 thresholded (t = 0); per-graph oracle t (label-using upper bound) for each method, and for linear diffusion. Paired differences are per test graph against linear diffusion, bootstrap 95% CI over graphs (2000 resamples).

## Files

`graphs.csv`, `curves_per_graph.csv.gz` (graph x method x sigma x t), `curve_summary.csv` (accuracy/MSE/contrast vs t by family, split, sigma, method), `selection.csv` (t* and training accuracy), `test_per_graph.csv`, `test_summary.csv` (per family and `all`), `paired_vs_linear.csv` (comparisons: `frozen_vs_frozen_linear`, `oracle_vs_oracle_linear`, `frozen_vs_x0`, `best_sign_frozen_vs_frozen_linear`), `feature_denoise.py`.

## Results (36 test graphs x 8 draws; 0 failures)

Mean sign accuracy. Every method lifts x0 (0.82 / 0.69 / 0.63) to 0.95-1.00.

| sigma | method | x0 | frozen t* | per-graph oracle t | linear oracle | MSE (frozen) | contrast (frozen, median) |
|---|---|---|---|---|---|---|---|
| 1 | mean_diffusion | 0.815 | 0.988 | 0.992 | 0.992 | 0.168 | 12 |
| 1 | mean_wave | 0.815 | 0.987 | 0.991 | 0.992 | 0.174 | 12 |
| 1 | huber_diffusion | 0.815 | 0.996 | 0.997 | 0.992 | 0.146 | 33 |
| 1 | huber_wave | 0.815 | 0.987 | 0.996 | 0.992 | 0.170 | 11 |
| 1 | huber_local_diffusion | 0.815 | 0.997 | 0.999 | 0.992 | 0.145 | 57 |
| 1 | huber_local_wave | 0.815 | 0.995 | 0.998 | 0.992 | 0.144 | 74 |
| 3 | mean_diffusion | 0.631 | 0.979 | 0.984 | 0.984 | 0.187 | 17 |
| 3 | mean_wave | 0.631 | 0.947 | 0.978 | 0.984 | 0.232 | 7 |
| 3 | huber_diffusion | 0.631 | 0.983 | 0.987 | 0.984 | 0.166 | 43 |
| 3 | huber_wave | 0.631 | 0.974 | 0.984 | 0.984 | 0.186 | 17 |
| 3 | huber_local_diffusion | 0.631 | 0.989 | 0.991 | 0.984 | 0.162 | 96 |
| 3 | huber_local_wave | 0.631 | 0.953 | 0.970 | 0.984 | 0.166 | 308 |

(sigma = 2 is in `test_summary.csv`.) Paired difference in frozen-t* accuracy vs linear diffusion, all test graphs, mean [95% CI]; oracle column is per-graph best t vs linear's per-graph best t:

| sigma | method | frozen diff | oracle diff |
|---|---|---|---|
| 1 | huber_diffusion | +0.009 [+0.004, +0.014] | +0.005 [+0.002, +0.009] |
| 1 | huber_local_diffusion | +0.010 [+0.004, +0.018] | +0.007 [+0.002, +0.012] |
| 1 | huber_local_wave | +0.008 [+0.001, +0.016] | +0.006 [+0.002, +0.011] |
| 1 | mean_wave | -0.000 [-0.003, +0.002] | -0.001 [-0.002, -0.000] |
| 2 | huber_diffusion | +0.005 [+0.002, +0.008] | +0.004 [+0.001, +0.007] |
| 2 | huber_local_diffusion | +0.008 [+0.003, +0.015] | +0.007 [+0.001, +0.013] |
| 2 | mean_wave | -0.028 [-0.073, -0.002] | -0.001 [-0.003, +0.000] |
| 3 | huber_diffusion | +0.004 [+0.000, +0.010] | +0.004 [-0.000, +0.009] |
| 3 | huber_local_diffusion | +0.010 [+0.001, +0.021] | +0.007 [-0.002, +0.016] |
| 3 | mean_wave | -0.032 [-0.067, -0.011] | -0.005 [-0.010, -0.001] |
| 3 | huber_local_wave | -0.026 [-0.050, -0.003] | -0.014 [-0.031, +0.002] |

All 15 rows (including the wave variants at sigma 1-3) and per-family rows are in `paired_vs_linear.csv`.

- Clipping adds a small, consistent gain over linear diffusion: +0.004 to +0.010 accuracy, with the CI excluding 0 for fixed Huber diffusion at all sigma and for local Huber diffusion at sigma 1-3 (the lower end is 0.001 at sigma 3). The ceiling is high: linear diffusion already reaches 0.98-0.99.
- Boundary contrast is much higher with clipping: local Huber diffusion about 5x (sigma 1) to 6x (sigma 3) the linear value (57-96 vs 12-17), fixed Huber 2.5-3x. Sign accuracy and MSE move far less than the contrast.
- Wave smoothing gives no gain over diffusion with the same force. Mean wave is equal at sigma 1 and below linear at sigma 2-3. The frozen-t gap is much larger than the oracle gap (e.g. -0.032 vs -0.005 at sigma 3), so its best time is less stable across seeds. Local Huber wave is worse at sigma 3.
- Per family, the wave and local Huber effects are not uniform: at sigma 3, local Huber diffusion is +0.010 on `sbm` (n = 12) and -0.027 on `ring4` (n = 2); see `paired_vs_linear.csv` for all strata. Some strata have only 2-6 graphs, so their CIs are crude.

Limitations: development graphs only, seeds 2-3 as the held-out half of a split on the same generators; fixed Huber's tau = 0.5 is not tuned to the noise scale; the wave horizon is only 512; contrast ratios are heavy-tailed and summarized by medians; sbm_het and ring4 settings differ from the request's literal defaults as described above.
