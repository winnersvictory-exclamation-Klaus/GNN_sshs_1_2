# 008 — ARI vs cumulative edge operations (diffusion variants and label-free-damped wave)

Development only: the 48 003 graphs and their 8 saved starts (`part2_checkpoints.npz`), no new seeds, no tuning. Reuses 003's graphs, `huber_local` force, `rx.lowpass`, stable Kruskal `rx.cut_partition` (q=2). ARI is descriptive.

## Reproduce

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/008-ari-vs-ops/ari_vs_ops.py --out /tmp/008-rerun
```

About 40 s on one core. Output is written to `<out>.tmp` and renamed, so it appears atomically.

## Design

- Force `F(u) = -L_rw u` (mean, degree-normalized). Diffusion: `u+ = u + hF`. Wave: `u'' + g u' = F` with the 003 centered-damping scheme and zero initial velocity. Failure: any `|u| >= 20` or nonfinite zeroes that graph's plan ARI from then on (checked at every substep). T = 250.
- Methods (`config`):
  - `diff_h0.25`: mean diffusion at 003's h.
  - `diff_fast`: mean diffusion at h = 1.8/lmax (stability limit 2/lmax), the denominator of the ratios.
  - `huber_local_h0.25`, `huber_local_h0.5`: endpoint-local Huber diffusion.
  - `wave_gstar`: mean wave at h = 1.8/sqrt(lmax) (stability limit 2/sqrt(lmax), independent of g), g* = 2·sqrt(lam2_est).
  - `wave_0.5gstar`, `wave_2gstar`, `wave_g0.1`: sensitivity runs at the same h.
  - Spectral reference: `eigsh` (k=2, LA) on the normalized adjacency, ARI from 1-D 2-means of `v/sqrt(d)`, matvecs counted.
- lmax: 50-matvec power iteration on `I - N`. lam2_est: 30-matvec Lanczos (full reorthogonalization) on `N` with the trivial eigenvector projected out; `lam2 = 1 - largest Ritz value`. Dense `eigvalsh` is used only to audit these (`graphs.csv`), never to pick steps.
- Checks every 10 steps. Score: per-start edge RMS over the trailing 50 time units (all available samples before t = 50), median over the 8 starts; raw, or low-pass first (cutoff 0.04 cycles per time unit = 003's 0.02 per 0.5-spaced frame, applied inside the window). Same cutter for all methods.
- Target: first check that begins 3 consecutive checks with ARI ≥ 0.8 (`ops_target` is at that first check; `ops_target_confirm` at the third).
- Ops counting (per graph, all 8 starts): `ops_total = 8·m·(force evals + scored steps) + setup·m`. Each step is charged 1 op per edge for one force evaluation and 1 per edge for scoring (edge-difference accumulation). Setup is the matvec count × m (power iteration 50; Lanczos 30 for the g*-based waves). So `ops_total = 16·m·steps + setup·m`. `ops_force` is force only. Local Huber's per-node median sort is not counted extra, which understates it. Spectral ops are `matvecs·m` for one run, not ×8, and are in `graphs.csv`.

## Files

`graphs.csv` (spectral ARI/matvecs, eigenvalue estimates vs dense), `curves.csv.gz` (graph × config × readout × check: t, ARI, ops, failures), `targets.csv` (graph × config × readout), `reach_summary.csv`, `curve_summary.csv` (mean ARI at fixed ops-per-start-edge budgets), `paired_ops_ratio.csv` (graph-paired ratios), `ari_vs_ops.py`.

## Results

Denominators: 48 graphs = 16 per family (clique, cycle, sbm) × 8 starts each; 768 graph/config/readout rows, all run. No failed starts or graphs in any config. The dense audit flags no unstable step (`stable_true`). lmax_est/lmax_true is 0.966–1.000. lam2_est/lam2_true is 1.0–2.57, overestimated on cycles, so g* is somewhat too large there. Spectral: ARI ≥ 0.8 on 36/48 (clique 16, sbm 16, cycle 4).

Ratios are ops-to-target against `diff_fast`, over graphs where both reach; ratio > 1 means the numerator is costlier. Reached counts (of 16 per family; cycles are 8 for every config except where noted):

- `diff_fast`: all 16/16 clique and sbm and 8/16 cycle (both readouts); median ops per start-edge 26 (clique), 26 lp / 116 raw (sbm), 166 (cycle).
- `wave_gstar`: 16/16 clique, 16/16 sbm, cycle 8/16 lp and 2/16 raw.
- The sensitivity waves reach 27/48 (g = 0.1, 0.5g*) lp and 12–14/48 raw, against 40/48 for g* and 2g*.

| family | readout | numerator | both | only num | only den | neither | median ratio | geomean | ARI@target num/den |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| clique | lp | wave_gstar | 16 | 0 | 0 | 0 | 1.14 | 1.14 | 1.00/1.00 |
| cycle | lp | wave_gstar | 8 | 0 | 0 | 8 | 0.96 | 0.83 | 1.00/1.00 |
| sbm | lp | wave_gstar | 16 | 0 | 0 | 0 | 1.14 | 1.16 | 0.98/0.98 |
| clique | raw | wave_gstar | 16 | 0 | 0 | 0 | 1.14 | 1.16 | 1.00/1.00 |
| cycle | raw | wave_gstar | 2 | 0 | 6 | 8 | 1.14 | 0.89 | 1.00/1.00 |
| sbm | raw | wave_gstar | 16 | 0 | 0 | 0 | 1.38 | 1.32 | 0.98/0.98 |
| clique | lp | huber_local_h0.25 | 16 | 0 | 0 | 0 | 0.76 | 0.91 | 1.00/1.00 |
| cycle | lp | huber_local_h0.25 | 8 | 0 | 0 | 8 | 3.26 | 3.42 | 1.00/1.00 |
| sbm | lp | huber_local_h0.25 | 16 | 0 | 0 | 0 | 2.29 | 1.98 | 0.97/0.98 |
| clique | raw | huber_local_h0.25 | 16 | 0 | 0 | 0 | 0.76 | 0.81 | 1.00/1.00 |
| cycle | raw | huber_local_h0.25 | 8 | 0 | 0 | 8 | 2.87 | 2.51 | 1.00/1.00 |
| sbm | raw | huber_local_h0.25 | 16 | 0 | 0 | 0 | 0.56 | 0.61 | 0.97/0.98 |

`paired_ops_ratio.csv` also has `huber_local_h0.5`, `diff_h0.25` and the three sensitivity waves against `diff_fast`, plus `all`.

Limitations: ratios are quantized by the 10-step check interval (20 ops per start-edge), so many ratios take the same few values (e.g. 1.14 = 30/26.25). The same 8 cycle graphs are "neither reaches" for the diffusion and wave configs in the tables. Target time is the first hit of a 3-check streak, not time to stay above. Scoring is charged as 1 op per edge per step regardless of readout, and low-pass FFT cost is not counted. With lmax near 1-2 the wave step (1.8/sqrt(lmax)) is not much larger than the fast diffusion step (1.8/lmax), so their per-step cost is similar.
