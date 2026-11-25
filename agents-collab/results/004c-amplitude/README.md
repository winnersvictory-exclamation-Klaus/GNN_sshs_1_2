# 004c — SBM amplitude sensitivity

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/004c-amplitude/amplitude.py --out /tmp/004c-rerun
```

Uses the same 16 development SBM graphs and eight starts from 003, scaled by 0.5 or 2. Both step sizes and all 003 parameters are fixed. The adaptive wave branches at t=50 from the same current and preceding substep; symmetric, directional and unweighted forces reuse 004a/004b. Pure Mean wave and same-force diffusion start at t=0. No amplitude-1 trajectories or spectral labels are rerun. Archived amplitude-1 edge scores reproduce their recorded Kruskal gaps and survivor ARIs within `1e-12`.

Each updated substep checks nonfinite or `|u| >= 20` and separately `|u| >= 20*A`. Nonfinite values or `|u| > 1e8` censor a start; only the absolute-20 flag drives plan ARI and survivor selection. The common saved cadence and 251-frame late window [125,250] match 003; low-pass is the within-window FFT at 0.02 cycles per frame. Spectral ARI is copied unchanged. `cut_changed` compares labels up to swapping the two groups, not just ARI.

- `graphs.csv`: 768 graph/readout rows with thresholds, ARIs, AUC, gaps, near-ties, cut changes and discrepancies between `scores/A` and archived scores.
- `initializations.csv`: 3,072 start records; first threshold and censor substeps (`-1` means no hit), with physical threshold times. Readout does not duplicate these failure records.
- `paired_graphs.csv` / `paired_summary.csv`: ARI differences versus amplitude 1 and fork comparisons at the same amplitude. Means are over available pairs with denominators shown; each group has 16 eligible graphs.
- `edges.npz`: `(graph,h,amplitude,method)` selects a slice `edge_ptr[k]:edge_ptr[k+1]` of `i,j,bridge`. `scores` and `amplitude_1_scores` have axes `[readout, concatenated edge]`, with raw/LP order in `readout`.
- `epsilon.csv`: per-start normalized adaptive-force defect at t=0 and t=50. For the same normalized state, the local threshold at amplitude A, divided by A, differs from the amplitude-1 threshold by `EPS*(1/A-1)`. Clipping and the degree-weighted averaging bound `max|F(A*v)/A - F(v)|` by `EPS*|1/A-1|`; check allows `1e-14` rounding slack. This is a same-state comparison, not a long-horizon error bound.

**Result:** Neither threshold was hit (0/3,072 start records), every cut was scorable, and no cut changed relative to amplitude 1. All paired ARI changes versus amplitude 1 were zero. Maximum normalized edge-score discrepancy was `2.61e-10` (continued adaptive wave); pure Mean was bit-identical after normalization. The measured same-state force defect peaked at `5.00e-13`. The small departures from exact scaling are consistent with the endpoint epsilon and accumulated numerical error; the test does not establish exact scale invariance or behavior outside these amplitudes.
