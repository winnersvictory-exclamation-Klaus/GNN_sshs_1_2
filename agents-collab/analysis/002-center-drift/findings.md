# 002 — Center drift explains the first threshold hits; relative growth remains

**Accepted as a development stability diagnostic, not a partition result.** Degree-weighted center drift was necessary for all 69 first absolute threshold hits in the legacy replay. Relative motion nevertheless grew substantially in many runs, and in four it later crossed 20 on its own. Projection separates these effects without improving the cut. The edge-symmetric force avoided threshold hits on the 12-graph slice, but its partition scores depended on step size and did not consistently beat Mean. Keep confirmation closed.

## What caused the legacy failures?

The replay matches all 48 archived graphs on failure counts, mean failure step, and raw/low-pass survivor ARI and bridge AUC. The unzeroed diagnostic matches all first-hit times and pre-hit prefixes. There was no censoring in this replay.

For each failed initialization, I checked `centered_max_at_first_hit + |initial_weighted_center| < 20`. The inequality held for **69/69**: keeping the initial center fixed while retaining the observed relative motion could not have crossed 20 at those frames. The stronger bound using the peak centered amplitude over the entire horizon held for **65/69**; the other four had a later centered crossing. This bound does not assume that the two amplitude peaks have the same sign or occur together.

| Family (128 starts each) | Center-only by diagnostic rule | Absolute crossing with ≥2× centered RMS growth, but no centered crossing | Later centered crossing |
|---|---:|---:|---:|
| Cycle | 0 | 47 | 0 |
| Clique | 0 | 0 | 3 |
| Degree-corrected SBM | 6 | 12 | 1 |

The 2× ratio is descriptive, not an instability threshold: 131 of 315 starts with **no** absolute crossing also reached it, including all 81 non-failing cycle starts. Conversely, the four centered crossings are direct evidence of large relative motion on this horizon, not evidence of divergence as time tends to infinity.

## Fixed-horizon controls

These 12-graph results use Taylor startup at both step sizes, not the exact replay startup. Each cell below reports failed starts / failed graphs out of 96 / 12.

| Force | h = 0.5 | h = 0.25 | Descriptive low-pass ARI, coarse → fine |
|---|---:|---:|---:|
| Original `huber_local` | 17 / 8 | 12 / 6 | 0.877 → 0.847 |
| Projected `huber_local` | 1 / 1 | 0 / 0 | 0.877 → 0.847 |
| Edge-symmetric `huber_local` | 0 / 0 | 0 / 0 | 0.672 → 0.850 |
| Mean | 0 / 0 | 0 / 0 | 0.848 → 0.835 |
| Fixed Huber | 0 / 0 | 0 / 0 | 0.518 → 0.503 |

- **Projection changes the failure penalty, not the relative signal.** Subtracting a uniform force leaves edge differences unchanged in exact arithmetic because the original force depends only on those differences. In the saved slice, graphwise raw/low-pass ARI and AUC match exactly, and the raw/low-pass bridge and internal RMS medians match within `7.5e−10`. The plan-score gains are entirely due to fewer graphs being assigned zero. The remaining coarse-step failure is a genuine relative crossing; it does not appear at the fine step. The pre-set full-horizon recenter tolerance was **exceeded** (`2.26e−8` vs `1e−8`); early agreement and the post hoc one-ulp twin support roundoff amplification, not exact numerical agreement over the full horizon.
- **Edge symmetrization removes net force but reduces amplitude contrast.** Its weighted net-force residual is below `3e−16` relative. At h=0.5, median low-pass bridge/internal edge RMS falls from `0.498/0.024` for the original to `0.067/0.016` for the candidate. Bridge ranking is largely retained (median low-pass AUC 0.998 coarse, 1.000 fine), so it is too strong to say all bridge information was lost. It is also too strong to infer discovery: two coarse-step SBM cuts have ARI 0 despite AUCs of 0.994 and 0.995. Candidate low-pass ARI relative to Mean changes from −0.175 to +0.015 on the same 12 graphs, and 5/12 graphs are flagged as step-sensitive. Zero hits at these two step sizes do not establish energy conservation, general stability, or numerical convergence.
- **OWA controls separate drift from relative growth.** On the selected cycle, OWA reached the emergency limit for all eight starts even after projection, at both step sizes. On the selected clique, original OWA had 8/8 absolute hits without a centered hit, but projected OWA still had 3/8 and 2/8 relative hits at the two step sizes. `odd_owa` had no hits on these two graphs, but its unprojected weighted center moved. Oddness does not imply pairwise conservation. The claim that every cut was scorable applies to the main slice configurations, not to the OWA cycle controls: all starts there were censored and descriptive cuts are missing.

## Verification and next decision

Reran the pinned reproduction command into a temporary directory; all generated artifacts matched byte-for-byte. Independently checked archived graph failure counts against the per-init replay, reconciled the 136 slice graph/configuration rows with their eight starts each, and recomputed failure penalties and the comparisons above. The linear-mode and first bridge-force reference checks passed. Diagnostic ARI in the slice uses **all uncensored** starts, including those that hit 20; it is not the survivor-only ARI in 001.

Treat projection as a diagnostic and edge symmetrization as an unconfirmed mechanism control. The useful next test is the plan §2 matched experiment: measure whether clipping selectively weakens bridges rather than all edges, separate labeled block preservation from random-start discovery, and retain same-readout Mean wave and same-force diffusion at matched budgets. Do not choose a confirmation setting from the better of the two step sizes on this slice.
