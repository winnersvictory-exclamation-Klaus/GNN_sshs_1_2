# 006a — Smoothed p-force passes implementation checks

**Accepted as an implementation and step-size diagnostic.** All 26 checks passed on the 5+7 clique and cycle graphs with one weight-0.25 bridge. At p=2 the implementation matches Mean within `1.12e−16`. Opposite weighted edge fluxes cancel before division by the original weighted degree, as required. For p=1.5 the largest finite-difference error against the negative degree-scaled potential gradient was `4.32e−11`; the largest small-amplitude tangent error against `sqrt(2) × Mean` was `7.31e−10`. Both are below the declared `1e−8` tolerances.

The largest tested screen values were **0.672** for wave (limit 4) and **1.344** for forward Euler (limit 2), both on the cycle at h=0.5. The screen bounds stiffness, not trajectory error or wave energy monotonicity. All eight integrations completed without a threshold failure. Across the saved frames the weighted center stayed within `1.2e−13` of its initial value.

| Graph | Max same-time energy difference, wave (fraction of initial energy) | Diffusion | Coarse wave energy increases above `1e−12` |
|---|---:|---:|---:|
| Clique | 5.72% | 4.48% | 47 |
| Cycle | 5.13% | 3.62% | 50 |

Both wave steps show long-run energy decay, but the coarse wave has nonmonotone saved energy; the largest single increase is 0.64% of initial energy on the cycle. The fine wave and both diffusion steps show no increases above the stated tolerance on these starts. Comparing only the final energies would miss the transient step-size differences.

The pinned command reproduced all four generated CSVs byte-for-byte. I checked all 4,008 trace rows for common-time coverage, recomputed energy totals and increments, and verified the screen and error tolerances. This validates the p-force helper for further development tests. It does not validate a partition method or static p-cut baseline, establish numerical convergence, or justify confirmation.
