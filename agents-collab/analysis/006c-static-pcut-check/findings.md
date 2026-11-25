# 006c — Static p-cut sweep passes toy checks; optimizer convergence is not certified

**Accepted as validation of a numerical baseline.** The p=1.5 implementation uses no truth labels: it selects the lowest continuous objective among four starts, then the lowest binary objective among distinct thresholds of that vector. On both nine-node toys, its chosen cut minimizes the objective among the eight available thresholds and attains the smallest objective among all **255** nontrivial cuts modulo complements. This certifies the selected **binary cut on these toys only**, not the optimizer’s continuous solution or the sweep on larger graphs.

| Toy graph | Normalized Fiedler eigenvalue (p=2) | Selected p=1.5 binary objective | Binary regret |
|---|---:|---:|---:|
| 4+5 clique pair | 0.03015676 | 0.02385181 | 0 |
| 4+5 cycle pair | 0.04641811 | 0.03889933 | 0 |

The analytic gradient agrees with the centered finite difference within `3.79e−10`; the p=2 Fiedler and optimized quotient discrepancies are below `2e−16`. I independently reconstructed the two weighted adjacencies and checked all 510 binary objectives using `cut_weight × hypot(a,b)/(a*b)` for p=1.5, where `a,b` are the two weighted volumes. The maximum discrepancy from the helper was `2.23e−16`. I also checked the selected cut and threshold eligibility against the saved vectors. The pinned command regenerated all four CSVs byte-for-byte.

**Keep optimizer status separate from cut quality.** All eight p=1.5 BFGS trials reported precision loss (status 2), not convergence. The selected vectors have gradient infinity norms `2.69e−6` (clique) and `5.03e−7` (cycle). The continuous objectives across starts agree to nearly machine precision on each toy, but the residuals differ; selecting the smallest recorded objective does not select the smallest gradient residual. A larger-grid comparison may use this as an approximate unsmoothed static p-Rayleigh baseline, provided it carries optimizer statuses and residuals forward. These checks supply no performance result and do not make the static objective the same as 006a’s smoothed dynamical potential.
