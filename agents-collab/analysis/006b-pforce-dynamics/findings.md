# 006b — Smoothed p-force does not show a step-robust wave advantage

**Accepted as a development comparison.** With p=1.5 and ε=0.5, the wave’s late low-pass ARI advantage over its tangent-matched linear control reverses when the step is halved. It trails same-force diffusion and spectral in that readout at both steps. No starts failed and all cuts were scorable, so failure penalties do not explain these differences.

## Paired partition results

Mean p-wave minus comparator plan ARI on the same 48 graphs per cell. Entries show h=0.5 / 0.25.

| Comparator | Early raw | Early low-pass | Late raw | Late low-pass |
|---|---:|---:|---:|---:|
| Tangent-matched linear wave | −0.016 / +0.016 | −0.011 / +0.002 | −0.130 / −0.110 | +0.084 / −0.026 |
| Ordinary Mean wave | −0.027 / −0.005 | −0.034 / −0.035 | −0.101 / −0.075 | +0.105 / +0.085 |
| Same-force diffusion | −0.280 / −0.238 | +0.056 / +0.087 | −0.708 / −0.686 | −0.235 / −0.329 |
| Spectral | −0.532 / −0.511 | −0.089 / −0.089 | −0.685 / −0.663 | −0.213 / −0.306 |

The late low-pass gain over ordinary Mean is not sufficient evidence for nonlinear weighting: it does not survive the tangent-matched comparison at the finer step. The early low-pass gain over same-force diffusion is descriptive and does not beat ordinary Mean, the matched wave at both steps, or spectral. For diffusion, **graphwise late ARIs are identical** for p-force and its matched linear control at both steps and both readouts (mean 0.836; 39/48 exact recoveries per cell). Early raw diffusion differences reverse sign (+0.054 / −0.021), and early low-pass differences favor the matched linear control (−0.034 / −0.040).

## Mechanism and numerical checks

At h=0.25, early p-wave bridge/interior secant stiffness averages **0.965 / 0.963 of the tangent limit**: both classes soften similarly, with no mean bridge-selective weakening in this window. Both late wave means lie within `1e−7` of the limit. Diffusion shows slightly lower stiffness at bridges than at internal edges (early **0.990 / 0.996**; late **0.999369 / 0.999999**), without a late ARI difference from the matched control. These are graph-averaged summaries of unfiltered states, not a claim that individual edges have identical stiffness or that the earlier nonlinear evolution is irrelevant to later cuts.

The pinned command reproduced all generated files byte-for-byte. I reconciled the 3,072 unique start records, recomputed all 2,304 paired ARI differences, and verified class RMS and AUC against the 1,536 saved edge-score vectors. There were no flagged near ties or rows with both class RMS values below `1e−8`; strict gap signs agreed with exact recovery. Each method/dynamic/step cell had 0/384 failed starts and 0/48 failed graphs.

**Keep confirmation closed.** This pilot does not demonstrate an accuracy advantage from either inertia or nonlinear p-weighting at the tested settings. It does not test a static graph p-cut and supports no performance claim against one.
