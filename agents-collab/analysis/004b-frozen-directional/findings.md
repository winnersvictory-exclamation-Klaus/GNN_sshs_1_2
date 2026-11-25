# 004b — Keeping directional weights helps relative to symmetric freezing, but not reliably relative to unweighted continuation

**Accepted as a development control on the same 16 SBM graphs as 004a.** Freezing the original endpoint-specific coefficients preserves the adaptive force at the switch. This improves mean late low-pass ARI over symmetric freezing at both steps. It does not establish that freezing is sufficient: the directional fork beats continued adaptation and unweighted continuation at h=0.5, but trails both at h=0.25. Confirmation should remain closed.

## Paired results

Mean late low-pass ARI; 16 graph pairs per cell. All cuts were scorable and no directional starts failed (0/128 starts, 0/16 graphs per step), so plan and survivor ARIs agree.

| Continuation after adaptive warm-up | h = 0.5 | h = 0.25 |
|---|---:|---:|
| Directional freeze | 0.995 | 0.861 |
| Symmetric freeze | 0.926 | 0.807 |
| Original Mean force (unweighted) | 0.984 | 0.930 |
| Continued adaptation | 0.932 | 0.911 |

- **Symmetrizing accounts for part of the loss.** Directional minus symmetric is +0.069 / +0.053 (coarse / fine). The directional fork wins on 3 / 2 graphs and loses on 1 / 1, with the rest tied. Both use the same frozen endpoint multipliers, averaged only in the symmetric case; this is a direct comparison of the two frozen operators, not a general claim about conservative forces.
- **Neither the coarse-step match nor the low-pass gain generalizes across the tested readouts.** Directional minus adaptive is +0.063 / −0.051, and directional minus unweighted is +0.010 / −0.069. All directional low-pass cells have median AUC 1.000, yet exact recovery falls from 15/16 to 11/16 as the step is halved, with 1 / 5 negative Kruskal gaps and no near ties. Mean raw ARI is only 0.061 at both steps, −0.370 / −0.296 below continued adaptation. The fine-step raw mean ties both other forks, but three graph-level differences cancel; the exact recovery moves from seed 3 to seed 1 in the 16+32, one-bridge cases.

## Verification and limits

The pinned command regenerated every artifact byte-for-byte. I verified all 256 initialization records and 192 baseline joins, independently reconstructed the endpoint multipliers from the archived t=50 states, and confirmed that their arithmetic means exactly equal the saved 004a multipliers. The warm-up state error was zero, and the largest t=50 force error was `1.25e−16` against a `1e−12` tolerance. Saved edge scores reproduce the AUCs. The directional coefficients are fixed but asymmetric; the degree-weighted net force is already nonzero for some starts at the fork. Absence of threshold hits is not a conservation result.

The directional control narrows the interpretation of 004a: **symmetrization contributed to the low-pass deficit, but removing it did not produce a step-robust advantage over the simpler unweighted fork.** The warm-up itself is adaptive, so neither fork establishes a standalone linear method. The result is limited to the tested SBM development graphs, amplitude, switch time, and readouts.
