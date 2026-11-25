# 004a — Adaptive warm-up helps; symmetric freezing adds no demonstrated benefit

**Accepted as a development control on the 16 SBM graphs.** Switching to the original Mean force after the adaptive warm-up retained the late low-pass ARI gain over pure Mean at both step sizes. Keeping symmetrized fixed conductances instead gave lower mean ARI than that unweighted continuation at both steps. The symmetric freeze nearly matched continued adaptation at h=0.5 but did not at h=0.25. This does not establish a standalone fixed-weight method or distinguish the effect of freezing from the effect of symmetrizing.

## Paired results

Mean late low-pass ARI on the same 16 graphs per cell. Each fork had **0/128 failed starts and 0/16 failed graphs** at each step size, so plan and survivor scores agree.

| Continuation or baseline | h = 0.5 | h = 0.25 |
|---|---:|---:|
| Symmetric freeze after adaptive warm-up | 0.926 | 0.807 |
| Original Mean force after adaptive warm-up | 0.984 | 0.930 |
| Continued adaptation | 0.932 | 0.911 |
| Pure Mean wave | 0.317 | 0.177 |
| Same-force diffusion | 0.990 | 0.990 |
| Spectral | 0.984 | 0.984 |

- **Warm-up state matters.** The unweighted fork improves on pure Mean by +0.668 / +0.752 ARI (coarse / fine), and matches or exceeds continued adaptation in mean low-pass ARI (+0.052 / +0.018). Thus continued adaptation or retained conductance weights were not necessary for the observed *late low-pass gain over pure Mean* in this experiment. The fork inherits both position and the previous substep, so this does not identify position versus velocity as the carrier of the effect. It also does not show an advantage over diffusion or spectral.
- **The symmetric freeze is not a robust substitute.** Its low-pass difference versus the unweighted fork is −0.059 / −0.122. At the coarse step it wins on 2 graphs, loses on 2 and ties on 12; at the fine step it wins on none, loses on 2 and ties on 14. Relative to continued adaptation its mean difference changes from −0.006 to −0.104. The means do not imply a uniform loss on individual graphs.
- **The result depends on the readout.** Both forks have identical per-graph raw ARIs, averaging 0.061 at both steps, against 0.431 / 0.357 for continued adaptation. Their raw edge scores and AUCs are not identical. Median low-pass AUC is 1.000 for both forks even though the symmetric fork has 3 / 5 negative Kruskal gaps and the unweighted fork has 3 / 3. The saved fork gaps have no near ties under 003's diagnostic tolerance.

## Verification and scope

The pinned command regenerated every artifact byte-for-byte. All 32 t=50 checks match the archive bit-for-bit. Both forks use the same current and previous substep states. The symmetric force uses per-start multipliers in [0.0498, 1] with the original degree denominator. I checked the 512 fork-start records, the graph-keyed baseline joins and paired differences, and independently recomputed AUC from the saved edge scores.

The control isolates fixed symmetric conductances **conditional on adaptive warm-up**. It does not show that freezing the original directional coefficients would fail, that adaptation before t=50 is optimal, or that the result generalizes beyond these development SBM graphs and amplitude. No confirmation is warranted from this comparison.
