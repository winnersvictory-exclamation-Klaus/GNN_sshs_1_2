# Graph wave partitioning: development results

**Endpoint-local Huber clipping preserved a supplied split and selectively weakened bridges from label-free starts. In the matched pilot, that weakening was stronger in diffusion than in the wave. Neither the Huber wave nor the smoothed p-force follow-up demonstrated a reliable partitioning advantage over the corresponding diffusion and normalized spectral.** No confirmation sample was opened.

## What the mechanism tests showed

For noiseless +1 and −1 blocks, let Δ be the degree-weighted block-mean gap and let V₀, V₁ be the blocks’ summed weighted degrees. For an antisymmetric edge force, the initial gap acceleration is

`Δ″(0) = −(1/V₀ + 1/V₁) Σ_bridge w φ(2)`.

The initial velocity is zero, so damping does not enter this first acceleration. On a bridge of weight 0.25 the flux magnitude is 0.5 for Mean and 0.125 for fixed Huber (τ=0.5). For the tested noiseless clique and cycle starts, both endpoint-local thresholds equal EPS because most adjacent differences are zero; the initial flux magnitude is then 0.25·EPS. This is not a conservation identity for the endpoint-local force at later states.

With small noise added to the block starts, the local force kept at least 87.8% of the initial signed gap throughout the horizon. Fixed Huber also delayed gap loss, but the separation between internal mixing and gap loss depended on graph structure. At h=0.25 on the one-bridge clique, the Mean wave halved internal RMS at t=1.5 and the gap at 82.5; fixed Huber moved the gap crossing to 231. On the four-bridge cycle, the gap halved at 6.5 for Mean and 12 for fixed Huber, well before either halved internal RMS. These are preservation tests using known blocks, not discovery scores.

From random starts at h=0.25, late local-Huber bridge/interior stiffness averaged **0.600/0.836** in the wave and **0.218/0.964** in diffusion. Wave clipping weakened internal edges too. Fixed Huber was nearly linear in this pilot. Edge AUC alone overstated recoverability: an internal spanning-tree bottleneck above the weakest bridge can make the cut fail even at AUC near one. The diagnostic gap used truth only after cutting; near-zero gaps were not treated as robust separation.

## Accuracy, stability, and the frozen-weight test

Late low-pass plan ARI on the 48 development graphs. Dynamic methods share the same cutter; spectral is a separate baseline.

| Method | h=0.5 | h=0.25 |
|---|---:|---:|
| Mean wave | 0.496 | 0.423 |
| Local Huber wave | 0.768 | 0.789 |
| Same-force diffusion | 0.832 | 0.833 |
| Normalized spectral | 0.814 | 0.814 |

The damped pilot had zero failed graphs at either step, but late low-pass wave cuts were sensitive to halving h. In the older undamped configuration, 69/384 starts on 30/48 graphs failed. The plan’s failure penalty lowered its low-pass ARI from 0.826 to 0.365. Center drift was necessary for each first absolute threshold hit given the observed relative motion; four runs later crossed the threshold after centering as well. The damped and undamped protocols differ in more than damping, so their failure counts do not isolate a damping effect.

On the 16 SBM graphs, an adaptive warm-up to t=50 followed by unweighted Mean retained the mean late low-pass gain over pure Mean at both steps, but did not reproduce the adaptive wave’s raw scores. Retaining symmetric frozen weights was worse than the unweighted fork at both steps. Retaining directional weights beat the unweighted fork by +0.010 at h=0.5 and trailed it by −0.069 at h=0.25. Rescaling starts to 0.5× or 2× changed none of the cuts and did not remove the step sensitivity. These forks inherit an adaptive start; they are not standalone fixed-weight methods.

## Smoothed p-force follow-up

The symmetric p=1.5 force (ε=0.5) passed the Mean-limit, gradient, and weighted net-force checks. On the same 48 graphs no starts failed, but the wave’s late low-pass ARI difference from the small-signal-matched linear wave changed from **+0.084** at h=0.5 to **−0.026** at h=0.25. Relative to its tangent stiffness, early wave bridge/interior stiffness was 0.965/0.963 at the fine step, and both classes were within 1e−7 of the tangent limit late. The wave therefore showed no mean bridge softening relative to interior edges early and little softening of either class late. Late p-diffusion matched the tangent-matched diffusion graph by graph at both steps and with both readouts.

| p-force follow-up, late low-pass plan ARI | h=0.5 | h=0.25 |
|---|---:|---:|
| Smoothed p-wave | 0.601 | 0.508 |
| Same-force diffusion | 0.836 | 0.836 |
| Approximate static p-cut (no time step) | 0.851 | 0.851 |
| Normalized spectral (no time step) | 0.814 | 0.814 |

The unsmoothed static p-Rayleigh threshold sweep found the exact minimum binary objective on both enumerable toy graphs. On the 48 development graphs, it recovered 39 splits exactly and exceeded late p-diffusion by +0.015 ARI, entirely from cycles. Its selected optimizer reported precision loss on 28 graphs, so the returned larger-graph cuts are an approximate baseline, not certified solutions. The static and dynamical methods use different objectives and readouts; this is not a matched-cost comparison.

## Evidence and reproducibility

- [Representative bridge/internal RMS checkpoints](../results/005-mechanism-figures/checkpoint_contrasts.svg) · [Graph failures and late ARI](../results/005-mechanism-figures/failure_and_ari.svg) · [Figure data and command](../results/005-mechanism-figures/README.md)
- [Graph-paired scores with failure flags](../results/003-bridge-selectivity/part2_paired_graphs.csv) · [Graph IDs and seeds](../results/003-bridge-selectivity/part2_graphs.csv) · [Settings and reproduction command](../results/003-bridge-selectivity/README.md) · [Block and energy curves](../results/003-bridge-selectivity/part1_block_curves.csv.gz)
- [Archive audit](../results/001-baseline-audit/README.md) · [Center-drift replay](../results/002-center-drift/README.md) · [Symmetric freeze](../results/004a-frozen-sbm/README.md) · [Directional freeze](../results/004b-frozen-directional/README.md) · [Amplitude check](../results/004c-amplitude/README.md)
- [p-force verification](../results/006a-pforce-check/README.md) · [Paired p-wave and diffusion experiments](../results/006b-pforce-dynamics/README.md) · [Static p-cut toy checks](../results/006c-static-pcut-check/README.md) · [Static comparison and optimizer diagnostics](../results/006d-static-pcut-comparison/README.md)

This evidence supports a finite-time mechanism, not a nonconstant equilibrium, a robustness or runtime claim, statistical equivalence, or exclusion of an advantage on new graphs. The old quasi-equilibrium rule accepted every negative control while isolating a single node; it was not used to support these conclusions.
