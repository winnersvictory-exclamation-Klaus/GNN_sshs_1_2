# 003 — Local clipping preserves a given split; no demonstrated advantage from inertia

**Accepted as a development mechanism test, with one correction to the README: there are 12 near-tied Kruskal gaps, not zero.** Endpoint-local Huber preserved a supplied block contrast and showed selective bridge weakening from random starts. The weakening was stronger in diffusion, and the wave did not show a reproducible accuracy advantage over both same-force diffusion and spectral. No starts failed at either step size in this damped pilot. That clears the failure gate for these settings, but does not justify opening confirmation.

## Preservation is not discovery

With noiseless ±1 blocks, the endpoint-local neighbor median is zero at every bridge endpoint. The bridge flux is therefore `0.25 × EPS`, versus 0.5 for Mean and 0.125 for fixed Huber. Both local-Huber dynamics also preserved at least 87.8% of the initial signed block gap throughout the tested horizon with small added noise: none of the 16 noisy local runs reached half-gap, versus 14/16 for Mean and 12/16 for fixed Huber. Fixed clipping **did** delay gap loss (e.g. 231 vs 82.5 for the one-bridge clique wave), even though it made little difference to stiffness from random starts. The labeled-start and energy checks support the implementation and preservation result; they do not measure discovery.

## Random starts: selective weakening without a demonstrated wave benefit

At h=0.25, the graph-mean bridge-minus-interior stiffness for local Huber was **−0.043 early / −0.236 late in wave**, versus **−0.533 / −0.747 in diffusion**. In the late wave, bridge and interior means were 0.600 and 0.836, so weakening was not confined to bridges. Late diffusion had a much larger separation (0.218 vs 0.964). The coarse step gives the same qualitative result. Fixed Huber stiffness is close to one in both classes. The broad `|s|`-bin comparisons are descriptive: they do not eliminate within-bin amplitude differences, dependence among edge/time samples, or state-dependent differences in the endpoint thresholds.

These are paired low-pass plan-ARI differences on 48 graphs per cell, with h=0.5 / 0.25 shown side by side. There were 0/384 failed starts and 0/48 failed graphs in each of the 12 method-by-step cells, so the plan and survivor ARIs agree.

| Local Huber wave minus… | Early low-pass | Late low-pass |
|---|---:|---:|
| Mean wave | +0.023 / +0.006 | +0.273 / +0.366 |
| Local Huber diffusion | +0.073 / +0.051 | −0.064 / −0.044 |
| Spectral | −0.032 / −0.048 | −0.046 / −0.025 |

The early low-pass differences versus same-force diffusion have descriptive intervals spanning zero; the positive means come from cycles, where the wave still trails spectral on average. In raw scores, local Huber wave trails same-force diffusion by **0.36–0.40 ARI** at both windows and step sizes. Its late low-pass gain over Mean wave is real on this pilot but insufficient to establish a benefit from inertia. Mean wave’s late edge scores are small and its partition is worse; small magnitude alone cannot explain a change in a rank-based cut. Late low-pass wave cuts were step-sensitive on 11–14/48 graphs per force. In contrast, no late diffusion cut was flagged by the step-halving criterion; that criterion does not test sensitivity to floating-point ties.

One bounded lead outside the wave claim: local diffusion improves **early raw** ARI over Mean diffusion by +0.176 / +0.183, but late ARI differs by only −0.004 / −0.003. This is a development signal about the earlier readout, not proof of compute savings or improved final accuracy.

## Edge ranking, exact recovery, and the near-tie correction

I independently recomputed AUC and checked block connectivity using only edges scoring **strictly below the weakest bridge** in the saved NPZ. All 2,304 dynamic score vectors agree with the CSVs: the strict gap sign separates 1,282 exact recoveries from 1,022 nonrecoveries. In local-Huber low-pass wave, 13–16 of 48 graphs per cell have a negative gap despite a median AUC of 1.000; 2–4 negative-gap cases per cell still have AUC ≥0.99. The issue is an interior **connectivity bottleneck** above the weakest bridge, not just any interior edge ranked above a bridge.

The score table flags **12 near-tie rows** (`gap_near_tie=1`): three distinct SBM graphs in late local-Huber diffusion, each at both step sizes and both readouts. Their absolute gaps are about `6e−16` to `6e−14`; four rows are negative and eight positive. There are no *exact* ties, and none invalidate the strict-sign check on the saved scores. The README should distinguish this from numerical robustness: those three graphs do not show robust gap separation, and passing h-halving does not resolve that issue.

## Verification and decision

The pinned reproduction command regenerated every artifact byte-for-byte. I checked the 4,608 unique initialization records, the shared initial states in the checkpoint arrays, the paired means, the block-gap counts, and the saved edge scores as above. The stable damped pilot differs from the undamped replay in more than damping, so it does not isolate damping as the cause of the lower failure rate. The graph-paired bootstrap intervals are descriptive and are not the plan’s confirmation intervals.

**Do not advance the tested wave to accuracy confirmation.** The supported result is preservation from labeled starts and state-dependent bridge weakening from random starts, more pronounced in diffusion, without a demonstrated advantage from inertia. If the early diffusion lead is pursued, it needs a separate matched-cost test with an explicit stopping/readout rule rather than selection of a favorable window from this pilot.
