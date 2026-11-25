# Primary plan: endpoint-local clipping gets SBM partitions earlier

`PLAN-E.md` is the backup. It closes the wave question with a bounded negative and a mechanism map. This plan pursues the one development signal that looks like an advantage. It changes the candidate from the wave to **same-force diffusion**, which `연구계획.md` treated as a control. The claim type is §3 C (compute saving), not accuracy.

## The signal

From 003 at h=0.25, label-free random starts, 16 SBM graphs (ARI, exact recoveries out of 16):

| Diffusion force | Early [0,50] raw | Early low-pass | Late [125,250] |
|---|---:|---:|---:|
| Endpoint-local Huber | **0.984** (13) | **0.984** (13) | 0.990 (14) |
| Mean (linear) | 0.555 (8) | 0.798 (10) | 0.995 (15) |
| Fixed Huber | 0.423 (5) | 0.798 (10) | 0.995 (15) |

Late accuracy is the same for all three, and spectral scores 0.995. Endpoint-local clipping reaches the correct split within the early window, while linear diffusion needs the late one. The early bridge/interior stiffness of 0.34 vs 0.88 fits this: the force acts as an adaptive reweighting that weakens bridges early. Cycles show no gain (0.27 vs 0.15 early, 0.51 for both late), and cliques are trivial.

**Candidate claim:** on community-structured graphs, endpoint-local clipped diffusion with a label-free stopping rule gives a two-way partition as accurate as linear diffusion, using ≥20% fewer edge operations.

**Main threats, each with a test below:**
1. Linear diffusion run at a larger stable step might arrive just as early.
2. The 4-bridge, 24-node SBMs may be too easy.
3. Window-based readouts can't stop a real run.
4. Lanczos spectral might use fewer operations than any diffusion.

## Development jobs (003 graphs and starts; no new seeds)

**010a: ARI against time, with a speed control.**
- Run local-Huber, fixed-Huber and Mean diffusion with 003 settings at h=0.25 and 0.5. Add Mean diffusion at its largest stable forward-Euler step (h≈1.8/λ_max, with λ_max computed per graph), so a faster linear method has its chance.
- Every 10 steps, cut on the edge RMS over a trailing window of fixed physical length, raw and low-pass.
- Record ARI curves, the first time ARI ≥0.8 holds for 3 checks in a row (label-based, descriptive only), and cumulative edge operations.
- *Gate:* on SBM, local Huber must reach the target earlier than **both** Mean diffusion variants on most graphs, at both h. If it doesn't, the effect was step speed: stop and switch to PLAN-E.

**010b: harder community graphs.** Ask graph-maker for two-block degree-corrected SBMs.
- Inter-block edges: 4, 16, 48, or a p_out sweep up to the point where spectral ARI falls below about 0.9.
- Sizes 24+24 and 16+32, graph seeds 0–3.
- Rerun the 010a curves and add spectral.
- This shows where the early gain persists, whether local clipping costs late accuracy, and whether any accuracy gain over spectral appears near the hard end. Any such gain is reported as a secondary result and not chased.

**010c: label-free stopping rule and full cost.**
- *Rule:* stop at the first check where the q=2 cut has been unchanged for K consecutive checks and each side has ≥10% of the nodes. K ∈ {3, 5}, chosen on the development graphs and then frozen.
- Apply it to every diffusion variant. Report the stop time, ARI at the stop, and total edge operations for all 8 starts, including scoring and cutting.
- *Reference:* scipy `eigsh` normalized spectral at default tolerance, counting matrix–vector products. It's reported side by side, but the claim is made within the iterative, local-message-passing class unless the operation counts favour clipping.
- *Gate to confirmation:* on the development SBM families,
  - ARI at the stop is within 0.02 of Mean diffusion's late ARI;
  - edge operations are ≥20% lower than the cheaper of the two Mean-diffusion variants under the same rule;
  - both hold at both h;
  - no failed starts.

Otherwise stop and switch to PLAN-E.

## Confirmation (only if 010c passes)

**011a: lock.** Fixed before any confirmation run:
- the graph families and difficulty levels from 010b where the gate held;
- the stopping rule and K;
- the step;
- the comparators: Mean diffusion at the 003 step and at its largest stable step, plus spectral as a reference.

New graph seeds 1000–1031 are used only after test-maker confirms they appear in no archive. Use a fresh init stream `default_rng([11, …])`. The endpoints are the paired ratio of edge operations and the ARI difference at the stop, with 95% paired bootstrap intervals over seeds and Holm correction across families.

*Decision:*
- **Saving:** the upper bound on the operation ratio is ≤0.8, the lower bound on the ARI difference is ≥−0.02, and the fraction of graphs reaching the target drops by no more than 5 percentage points.
- **Bounded negative:** the interval on the ratio stays above 0.8.
- **Otherwise undecided.**

**011b: one run, then a report.** Report the claim limited to the confirmed families and difficulty range. Cycles are named as the case where it fails, and a spectral comparison is included whichever way it falls. The 008a/b mechanism map from PLAN-E is reused to explain where the gain appears: early bridge weakening only matters when there are enough bridges to slow linear diffusion's gap decay, and too few to keep the cap from engaging.

## Switching to PLAN-E

If either development gate fails, or confirmation ends negative or undecided, run PLAN-E unchanged. Its confirmation seeds become 2000–2031 so that the two confirmations never share a graph. Work on PLAN-F uses only development graphs, so it doesn't contaminate PLAN-E. The study can end with both: a compute-saving result for clipped diffusion, plus the bounded wave negative and the mechanism map.
