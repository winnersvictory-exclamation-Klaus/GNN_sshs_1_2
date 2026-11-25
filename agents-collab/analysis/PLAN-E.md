# Backup plan: closing the wave question (primary plan: `PLAN-F.md`)

Run this if PLAN-F fails a gate or ends negative or undecided. Its confirmation seeds are **2000–2031**, so they never overlap with PLAN-F's 1000–1031.

Development is done. No tested wave (local Huber, fixed Huber, p=1.5) beat both same-force diffusion and the same-readout linear wave, and none beat spectral. More force variants would just mean more searching. The remaining work turns the existing evidence into two of the endings in `연구계획.md` §5:

1. **Bounded negative performance result.** On a fresh, locked sample, test whether an accuracy gain of ≥0.05 ARI from the local-Huber wave can be excluded against its matched controls.
2. **Mechanism result.** Predict when a labelled split survives longer than internal mixing (bridge count, bridge weight, internal family, force saturation), and check whether that window explains random-start accuracy.

## What is dropped, and why

- **Branch B (robustness).** Initial values carry no label information here, and spectral ignores them, so a gain in robustness to init shocks cannot give a practical partitioning advantage.
- **Branch C (early availability).** The wave's early low-pass score beats diffusion (+0.05 to +0.07) but not Mean wave (+0.01 to +0.02) or spectral (−0.03 to −0.05). The early-raw lead of local *diffusion* over Mean diffusion (+0.18) doesn't address the wave question. It goes in the report as a future lead.
- **Out of scope** (per the §5 cut order): new forces, idea6, q>2, real networks, static p-cut certification, and the quasi-equilibrium detector. The detector accepted all negative controls, so it is formally removed from the metrics.

## Jobs (one small request each, run in order)

**007a: Step convergence (development graphs only).** Late low-pass wave cuts changed when h was halved on 11–14 of 48 graphs. Rerun local-Huber wave, Mean wave and local-Huber diffusion at h=0.125, using the 003 design: same graphs, starts, physical windows and readouts. Report how many graphs change cut and the ARI shift for each step pair (0.5→0.25→0.125). *Decision:* lock the coarsest h whose cuts match the next finer step on ≥45/48 graphs. If the wave still isn't converged at 0.125, the confirmation reports h=0.25 and h=0.125 side by side, and every conclusion must hold at both.

**007b: Locked protocol (I write it; test-maker only checks seed history).** Lock everything before any results exist:
- *Graphs:* the 003 grid ({clique, cycle, SBM} × {24+24, 16+32} × {1, 4} unit bridges) with new graph seeds 2000–2031, giving 384 graphs. Test-maker first confirms these seeds appear in no archive or earlier job. Fresh init stream `default_rng([7, 2, family_index, n0, n1, bridges, seed])`, 8 × U(−1, 1) per graph.
- *Methods:* all 003 settings, h from 007a. Candidate: local-Huber wave. Matched comparators: local-Huber diffusion and Mean wave. Reference: normalized spectral. Secondary arm: p-wave against p-diffusion and the tangent-matched linear wave.
- *Primary population:* cycle + SBM (256 graphs). Clique is reported only as an implementation check, since every method scores near 1 there and would dilute the differences.
- *Endpoints:* plan ARI with the 8-start failure rule, in two windows: late low-pass (the headline readout) and early low-pass (the wave's most favourable development cell). Intervals are 95% paired bootstraps over seeds (each seed carries its 8 graphs, 10,000 resamples), Holm-corrected over the advantage claims.
- *Pre-stated decisions:*
  - **Advantage:** the Holm-adjusted lower bound is >0 against *both* matched comparators, and the mean is ≥0.05, in a window. Then it's a performance ending, limited to the tested families and settings.
  - **Bounded negative:** in each window, the upper bound against at least one matched comparator is <0.05. The conclusion is that a ≥0.05 wave advantage is excluded under the tested conditions.
  - **Otherwise undecided:** report interval widths.
  - Spectral and the per-family breakdown are reported as secondary results and don't change the decision.
- *Power check:* development paired SDs were 0.21–0.33, which puts the half-width at about 0.03–0.04 for 256 graphs before seed clustering. The bootstrap gives the real number. If it exceeds 0.05, the result is reported as limited by precision, not rescued after the fact.
- *Sanity controls:* a disconnected two-block version of each family. Every method must recover the two components exactly; otherwise there's an implementation fault and confirmation stops.

**007c: Confirmation run.** Execute the 007b protocol exactly once (about 10 minutes, based on 003's runtime of roughly 1 min per 48 graphs). Publish graph-paired tables, failure records, edge scores and seeds. The results agent verifies the run against the protocol, with no reanalysis choices left open.

**008a: Linear mechanism map (labelled starts, not the confirmation sample).** For Mean, compare the predicted gap rate λ_g ≈ k·w·(1/V₀+1/V₁) with the within-block mixing rate λ_int, mapping each through the damped wave roots s²+γs+λ=0 and through diffusion. Predict the half-gap and half-internal-RMS times and check them against simulation. Sweep k ∈ {1,2,4,8}, w ∈ {0.0625, 0.25, 1} and families clique/SBM/cycle at 24+24. The output is a map of where T_gap/T_int ≫ 1, which is where a readout window exists.

**008b: Saturation on the map.** On the same sweep, measure fixed and local Huber against the prediction that the bridge flux is capped at w·φ(Δ). Then add the 003 random-start stiffness: in the wave, bridges reach 0.60 vs interior 0.84, against 0.22 vs 0.96 in diffusion. The aim is to show whether the cap actually engages from random starts. The expected statement: saturation lengthens the window only when bridge differences reach the clip scale, which the wave rarely does from random starts, whereas diffusion does.

**009: Final report and figures.** Produce the deliverables listed in §5:
- graph-paired confirmation table, including failures;
- representative bridge/internal curves (reuse 005);
- the mechanism map, plus stability and ARI by force and by h;
- seeds, settings and commands.

Cycles, where spectral scores 0.45 and the window doesn't exist (T_gap < T_int in 003 part 1), are stated as the boundary of the claim.

## Ending conditions

The study ends after 009 with one of three performance verdicts: advantage, bounded negative, or undecided. Each comes with the mechanism map, whatever the verdict. Looking at the confirmation results does not reopen force search or graph-family selection.
