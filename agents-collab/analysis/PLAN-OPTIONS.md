# Candidate plans and rough odds

The probabilities are subjective. Each is the chance of a **confirmed positive conclusion** on fresh seeds, judged from the 001–006 development data and the prior literature. "Dev gate" means the cheap first job that decides whether to continue (about 1–3 small jobs each). For each plan, *Answers wave question?* says whether a success would say anything about inertia, the original research question.

| # | Plan | Dev gate | Confirmed | Answers wave question? |
|---|---|---:|---:|---|
| F | Clipped diffusion: earlier partitions, fewer edge operations (current `PLAN-F.md`) | 55% | 40% (in-class); 10% vs Lanczos | No |
| G | Wave with tuned damping as momentum: fewer steps than diffusion | 60% | 45% (vs diffusion); 10% vs Chebyshev/Lanczos | **Yes** |
| H | Brief clipped diffusion as an edge reweighting before spectral | 40% | 30% | No |
| K | Attributed graphs: boundary-preserving feature smoothing | 60% | 50% (clipped vs linear); 20% (wave vs clipped diffusion) | Partly |
| L | Robustness to a few spurious heavy edges | 35% | 25% | No |
| I | Hard or degree-heterogeneous SBMs: accuracy over spectral | 25% | 15–20% | No |
| J | Static p-cut beats spectral on ring-like graphs | 75% | 60% | No (known result) |
| M | Phase/trajectory readout of the wave | 15% | 10% | Yes |
| E | Backup: bounded wave negative + mechanism map (`PLAN-E.md`) | — | ~75% clean ending | Yes (negative) |

## F: clipped diffusion saves edge operations
- **Basis:** on SBMs, the early ARI is 0.98 for local Huber vs 0.56 (raw) / 0.80 (low-pass) for Mean. Late ARI is the same for both.
- **Main risks:** a faster linear step arrives just as early; harder graphs remove the gain; Lanczos needs fewer matrix–vector products.
- **If it works:** "With a label-free stopping rule, endpoint-local clipped diffusion matches linear diffusion's partition accuracy on community graphs using ≥20% fewer edge operations, because it weakens bridges early. No gain on ring-like graphs."

## G: inertia as momentum acceleration
- **Basis:** a damped wave is the heavy-ball method applied to the diffusion energy (Polyak). With damping γ ≈ 2√λ, a mode decays at a rate of about √λ instead of λ, and the stable step grows from about 2/λ_max to about 2/√λ_max. Internal mixing is slowest exactly where diffusion struggles: cycles, and SBMs with many bridges.
- **What's new:** every pilot so far used a fixed γ=0.1, which is heavily underdamped for most internal modes. This was never tested.
- **Design:**
  - Set γ without labels, from a short power-iteration estimate of the spectrum.
  - Compare against diffusion at its largest stable step, counting steps to a stable cut with the same stopping rule.
  - Report Chebyshev-filtered diffusion and Lanczos as the honest ceiling.
- **Main risk:** a large gain over plain diffusion that disappears against Chebyshev/Lanczos, which are also accelerated linear methods.
- **If it works:** "Inertia helps graph partitioning as acceleration, not as selectivity. A suitably damped graph wave reaches the same partition in O(√κ) instead of O(κ) steps relative to diffusion, with the largest saving on slowly mixing communities. The nonlinear force adds nothing beyond that." This is the most direct *positive* answer to the original research question.
- **Cost:** it shares the ARI-over-time infrastructure with F, so one job can carry both.

## H: clip-then-spectral reweighting
- **Basis:** late local-diffusion stiffness is 0.22 on bridges vs 0.96 on internal edges. 004a showed that weights frozen after a warm-up carried part of the gain on SBMs. Spectral scores 0.45 on cycles because cutting through the rings is cheaper than cutting the bridges, so lowering bridge weights could fix exactly that.
- **Main risk:** the warm-up weights are noisy on cycles, where the 003 bridge AUC was weak.
- **If it works:** "A few label-free steps of clipped diffusion produce edge weights that make normalized spectral recover community splits it otherwise misses (cycles and many-bridge SBMs), at no loss elsewhere." This is a practical preprocessing contribution.

## K: attributed graphs and oversmoothing
- **Change of task:** the initial values become noisy node features that carry label information, instead of random starts. The goal is to smooth within communities without erasing the boundaries between them, as in Perona–Malik edge-preserving diffusion.
- **Comparison:** linear diffusion oversmooths, and spectral can't use the features, so the comparison is meaningful in a way branch B wasn't. Hyperbolic graph PDEs (PDE-GCN, GRAND-style models) claim less oversmoothing, which gives the wave a real chance here.
- **Main risks:** it moves to a new task, so the earlier results don't carry over; it needs a feature-noise model from graph-maker; and it's closer to existing GNN literature.
- **If it works:** "Endpoint-local clipping preserves community boundaries in feature smoothing better than linear diffusion, with ARI or denoising gains at matched smoothing. A wave adds or doesn't add a further gain." This connects back to GNNs.

## L: robustness to spurious edges
- **Design:** add a few heavy, wrong inter-block edges (structural noise, not noise in the initial values). Clipping caps the flux through any single edge, and spectral has no such cap.
- **Main risk:** weighted spectral may already be fairly robust to a few edges, and clipping also slows real mixing.
- **If it works:** "Clipped diffusion degrades less than spectral and linear diffusion when a small number of high-weight spurious edges are added, up to a stated contamination level."

## I: harder or heterogeneous SBMs
- **Basis:** the static p-cut beat spectral by +0.011 on SBMs, and spectral can weaken near the detection limit or with strong degree heterogeneity. But late clipped diffusion matched linear diffusion graph by graph in 006b.
- **If it works:** "Near the detectability boundary or under heavy degree heterogeneity, clipped diffusion beats normalized spectral by ≥0.05 ARI in a stated range."

## J: static p-cut on ring-like graphs
- **Basis:** a development gain of +0.10 on cycles. This is close to a replication of p-spectral clustering (Bühler & Hein 2009), and the optimizer isn't certified.
- **If it works:** "A p=1.5 graph cut recovers ring-like two-block splits better than normalized spectral." It's reliable but low in novelty, and it's not our method. It's best used as a baseline or a report section.

## M: wave phase readout
- Cluster nodes by the phase or spectrum of their trajectories instead of edge RMS. This most likely reproduces spectral clustering.
- **If it works:** "The wave's dispersive signal gives a readout that beats edge-RMS cuts." Low odds.

## Recommendation
1. **One combined dev job for F + G.** Track ARI against time and edge operations for clipped diffusion, Mean diffusion at its largest stable step, and a wave with label-free tuned damping, under the same stopping rule. G answers the original question; F is the safer win. Together, the chance that at least one passes its gate is about 75–80%.
2. **H as a cheap add-on.** It reuses 004a's warm-up code and needs no new dynamics.
3. **K only if F and G both fail**, or if the GNN connection matters more than the original wave question. It means changing the task.
4. **E remains the guaranteed ending** whichever way the others go. Each confirmation uses its own seed block.
