# 001 — Historical baseline audit: findings

**Accepted as a development-only reanalysis.** The historical low-pass advantage of `huber_local` wave over Mean wave and same-force diffusion does not survive the plan’s graph-level failure rule. The method failed in **69/384 initializations (18.0%)**, affecting **30/48 graphs (62.5%)**; every affected graph had some surviving runs. This is well above the plan’s 5% graph-failure gate. Do not advance this exact configuration to accuracy confirmation. The audit does not rule out the mechanism or establish a cause of instability.

## Paired partition results

Each difference is `huber_local` wave minus the comparator on the same graph and with the same readout; spectral is unchanged between readouts. All cells use 48 paired graphs. Historical available-case and zero-fill report policies coincide here because even the partially failed rows have both ARIs.

| Readout | Policy | Candidate ARI | vs Mean wave | vs `huber_local` diffusion | vs spectral |
|---|---|---:|---:|---:|---:|
| Raw | Historical | 0.3290 | +0.1550 | −0.4471 | −0.4997 |
| Raw | Plan | 0.2163 | +0.0423 | −0.5598 | −0.6124 |
| Low-pass | Historical | 0.8256 | +0.0740 | +0.0304 | −0.0031 |
| Low-pass | Plan | 0.3653 | −0.3862 | −0.4299 | −0.4634 |

Low-pass improves the candidate’s own mean even on the 18 graphs where all eight runs survived (+0.3974 over raw). The improvement is therefore not confined to graphs scored on survivors. But the candidate’s low-pass edge over Mean wave is absent on those 18 graphs (−0.0046); the 30 partially failed graphs account for the positive aggregate comparison (+0.1212 within that stratum). These strata are selected by the candidate’s outcome and describe the archive, not what the failed trajectories would have scored. Also, fixed Huber wave has no failed runs but only 0.5214 low-pass ARI, so stability alone did not deliver good partitions.

| Family (16 graphs each) | Candidate failed runs / 128 | Graphs with any failure / 16 | LP vs Mean, historical → plan |
|---|---:|---:|---:|
| Cycle | 47 | 15 | +0.1805 → −0.2785 |
| Clique | 3 | 3 | 0.0000 → −0.1875 |
| Degree-corrected SBM | 19 | 12 | +0.0417 → −0.6927 |

Under plan scoring, low-pass also trails same-force diffusion and spectral in every family and every bridge-count stratum. The positive pooled raw difference against Mean is descriptive and does not establish a wave advantage: raw trails same-force diffusion and spectral in every family. The full strata are in the [paired summary](../../results/001-baseline-audit/t310_pair_summary.csv) (12 graphs per bridge count) and [failure summary](../../results/001-baseline-audit/t310_failures.csv).

## Verification and limits

- Reran `audit.py` into a temporary directory: all five CSVs were byte-identical. Independently recomputed the paired means and failure counts from the full archived `t310_partition.csv`. There are 48 graph keys, each with 60 dynamic rows and one spectral row. ARI comes from one cut of median edge scores over surviving runs, not eight separate ARIs. Spectral has no run denominator.
- Checked the launchers and imported readout code: both cut on frame 250 through 500; low-pass filters all 501 frames first at 0.02 cycles/step. The legacy failure predicate excludes an updated state exactly at amplitude 20 as well as nonfinite states. The plan column reweights these archived outcomes; it does not apply a new threshold. The rank and t310 samples have disjoint recorded graph keys and different designs, wave speeds and initialization RNG offsets, so they are neither paired nor a single pooled estimate.
- The rank archive reproduces spectral / Mean diffusion / Mean low-pass wave ARI of 0.814 / 0.827 / 0.679 across 384 connected evaluation graphs. Spectral ARI is 1.000 for clique, 0.995 for SBM and 0.447 for cycle. The original Trim low-pass ARI of 1.00 is available-case (160/384), including 32 partially failed cycle graphs, not clique only. The original Mean low-pass quasi-equilibrium rule accepts 16/16 negative controls; all 16 cuts peel off one node. That rule should not be used as evidence of a valid partition.
- These are development summaries, not confirmation estimates. The archives do not identify failed initializations, preserve trajectories, or recover per-init ARI or counterfactual scores. They cannot tell whether local clipping weakens bridges selectively or whether failures reflect center drift or growing relative modes. Keep any follow-up at the mechanism/stability stage (plan §§2–4), with matched readouts and same-force diffusion retained as controls.
