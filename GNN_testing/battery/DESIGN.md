# Test battery for wave-update ideas 1–6

Scope: the six ideas registered in `../wave_test.py` (time schemes, forces, series, spread transforms).
Existing evidence is thin: 4 graph kinds × 3 seeds × bridges=2 × c∈{0.5,1}, metrics = fail/dead/ARI only.
This battery asks, per idea: **is it correct, is it stable and why, is it alive, does it partition,
does it pass signal, does it generalise, and is the gain anything other than diffusion/spectral in disguise.**

## 0. Ground rules (all agents)

- Python: `/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python` (numpy, scipy, sklearn, pandas, matplotlib). Install nothing.
- `../wave_test.py` and `../../GNN_sshs_1_2/rank_experiment.py` are **read-only**; import them (`import wave_test as wt`, `wt.rx`).
  New forces/schemes are registered from your own module with `@wt.force` / `@wt.scheme`.
- Write only inside `battery/<your dir>/`. Shared code lives in `battery/common.py` (read-only for idea agents;
  if it has a bug, work around it locally and report it).
- CPU: at most **2 worker processes** per agent (six agents share 8 cores). Budget ≈ 10 min wall-clock per test.
  If a grid does not fit, shrink it and **record the reduction** in the report.
- One script per test: `battery/ideaN/tNN_<slug>.py` → `battery/ideaN/results/tNN_<slug>.csv` (one row per graph × config, raw, not pre-averaged).
- Seeds: dev seeds 0–7 for any tuning/selection; held-out seeds 100–115 for final claims. Never select on held-out.
- Statistics: report mean with bootstrap 95% CI over graphs (2000 resamples); comparisons against a baseline are **paired by graph+seed**
  (Wilcoxon signed-rank, Holm-corrected within a test). Report n. No claim from a single seed.
- Report `battery/ideaN/REPORT.md`: per test — what was run (grid, n), the key numbers, verdict
  (`supports` / `refutes` / `inconclusive` / `not run`), and caveats. State plainly anything that failed, was skipped, or was reduced.
  End with a ≤10-line idea-level summary and the top-3 configurations (chosen on dev seeds) for the cross-idea phase.
- Do not tune thresholds to make a result look good. A negative result is a result.

Baselines in every comparative test: `wave+mean` (raw), `damp(k=0.1)+mean`, diffusion+mean (`rx.simulate(..., "diff")`), direct spectral (`rx.spectral_labels`).

## 1. Shared library `battery/common.py` (Phase 0)

**Graph zoo** — `make_graph(spec) -> (rx.G, truth|None, meta)`:
`two_block` (cycle/clique/sbm via rx; bridges 1,2,4,8; wb 1.0,0.25; sizes 24+24, 16+32, 8+40),
`barbell_path` (clique–path(len L)–clique), `grid_neck` (two a×a grids joined by a w-wide neck), `ring_of_cliques` (q=4,6),
`sbm_q` (q=3,4 blocks), `binary_tree` (depth d; truth = two root subtrees), `lollipop`, `watts_strogatz`,
`random_regular` (d=3,4,6,8; negative control), `er`, `shuffled` (rx), `sbm_het` (lognormal θ σ∈{0,0.3,0.6,1.0}),
size scaling n∈{48,96,192,384}. `zoo(name)` returns named lists: `"core"` (rx eval-like), `"wide"`, `"neg"`, `"scale"`.

**Initial conditions** — `init(kind, g, R, rng, **kw)`: `uniform`, `gauss`, `rademacher`, `student_t3`, `impulse` (one node),
`low_mode` (Fiedler + 10% noise), `high_mode` (top eigvec + noise), `block_offset` (±Δ/2 per true block + noise), modifiers `scale=s`, `dc=a`.

**Simulation** — `run(g, scheme, force, u0, c, steps, ps=None, pf=None)` wrapping `wt.simulate`; `run_diff(g, force, u0, eta, steps)` for the
diffusion update with any registered force (`u+ = u + eta·F`).

**Trajectory metrics** (all per realisation):
- `fail`, `fail_step`; `growth_rate` = least-squares slope of log AC-RMS over the last half (continuous version of "fail").
- `dc_drift` = mean_t-final(u) − mean(u0), plain and degree-weighted; `ac_final`, `dead`; **`fail_ac`** = fail recomputed after removing the node-mean each step (drift vs oscillation).
- `energy(X, g, c)` = ½‖v‖²_D + ½c²·uᵀ(D−A)u series → `energy_drift` = E_T/E_0, max/min.
- `mode_energy(X, g)` = projection on eigenvectors of L_rw (D-orthonormal): low (2 smallest nontrivial) / high (top quartile) energy fraction over time.
- `temporal_spectrum(X)`: centroid, Nyquist fraction (period-2 content).
- `lyapunov(g, scheme, force, u0, …)`: divergence rate of a 1e-8-perturbed twin.
- `activation` hooks where applicable (fraction of node-steps where a limiter/clip is active).

**Force property probes** — `force_props(g, force, pf, u)`:
oddness defect ‖F(u)+F(−u)‖/‖F(u)‖; shift defect ‖F(u+a)−F(u)‖; homogeneity defect ‖F(su)/s−F(u)‖ for s∈{0.1,10};
F(const)=0; momentum Σ_i d_i F_i (degree-weighted zero-sum); numerical Jacobian J (finite differences) →
D-symmetry defect ‖DJ−(DJ)ᵀ‖, max eigenvalue of −J (stiffness → predicted `c_crit = 2/√λmax`), most positive eigenvalue of sym(DJ) (anti-restoring directions).
`c_crit_empirical(g, scheme, force, …)` by bisection on c (fail_frac ≤ 0 at T=2000).

**Partition readouts** — `partitions(X, g, q=2) -> dict[name → labels]`: `edge_cut` (rx procedure), `edge_cut_lp` (lowpass fc=0.02),
`timeavg_2means` (2-means on time-averaged AC state), `traj_corr_spectral` (spectral clustering on node-trajectory correlation),
`fft_lowmode` (Sahai-style). Metrics: ARI, NMI, conductance, minfrac, plus **`edge_margin`** = median bridge-edge score / median intra-edge score
(continuous, needs truth) and `bridge_auc` (AUC of edge score for classifying true bridge edges). `ari_over_time(X, g, truth, window)`.
`q_select(score)` via largest gap, for q-unknown tests.

**Quasi-equilibrium** — `hold(X, g, labels, params, min_frac=0.1)`: rx criterion **plus** rejection of parts smaller than `min_frac` (fixes the known defect).
Fixed params ε=0.3, δ=0.1, ρ=2, W=50; also return the raw D_in(t), D_out(t) curves.

**Stimulus** — `stimulus(g, n0, runner, amps=(0.2,1,5), signs=(+1,−1), sigma=0.05, R=8)`: far_peak, transmit_ratio, arrival, SNR, noise gain,
and **detection AUC** (far-side peak response, stimulus vs noise-only realisations).

**Confounders** — `with_outliers(runner, frac, mode)` (stuck nodes / per-step heavy noise), `bridge_noise`.

**Stats/IO** — `boot_ci`, `paired_wilcoxon`, `holm`, `write_csv`, `pmap(fn, jobs, procs=2)`.

Self-test `battery/test_common.py`: mean force equals `-L_rw u`; energy is conserved to 1e-6 relative for wave+mean at c=0.5;
disconnected graph gives ARI 1; every zoo graph is connected (except disc controls); probes return zero defects for mean.

## 2. Idea 1 — linear damping in the time scheme (`damp`, `viscous`, `damp_visc`, `sched_decay`, `sched_grow`)

Hypothesis: damping removes blow-up of nonlinear forces without killing the wave before it is useful.

- **T1.01 theory check (mean).** Per mode λ the update is z² − (2−k−(c²+ν)λ)z + (1−k−νλ) = 0. Predict stability region and per-mode decay rate; compare with measured modal energy decay and empirical fail on k×ν×c grid (c up to 2.0). Any mismatch = bug.
- **T1.02 phase diagram.** k∈logspace(1e-3,0.5,10), ν∈{0,0.05,0.1,0.2,0.4}, c∈{0.25,0.5,1,1.4,1.8} × forces {mean,trim,owa,vw,median,gauss,absvw,symowa_ext,symowa_med,huber}: fail_frac, growth_rate.
- **T1.03 minimal rescue.** Per (force, graph kind): smallest k (damp) and smallest ν (viscous) with fail_frac=0 at T=2000, by bisection.
- **T1.04 alive-vs-stable.** At the minimal rescue value and 2×, 4×: useful lifetime = steps until AC-RMS < 5% of initial. Is there any window that is neither blown up nor dead? Pareto fail vs dead.
- **T1.05 drift vs oscillation.** For owa/vw: `fail` vs `fail_ac`, dc_drift(t). Control scheme `recenter` (subtract node-mean every step) and `recenter_dw` (degree-weighted). Does blow-up survive removal of the mean? (Tests the notes' unverified "upward bias = driving force" explanation.)
- **T1.06 mode selectivity.** damp vs viscous at matched total energy loss: low/high mode-energy fraction over time; ARI over time windows — best readout time and its ARI.
- **T1.07 schedules.** sched_decay k0×Ts, sched_grow kinf×Ts: fail_step distribution relative to Ts. Is instability transient (rescued by early damping) or persistent?
- **T1.08 raw-wave quasi-equilibrium.** hold (min_frac fix) on raw damped waveform vs undamped (known: 0%) vs diffusion.
- **T1.09 is it just diffusion?** As k→1: Spearman of edge scores between damp(k) and diffusion, ARI agreement; where on the k axis does the damped wave differ from diffusion at all, and is it better there?
- **T1.10 stimulus cost.** far_peak, arrival, SNR, detection AUC vs k, ν (amps 0.2/1/5, both signs).
- **T1.11 long horizon.** T=5000 on marginal settings (growth_rate within ±1e-3): fail_frac(T).
- **T1.12 amplitude scaling.** u0 ×{0.01,0.1,1,10} and dc=+3: invariance expected for homogeneous forces; flag violations.
- **T1.13 generalisation.** Top settings from T1.03 on zoo `wide` + `scale`, bridges 1–8, unequal sizes.
- **T1.14 negative controls.** zoo `neg`: false-partition rate (minfrac≥0.1 & hold>0), degenerate-cut rate.
- **T1.15 held-out statistics.** Top-3 configs vs baselines on seeds 100–115: ARI, edge_margin, hold, transmit; paired tests.

## 3. Idea 2 — nonlinear stabilisers (`vel_shrink`, `drag`, `limiter`, `cfl`, `energy`, `saturate`)

Hypothesis: state-dependent limiting prevents blow-up while leaving normal dynamics untouched.

- **T2.01 phase diagram.** Each scheme's parameter (6 log-spaced values) × c∈{0.5,1,1.4,1.8} × the 10 forces of T1.02.
- **T2.02 do-no-harm.** With mean force at c≤1 (already stable): trajectory deviation from plain wave, temporal-spectrum shift, energy loss, ARI change.
- **T2.03 activation audit.** Fraction of node-steps where the stabiliser is active, over time. Transient guard or permanently-on (= a different dynamic)?
- **T2.04 normalisation flaw check.** Schemes normalise by RMS(u0), which includes DC. Run u0, u0+3, 0.01·u0, 10·u0, impulse: outcomes should be shift/scale invariant; quantify violations. Add `*_ac` variants normalised by AC-RMS and compare.
- **T2.05 cfl estimator.** λ̂_t vs true λmax(−J) (mean: exact spectrum; nonlinear: numerical Jacobian along the trajectory): bias, lag, how often c_t<c.
- **T2.06 energy trigger.** `energy` uses RMS(u) incl. DC: with owa/vw, is it triggered by drift rather than oscillation? Variant with AC energy.
- **T2.07 minimal rescue + alive/dead Pareto** (as T1.03/T1.04).
- **T2.08 bounded but garbage?** For non-failing runs: Lyapunov rate, Nyquist fraction (checkerboard), high-mode fraction, amplitude limit cycle. Bounded chaos is not success.
- **T2.09 symmetry/conservation.** Run u0 and −u0 (trajectories should be negatives), node relabelling (equivariance), dc_drift.
- **T2.10 partition + hold** with all readouts and ARI over time.
- **T2.11 stimulus saturation curve.** far_peak vs amp∈{0.2,1,5,20}: limiter/saturate should clip large stimuli — quantify the H2 cost; detection AUC.
- **T2.12 combination with linear damping.** vel_shrink alpha×k grid; limiter+damp, saturate+viscous (register composites).
- **T2.13 impulse initial condition.** Single-node impulse (very non-uniform |v|), per-node stabilisers vs per-realisation ones.
- **T2.14 long horizon** T=5000; **T2.15 generalisation** (`wide`,`scale`); **T2.16 negative controls**; **T2.17 held-out statistics**.

## 4. Idea 3 — replacing the mean (forces A1–A3, B1–B4, C1–C5, odd_owa, odd_vw, repo trim/owa/vw)

Hypothesis: some non-mean force is stable under the wave update *and* gives better partition or transmission than mean.

- **T3.01 limit consistency (correctness).** huber/tanh τ→∞, amp β=0, gauss σ→∞, absvw β=0, symowa γ=0, edge_memory ρ=0 must reproduce mean to 1e-9; trim on degree-2 nodes equals mean.
- **T3.02 property table.** `force_props` for every force on 4 graph kinds × 3 states (random, mid-trajectory, near-consensus).
- **T3.03 which property predicts blow-up?** Regress empirical fail / growth_rate (wave, c∈{0.5,1}) on the property defects (oddness, momentum, D-symmetry, anti-restoring eigenvalue). Tests the "non-odd = driving" hypothesis directly; include `owa` vs `odd_owa` vs momentum-corrected owa (F − d-weighted mean of F).
- **T3.04 Jacobian-predicted vs empirical c_crit.** Bisection on c per force × graph; compare with 2/√λmax(−J) sampled along trajectories. Where linear theory fails, report non-normality.
- **T3.05 energy conservation.** Form-A forces (huber, tanh, amp, edge_memory) have a potential Φ=Σ w ρ(x): measure drift of ½‖v‖²+c²Φ at c∈{0.1,0.25,0.5}. Form-B (gauss, absvw, symowa, median, trim): secular drift expected — measure rate.
- **T3.06 instability localisation.** On cycle graphs trim equals mean except at the degree-3 bridge endpoints, yet 93% fail: record the first node to exceed amplitude 5 and its graph distance to a bridge endpoint, for all rank forces and graph kinds.
- **T3.07 parameter sweeps.** τ, σ, β, γ, κ, ρ, Ta (6–8 values each) × c × {wave, damp 0.1, diffusion}: fail, dead, ARI, edge_margin.
- **T3.08 discontinuity sensitivity.** 1e-10 perturbation of u0 → divergence time (rank forces are discontinuous); neighbour-order permutation invariance; tie handling with Rademacher u0 (all ties).
- **T3.09 mechanism test for clipping forces.** For huber*, gauss, edge_memory, amp_gated: per-step set of clipped/down-weighted edges vs true bridge edges — precision/recall/AUC over time; τ_t trajectories for adaptive forces. Does the force actually single out bridges, or just high-degree/noisy edges?
- **T3.10 partition.** All forces × {wave (if stable), damp 0.1, diffusion}: all readouts, ARI, edge_margin, bridge_auc, conductance, vs baselines; zoo `core`.
- **T3.11 quasi-equilibrium.** hold with min_frac fix on raw trajectories.
- **T3.12 stimulus (H2).** amps 0.2/1/5 × sign ±: far_peak, arrival, SNR, noise gain, detection AUC. Sign asymmetry is expected for owa/vw — quantify.
- **T3.13 partition–transmission plane.** ARI vs transmit_ratio vs noise gain for every force: is any force Pareto-better than mean on both?
- **T3.14 confounders.** Outlier nodes (5%, stuck or noisy) and bridge noise: ARI degradation relative to mean (robust aggregation's home turf).
- **T3.15 degree effects.** `sbm_het` σ sweep, `random_regular` d∈{3,4,6,8}: fail and ARI vs degree heterogeneity/low degree (trim at d=3 is the median).
- **T3.16 amplitude regime.** Fixed-threshold forces under u0 scaling and under decay (they become linear at small amplitude): late-time behaviour vs mean.
- **T3.17 cost.** sec/step, n∈{256,1024,4096}.
- **T3.18 generalisation / long horizon / negative controls / held-out statistics** (four sub-tests as in idea 1).

## 5. Idea 4 — high-order even series (space: `hi_acc`, `hi_alt`, `hi_geo`, `hi_geoneg`; time: `t6`, `t6neg`)

Hypothesis: adding higher even-order terms reshapes dispersion usefully. For mean base the force is −g(L)u, g(λ)=Σ a_k λ^k.

- **T4.01 linear theory (space).** g(λ) on each graph's spectrum (λ∈[0,2]); predicted stability 0<c²g(λ)<4 ∀λ; predicted vs empirical c_crit; sign changes of g → exponential growth modes.
- **T4.02 dispersion.** Modal frequencies from FFT of modal projections vs cos ω = 1 − c²g(λ)/2.
- **T4.03 order/weight sweep.** K=1..6, eps∈[−0.5,0.5] (21 values): stability map; generalised `series` force with arbitrary weight vector.
- **T4.04 spectral-gap reshaping.** Relative gap g(λ₃)/g(λ₂) and frequency separation of the cluster mode; **steps-to-partition**: smallest T at which `fft_lowmode` / `edge_cut_lp` reach ARI≥0.9, series vs plain wave. This is the plausible benefit — test it head-on.
- **T4.05 equal-cost comparison.** A K-term series costs K aggregations per step: compare with plain wave given K× the steps (equal aggregation budget) on ARI and steps-to-partition.
- **T4.06 nonlinear bases.** Series over trim/owa/vw/huber: `force_props` of the composed force (is G_k=(−1)^{k+1}F^{∘k} still restoring?), fail maps, Jacobian definiteness.
- **T4.07 time-series root analysis.** Characteristic polynomial per mode for t6 (degree 6): parasitic roots |z|>1 as a function of (e4,e6,c²λ). Predict whether *any* (e4,e6) is stable; confirm on a 2-D sweep including tiny/negative values (±1e-3…±0.5). Existing data show t4 100% fail — explain it.
- **T4.08 start-up sensitivity.** History init u(−k)=u0 vs a consistent start-up (plain wave for the first steps): does failure come from start-up or from parasitic roots?
- **T4.09 matched space–time orders.** Pairs chosen so temporal truncation error cancels the spatial series (modified-equation idea): dispersion error vs ω=c√λ.
- **T4.10 receptive field / stimulus.** A k-th order term reaches k hops per step: arrival time across the bridge, far_peak, SNR, detection AUC.
- **T4.11 partition + hold** for stable configs; **T4.12 damping rescue** (damp/viscous with unstable series); **T4.13 generalisation**; **T4.14 long horizon**; **T4.15 negative controls**; **T4.16 held-out statistics**.

## 6. Idea 5 — a single fourth-order term (space: `d4*`; time: `t4*`)

Hypothesis: one extra term (L+εL², or f''+εf'''') is the minimal useful modification.

- **T5.01 fine ε sweep (space).** ε∈[−1,1] (41 values) × c∈{0.25,0.5,1,1.4}: stability vs theory g(λ)=λ(1+ελ).
- **T5.02 dispersion measurement** as T4.02.
- **T5.03 mode-selective stiffness.** High-mode energy fraction of the edge-difference signal vs ε; effect on raw-wave hold and edge_margin (ε<0 softens high modes — does raw quasi-equilibrium appear?).
- **T5.04 nonlinear bases.** F−εF∘F for trim/owa/vw/huber: fail maps, props.
- **T5.05 time root analysis.** Quartic characteristic polynomial; sweep ε∈[−0.25,0.5] including **ε=−1/12** (cancels the leading time-truncation error); stability bound in c²λ; parasitic-root growth rate vs observed fail_step.
- **T5.06 start-up sensitivity** as T4.08.
- **T5.07 pure biharmonic control.** Force −L²u alone ("plate"): stability, partition, to separate the effect of the L² term from its mixture.
- **T5.08 is it just rewiring?** Wave on L+εL² vs plain wave on an explicitly 2-hop-augmented graph with matching weights: trajectories/ARI should coincide for mean base — then the idea is a graph rewiring; quantify.
- **T5.09 ε × bridge count.** ARI, edge_margin, steps-to-partition vs bridges∈{1,2,4,8} and ε.
- **T5.10 equal-cost comparison** (2 aggregations/step vs 2× steps).
- **T5.11 stimulus**, **T5.12 partition + hold**, **T5.13 damping rescue** (t4+damping, d4neg+viscous), **T5.14 generalisation**, **T5.15 long horizon**, **T5.16 negative controls**, **T5.17 held-out statistics**.

## 7. Idea 6 — spread transform before the force (`quad`, `sq`, `sqrt`, `invq`, `tanh` × global/local × 5 bases)

Hypothesis: pushing values away from (or toward) the mean before aggregating sharpens block structure or strengthens large stimuli.

- **T6.01 property table.** `force_props`; numerical safety of local scope when all neighbours are equal (s→EPS).
- **T6.02 predicted stiffness.** Global scope, mean base: J=−L·diag(φ′(d)); quad gives φ′ up to 1+3γ(d/s)² → predicted c_crit = c_mean/√max φ′. Verify against bisection across u0 distributions (uniform, gauss, student_t3, rademacher, impulse — impulse has huge d/s).
- **T6.03 γ sweep.** γ∈logspace(0.01,10,9) for quad/invq × c × scope: fail, dead, growth_rate.
- **T6.04 non-monotone invq.** φ′<0 beyond s/√γ → anti-restoring. Does it produce clustered/bistable states? Bimodality coefficient of late states, and alignment of the two modes with true blocks (ARI of sign split). Check on `neg` graphs too (clusters from nothing = artefact).
- **T6.05 block-offset initial conditions.** `block_offset` Δ∈{0,0.25,0.5,1,2}: lifetime of inter-block contrast (D_out/D_in over time, hold) vs mean force. This is where the idea should help most; if not here, nowhere.
- **T6.06 global vs local scope.** dc_drift, behaviour when blocks have different means, fail maps.
- **T6.07 adaptive vs frozen scale.** s recomputed each step (current) vs frozen at t=0: is the per-step renormalisation the source of instability (it keeps the nonlinearity active as amplitude decays)?
- **T6.08 mode transfer.** High-mode energy fraction over time (nonlinear transfer to high modes), temporal Nyquist fraction.
- **T6.09 transform + diffusion.** Same transforms with the diffusion update (stable): ARI, hold, transmit. Does the transform help where stability is not the issue?
- **T6.10 base × transform interaction.** Full 5 bases × 10 transforms × c: two-way decomposition of log growth_rate (main effects vs interaction).
- **T6.11 partition + hold** (all readouts, edge_margin); **T6.12 stimulus** (quad should amplify large stimuli, sqrt/tanh suppress: far_peak vs amp, noise gain, detection AUC, sign symmetry).
- **T6.13 symmetry.** u0 vs −u0, node relabelling.
- **T6.14 rescue.** Minimal damp k; with limiter/saturate.
- **T6.15 generalisation**, **T6.16 long horizon**, **T6.17 negative controls** (false partitions on ER/regular), **T6.18 held-out statistics**.

## 8. Cross-idea phase (after all idea reports)

- **X1 leaderboard.** Each idea's top-3 (dev-selected) + baselines on held-out seeds 100–115 × zoo `core`+`wide`: fail, dead, ARI, edge_margin, hold, transmit, detection AUC, cost.
- **X2 Pareto.** ARI vs transmission vs noise gain across everything.
- **X3 combinations.** Limited factorial: best scheme (ideas 1/2) × best force (3) × best series (4/5) × best transform (6).
- **X4 spectral re-expression.** Spearman between each method's edge scores and Fiedler-vector edge differences; ARI on graphs where spectral fails (cycle blocks) — any method that wins there?
- **X5 threshold sensitivity.** Conclusions under FAIL_AMP∈{10,20,100,1e6} and T∈{250,500,2000}.
- **X6 multiplicity.** Number of configurations tried per idea vs the best observed gain; permutation null for "best of N".
- **X7 q unknown.** `q_select` accuracy on q∈{2,3,4} graphs and on `neg`.
- **X8 original question.** `experiment.py`-style: within-distance Spearman of source→target response vs (P⁶)[t,s], controlling for effective resistance, using the surviving dynamics.
