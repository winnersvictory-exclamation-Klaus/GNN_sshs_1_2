# 004a — Symmetric freeze vs unweighted continuation

From the repository root (using 003's Python environment):

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/004a-frozen-sbm/frozen_sbm.py --out /tmp/004a-rerun
```

Uses 003's 16 development SBM graphs, eight archived starts at amplitude 1, both step sizes, and the same centered-damping wave update (`gamma=0.1`). All 32 adaptive warm-ups matched the archived t=50 state bit for bit (required max absolute error ≤ `1e-12`). Each fork inherits both the t=50 state and the preceding substep state. The freeze uses per-start `w*(a_i+a_j)/2` at t=50, with ratio 1 at zero difference and the **original** weighted degree denominator. Multipliers are verified finite and in [0,1]; the symmetric adjacency is read-only. An independent oriented-flux sum checks the initial force and its zero weighted total at tolerance `1e-12`. The other fork uses Mean's original graph weights (unit weights here).

Both forks check all substeps for nonfinite values or `abs(u)>=20`. As in 003, nonfinite or `max(abs(u))>1e8` stops that start; threshold failures are excluded from survivor readouts and set whole-graph plan ARI to zero. Unscorable cuts are flagged separately. The common saved cadence is 0.5, with 251 frames on [125,250]; scores use 003's raw or within-window FFT low-pass (`fc=0.02`), median of per-start RMS, and stable q=2 Kruskal cut. Baselines are joined from 003 by graph, step size and readout; spectral has neither a dynamical run denominator nor edge scores. No baselines were rerun.

**Outputs:** `warmup.csv` (32 checks), `initializations.csv` (512 starts), `graphs.csv` (128 fork/readout rows), `paired_graphs.csv` and `paired_summary.csv` (fork minus comparator, with metric-specific paired denominators). `edges.npz`: each `(graph,h)` row owns the edge slice `edge_ptr[k]:edge_ptr[k+1]` of `i,j,w,bridge`; `multiplier` has axes `[init, concatenated edge]`, and `scores` has `[fork,readout,concatenated edge]` with ordering in `fork` and `readout`. The actual conductance is `w * multiplier`.

**Descriptive result:** zero failures and unscorable cuts for both forks (0/128 starts, 0/16 graphs per fork and step size). Raw ARI is identical between forks at both step sizes. On low-pass ARI, freeze minus unweighted is −0.059 at h=0.5 and −0.122 at h=0.25; freeze minus continued adaptive is −0.006 and −0.104, respectively. The near-match to adaptive at the coarser step did not persist under step halving, and the symmetric conductances did not improve on unweighted continuation by mean ARI. This comparison is conditional on adaptive warm-up and does not separate freezing from symmetrizing.
