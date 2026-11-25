# 004b — Directional stiffness freeze

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/004b-frozen-directional/directional.py --out /tmp/004b-rerun
```

Same 16 development SBM graphs, eight archived amplitude-1 starts, both step sizes, and same 004a wave continuation and 003 late [125,250] raw/within-window low-pass cut. The adaptive warm-ups exactly matched the t=50 archive; both current and previous substep states were carried forward. The directional fork fixes separate multipliers `a_i` and `a_j` at t=50 and uses the original weighted degrees. Multipliers are checked finite and in [0,1]; the force closes over read-only directed conductances. Its t=50 force matches adaptive on every start (max absolute error `1.25e-16`, tolerance `1e-12`). No comparison branches were rerun.

`initializations.csv` logs threshold/censor substeps and fork force errors; `graphs.csv` gives ARI, AUC, gap and cut status. At every substep the 004a/003 rule flags nonfinite values or `|u|≥20`, with emergency censoring for nonfinite or `|u|>1e8`. Any failure zeros whole-graph plan ARI; survivor ARI and unscorable cuts are reported separately. `paired_graphs.csv` gives directional minus each archived comparator; `paired_summary.csv` averages paired differences with metric-specific denominators. `edges.npz`: each `(graph,h)` row uses slice `edge_ptr[k]:edge_ptr[k+1]`; `multiplier` axes are `[endpoint (i,j), init, concatenated edge]`; `scores` axes are `[raw/lp, concatenated edge]`. Multiply by `w` for directed conductances.

No starts failed (0/128 per h); all cuts were scored. Low-pass ARI versus symmetric freeze was +0.069 at h=0.5 and +0.053 at h=0.25. Versus continued adaptive it was +0.063 and −0.051, respectively. The comparison is conditional on adaptive warm-up; the coarse-step advantage over adaptive did not survive step halving.
