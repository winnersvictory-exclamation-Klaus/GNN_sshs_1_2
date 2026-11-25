# 006b — p-force wave and diffusion

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/006b-pforce-dynamics/dynamics.py --out /tmp/006b-rerun
```

Uses all 48 development graphs and eight saved starts per graph from 003. The new forces are 006a's smoothed p-force (`p=1.5`, `epsilon=0.5`) and `sqrt(2)*Mean`, its tangent-matched linear control. Both dynamics reuse 003's integrator through t=250 at h=0.5 and 0.25. Saved frames are 0.5 apart; edge RMS uses [0,50] and [125,250] separately and takes the median over survivors. LP is 003's within-window FFT cutoff (0.02 cycles/frame). The same score vector goes into both the stable q=2 cut and the descriptive AUC/gap diagnostics. The code checks unique archive score keys, start-array dimensions, graph truth and full output coverage. Ordinary Mean and spectral scores are joined, not rerun.

`initializations.csv` records first nonfinite or `abs(u)>=20` hits at any substep, with the inherited safety censor at nonfinite or `max(abs(u))>1e8`. Any failure zeros whole-graph plan ARI; unavailable cuts are flagged separately. `graphs.csv` has per-run-window-readout metrics, including class RMS across the edge score vectors and a flag when both are below `1e-8`. For p-force, bridge/internal secant stiffness is averaged over unfiltered frames, edges and surviving starts, and also given as a fraction of `sqrt(2)`; it is identical across readouts for the same window. `paired_graphs.csv` and `paired_summary.csv` give p-force minus comparator plan ARI, with overall and family denominators in the latter. For `edges.npz`, graph edge metadata uses `edge_ptr`; run metadata uses the `run_*` columns, and `scores` has axes `[window, readout, concatenated run edge]`, sliced via `run_ptr` for each run. Edge order matches the run's graph.

**Development result:** 0/3,072 starts failed; all cuts were scorable and no score vector triggered the both-class RMS floor. Late LP p-wave minus matched-linear wave was +0.084 at h=0.5 and −0.026 at h=0.25. It trailed same-force diffusion at both steps (−0.235 and −0.329), and p-diffusion and matched-linear diffusion had the same late LP mean ARI at both steps. In the h=0.25 late wave the mean secant stiffnesses for both classes were within `1e-7` of the tangent limit in relative units. These are fixed development comparisons, not a parameter search or confirmation test.
