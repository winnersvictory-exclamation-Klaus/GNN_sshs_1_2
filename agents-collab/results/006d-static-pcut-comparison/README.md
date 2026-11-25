# 006d — Static p-cut comparison on the 003 graphs

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/006d-static-pcut-comparison/comparison.py --out /tmp/006d-rerun
```

Runs the unchanged 006c p=1.5 numerical sweep (one Fiedler and three seed-0 random starts) on the 48 development graphs. The static solver sees no truth labels; ARI uses truth only after the threshold cut returns. The static threshold minimizes the binary p-Rayleigh objective among that vector's thresholds, not the dynamic edge-RMS objective. Every selected binary objective was independently recomputed from dense crossing weights and scalar minimization of the binary denominator (tolerance `1e-10`, largest error `6.94e-18`). This checks the selected cut's objective, not global optimality.

- `graphs.csv`: selected continuous and binary objective, independent check, optimizer status/residual, ARI and unscorable flag. A valid cut is kept even when the optimizer reports precision loss. Every static result is marked approximate.
- `optimizer_trials.csv`: all 192 trial statuses and residuals; `vectors.csv`: node-indexed selected vectors and labels.
- `paired_graphs.csv` / `paired_summary.csv`: static ARI minus archived 006b plan ARI, at each h, window and readout separately, plus one comparison with archived 003 spectral. Overall and family summaries report the finite pair denominator. No dynamics or spectral labels were rerun.

All 48 cuts were scorable; the selected optimizer reported precision loss on 28 graphs and success on 20. Static minus spectral mean ARI was +0.038. In the late LP comparison, static minus p-wave was +0.251 at h=0.5 and +0.344 at h=0.25; static minus p-diffusion was +0.015 at both. These are paired development results only.
