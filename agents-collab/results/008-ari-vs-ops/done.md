---
id: 008-ari-vs-ops
from: test-maker
to: orchestrator
status: ok
tags: []
artifacts: [README.md, ari_vs_ops.py, graphs.csv, curves.csv.gz, targets.csv, reach_summary.csv, curve_summary.csv, paired_ops_ratio.csv]
---
All 8 configs x 48 graphs x 2 readouts ran (768 rows), 0 failed starts, no unstable step by dense audit; README has repro command and ops accounting.
Wave (g*) vs fast diffusion, lp: median ops ratio 1.14 clique, 1.14 sbm, 0.96 cycle (8/16 reach); local Huber h=0.25 vs fast diffusion, lp: 0.76 clique, 2.29 sbm, 3.26 cycle. Raw readout: wave g* reaches only 2/16 cycle graphs.
Sensitivity g=0.1 and 0.5g* reach target on 27/48 (lp), 2g* on 40/48. 8 cycle graphs reach 0.8 for no diffusion/g* wave config; spectral reaches it on 36/48.
