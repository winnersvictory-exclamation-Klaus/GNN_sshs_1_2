---
id: 009-feature-denoise
from: test-maker
to: orchestrator
status: ok
tags: []
artifacts: [README.md, feature_denoise.py, graphs.csv, curves_per_graph.csv.gz, curve_summary.csv, selection.csv, test_per_graph.csv, test_summary.csv, paired_vs_linear.csv]
---
72 dev graphs x 6 methods x 3 sigma x 8 draws, 0 failures; 36 held-out (seeds 2-3) graphs, frozen t* from seeds 0-1; bootstrap CIs over graphs.
Clipped diffusion beats linear by a small margin (frozen t*: +0.004 to +0.010 accuracy, CI mostly excludes 0; linear already 0.98-0.99), and boundary contrast is 2.5-6x higher. Wave variants never beat same-force diffusion (mean wave -0.03 at sigma 3, frozen t).
Deviations: sbm_het uses sigma=1.0 (default 0.6 equals sbm exactly); ring_of_cliques q=2 impossible, used q=4 with two sign halves.
