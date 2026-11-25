# 005 — Development mechanism figures

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with matplotlib==3.11.2 python agents-collab/results/005-mechanism-figures/figures.py --out /tmp/005-rerun
```

- [`checkpoint_contrasts.svg`](checkpoint_contrasts.svg) shows the four saved checkpoints for one illustrative 24+24 SBM. Each point is the median over eight starts of the RMS across edges in that class. Dotted lines connect observed points, not additional measurements. [`checkpoint_contrasts.csv`](checkpoint_contrasts.csv) gives the 32 plotted values and class edge counts.
- [`failure_and_ari.svg`](failure_and_ari.svg) shows graph failure counts and mean late-window plan ARI for all 12 configurations across 48 graphs. Spectral is an ARI reference only. [`failure_and_ari.csv`](failure_and_ari.csv) gives the 24 configuration/readout rows and one spectral row.

Only the saved 003 checkpoints, edge metadata, scores and failure counts are read. No dynamics are rerun.
