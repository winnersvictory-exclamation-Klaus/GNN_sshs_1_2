# 006c — Static p-Rayleigh sweep validation

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/006c-static-pcut-check/check.py --out /tmp/006c-rerun
```

`pcut.p_rayleigh_sweep(g, p=1.5, seed=0)` takes a connected weighted graph, not partition labels. Each evaluation solves for the degree-weighted p-center and uses the envelope gradient of the quotient. BFGS starts from the degree-rescaled normalized Fiedler vector and three seeded Gaussian vectors. Starts and returned vectors are p-centered and normalized to unit weighted p-deviation. The lowest continuous objective selects the vector, then the lowest binary objective over distinct thresholds selects its cut. Ties use start order and then the smallest threshold, respectively. This is a numerical sweep, not a certificate of a continuous or binary global optimum. Check `optimizer_trials.csv` for each status and residual; the selected p=1.5 trials reported precision loss (status 2), with gradient infinity norms `2.69e-6` and `5.03e-7` on the normalized vectors.

`validation.csv` contains checks for 4+5 clique and cycle pairs connected by one weight-0.25 bridge, graph seed 0. The centered-difference gradient (`step=1e-5`, tolerance `1e-7`) matched to `3.79e-10` or better. The p=2 Fiedler and optimized Rayleigh values matched the normalized Laplacian eigenvalue within `1e-12`. `binary_cuts.csv` enumerates 255 cuts per graph, modulo complements, with an independent dense crossing-weight calculation and one-dimensional minimization for the denominator. Both binary objective and threshold-minimum errors were below `1e-10`. The returned thresholds had zero observed regret against the enumerated binary optima on these two toys; this does not imply the same on other graphs. `vectors.csv` records the selected vectors and cuts. No development-grid runs were done.
