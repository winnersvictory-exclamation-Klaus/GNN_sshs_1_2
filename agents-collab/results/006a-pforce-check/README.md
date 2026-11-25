# 006a — Smoothed graph p-force check

From the repository root:

```sh
uv run --no-project --python ~/.venvs/math/bin/python --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python agents-collab/results/006a-pforce-check/check.py --out /tmp/006a-rerun
```

`pforce.py` provides `make_pforce(g, p=1.5, epsilon=0.5)` and `potential(g, u, p, epsilon)`. The force adds opposite weighted fluxes at each edge and divides by original weighted degrees. Checks use 5+7 clique and cycle blocks, a single weight-0.25 bridge, and one `default_rng(0)` uniform start per graph. `checks.csv` verifies p=2 versus 003 Mean and degree-weighted net force (`1e-12`), the p=1.5 negative degree-scaled potential gradient (centered finite differences with increment `1e-5`), and the tangent at `1e-5` times the seeded state (both `1e-8`). All 26 checks passed; largest gradient and normalized tangent errors were `4.32e-11` and `7.31e-10` respectively.

`step_screen.csv` records the linearized upper stiffness factor `epsilon^(p−2)=sqrt(2)` times the largest normalized Laplacian eigenvalue. Both tested steps satisfy `h²*lambda_max*sqrt(2)<4` for 003's centered damped wave and `h*lambda_max*sqrt(2)<2` for forward Euler. This is a step-size screen, not a proof of nonlinear trajectory accuracy.

`traces.csv` contains 4,008 rows at common times t=0, 0.5, …, 250. Wave uses 003's zero-velocity startup and damping `gamma=0.1`; kinetic energy is `0.5*sum(d_i*v_i²)` using the integrator's centered velocities. Diffusion energy is just the edge potential. Amplitude columns give max absolute state and degree-weighted centered RMS in state units. `h_halving.csv` compares same-time traces and counts saved energy increases larger than `1e-12`. No runs failed. Coarse wave had 47 and 50 such increases on clique and cycle respectively; fine wave and both diffusion steps had none on these starts. Fine and coarse energy trends both decayed, but exact discrete monotonicity is not assumed. No cuts or parameter sweeps were run.
