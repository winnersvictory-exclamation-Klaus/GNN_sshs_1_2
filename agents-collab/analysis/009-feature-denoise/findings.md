# 009 — Clipped diffusion denoised block features slightly better than linear; wave added nothing

**Development comparison (36 held-out graphs × 8 draws, frozen t\* from seeds 0–1).** I checked the paired differences and CIs in the README against `paired_vs_linear.csv`; they match. I did not re-run `feature_denoise.py`.

- **Clipping gives a small, real gain.** Frozen-t\* accuracy over linear diffusion is +0.004 to +0.010 for fixed Huber and local Huber diffusion. The CIs exclude 0 at every sigma, but the lowest bound is +0.0002 (fixed Huber, sigma 3). Linear already scores 0.98–0.99, so the room is under 0.02.
- **Boundary contrast is the larger effect.** Local Huber diffusion's median contrast is about 5–6× linear's, and fixed Huber's is 2.5–3×. Accuracy and MSE move far less, so contrast measures edge preservation rather than better denoising.
- **Wave never beats the same-force diffusion.** Mean wave is level with linear at sigma 1 and 0.03 worse at sigma 2–3. Local Huber wave is +0.008 over linear at sigma 1 but −0.026 at sigma 3. The frozen-t gap exceeds the oracle-t gap (−0.032 vs −0.005 at sigma 3), so the wave's best time is unstable across seeds, not necessarily worse at its best.

## Limits on what this supports

- With 36 graphs, 5 families and 2 ring4 test graphs, per-family effects have crude CIs. At sigma 3, local Huber is +0.010 on `sbm` (n=12) and −0.027 on `ring4` (n=2), so the pooled gain is not uniform.
- The split is by seed on the same generators, so the held-out half is not an independent graph distribution. Generated graphs are deterministic for grid_neck, barbell and ring4, which makes seeds 0–1 and 2–3 differ only in noise draws.
- Fixed Huber's tau=0.5 is not scaled to the noise. Wave horizon is 512, g uses an estimated lam2, and `sbm_het` (sigma 1.0) and `ring4` (q=4) deviate from the requested defaults, as the README states.
- Clipped diffusion's gain is likely too small to matter in practice given the 0.98–0.99 baseline. No claim of wave benefit for feature denoising advances.
