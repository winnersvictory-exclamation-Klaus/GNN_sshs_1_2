# 004c — Changing initialization amplitude did not change the observed cuts

**Accepted as a development sensitivity check.** Halving or doubling the initial amplitudes left all 768 late raw and low-pass cuts unchanged up to label swapping, across the same 16 SBM graphs, six methods and both step sizes. All paired ARI changes versus amplitude 1 were exactly zero. Neither the absolute nor the amplitude-scaled threshold was hit in 3,072 method/start records, and no cut was unscorable. The earlier step-size sensitivity therefore persists at both tested amplitudes; it is not removed by rescaling the starts.

## Scaling and the epsilon check

The tested local rule is nearly positively homogeneous by construction. At a fixed normalized state `v`, scaling the input to `A*v` and dividing the force by `A` changes the normalized clipping threshold only by `EPS*(1/A−1)`. The recorded maximum same-state force differences are about `5.00e−13` at A=0.5 and `2.50e−13` at A=2, within rounding slack of the analytic bound. That bound is for a force evaluation, not a bound on the later trajectory.

The maximum absolute discrepancy between `score/A` and the archived amplitude-1 score was `2.61e−10` for continued adaptive wave, `5.56e−11` for same-force diffusion, and at most `1.53e−12` for any warm-up fork. Pure Mean edge scores were bit-identical after normalization. These magnitudes are consistent with near-homogeneity; they do not prove exact scale invariance or separate long-horizon epsilon effects from numerical error. Fixed-threshold Huber was not among the tested methods.

## What carries over from 004a and 004b

At both amplitudes, directional minus adaptive late low-pass ARI is still **+0.063 at h=0.5 and −0.051 at h=0.25**; directional minus unweighted is **+0.010 and −0.069**. Symmetric freezing still trails unweighted continuation at both steps (−0.059 and −0.122). Amplitude robustness of these particular observations does not make the freeze advantage robust to step halving, establish a standalone fixed-weight method, or warrant confirmation.

There were **22 flagged near-tie rows at the new amplitudes**, all in late same-force diffusion. No Kruskal gap changed sign and no partition changed. Nevertheless, persistence of the observed labels should not be taken as robust score separation: in some rows the normalized score perturbation is larger than the tiny archived gap.

## Verification

The pinned command regenerated every artifact byte-for-byte. I checked the 3,072 unique start records and threshold flags, recomputed the normalized edge-score differences from the NPZ, and independently reran stable Kruskal on the stored score vectors to verify label equivalence for every comparison. This check is limited to the two rescalings of the same development starts, not new graphs or independent initializations.
