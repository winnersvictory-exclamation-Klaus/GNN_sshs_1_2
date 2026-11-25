# 006d — Approximate static p-cut outscored p-wave on the development grid

**Accepted as a paired development comparison.** The static p=1.5 sweep scored all 48 graphs, with mean ARI **0.851** and 39 exact recoveries. It outscored p-wave in every prespecified window/readout/step cell. Its late advantage over p-diffusion was much smaller, and neither the static nor the dynamic results establish a confirmation or compute-cost claim.

| Comparator | Static minus comparator ARI | Static wins / ties / losses (48 graphs) |
|---|---:|---:|
| Spectral | +0.038 | 11 / 33 / 4 |
| Late low-pass p-diffusion, either h | +0.015 | 6 / 40 / 2 |
| Late low-pass p-wave, h=0.5 | +0.251 | 20 / 25 / 3 |
| Late low-pass p-wave, h=0.25 | +0.344 | 22 / 23 / 3 |

The late diffusion difference is entirely from cycles (+0.045 on 16 graphs); clique and SBM differences are zero. The static–spectral difference is also mostly from cycles (+0.102), with +0.011 on SBM and no clique difference. For p-wave, even its closest overall comparison (early low-pass) leaves a gap of about **+0.127** at each step. The static objective is an **unsmoothed** Rayleigh quotient followed by thresholding; the dynamic methods evolve a smoothed force and cut by edge RMS. These results compare complete methods, not inertia in isolation, and equal graph sets do not imply equal computation budgets.

## Verification and optimizer status

The pinned command reproduced all five generated CSVs byte-for-byte. Using the archived graph edges, I independently evaluated every distinct threshold of each saved vector; all selected thresholds attain the best binary objective for their own vector to numerical tolerance. The largest independent discrepancy in the selected objective was `1.39e−17`. I recomputed all 48 ARIs from the saved labels and all 816 graph-paired differences against the archived dynamic or spectral references.

Of 192 optimizer trials, 109 reported success, 82 precision loss, and one the iteration limit. The selected trial reported success on **20/48** graphs and precision loss on **28/48**; the maximum selected gradient infinity norm was `2.36e−5`. All selected cuts are finite and valid. In particular, 27 of the 28 precision-loss selections recovered the truth exactly, illustrating that the optimizer status and partition accuracy answer different questions. Threshold validation does not certify a global binary optimum or convergence of the continuous relaxation on these 48 graphs.

**No tested p-wave accuracy claim advances.** The static result supplies a useful approximate baseline on this development grid, not confirmed superiority over spectral or diffusion.
