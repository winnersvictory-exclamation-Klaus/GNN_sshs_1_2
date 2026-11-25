# 008 — Wave with label-free damping showed no ops advantage over fast diffusion

**Development comparison only (48 graphs, 8 saved starts each; no new seeds).** I checked every number quoted in the 008 `done.md` against `paired_ops_ratio.csv`, `reach_summary.csv` and `targets.csv`, and all match. I did not re-run `ari_vs_ops.py`.

| Low-pass readout, ops to target ARI ≥ 0.8 vs `diff_fast` | clique | sbm | cycle (8/16 reach) |
|---|---:|---:|---:|
| `wave_gstar` median ratio | 1.14 | 1.14 | 0.96 |
| `huber_local_h0.25` median ratio | 0.76 | 2.29 | 3.26 |

- **No wave advantage.** Wave at g* costs about 14% more than fast diffusion on clique and SBM, and about equal on cycles. The cycle geomean is 0.83, but this covers only the 8 graphs both methods reach and it rests on the 0.96 median, so I read it as parity rather than a win.
- **Doubling g\* helps on cycles only.** `wave_2gstar` has a cycle median ratio of 0.64 (8 graphs). Its all-family geomean is 0.998 and its median is 1.14. This is a sensitivity run, not a tuned method, so it is a lead to check, not a claim.
- **Damping needs to be at least g\*.** At g = 0.1 and 0.5g*, the wave reaches the target on 27/48 graphs (low-pass), against 40/48 for g* and 2g*. On those graphs diffusion reaches it in 13 more cases. Without the low-pass readout the wave is much worse (g* reaches 2/16 cycles raw), so the wave depends on the readout.
- **Cycles are limited by the graph, not by the dynamics.** For every config except two sensitivity runs, 8 of 16 cycle graphs never reach 0.8, and spectral reaches it on only 4/16. Cycle ratios therefore describe the easier half.
- **Local Huber is not cheaper where the problem is hard.** It is cheaper on cliques at h=0.25 (0.76) but 2.3× on SBM and 3.3× on cycles. Its per-node median sort is not counted, so its cost is understated.

## Caveats on the ratios

- Checks happen every 10 steps and ops are linear in steps, so targets sit on a coarse grid. The clique ratio of 1.14 (8/7) is one grid step, not a measured effect size, and clique medians of 0.76 and 1.14 are probably similar quantization.
- Ops count one force evaluation and one scoring pass per step. Wall-clock time, memory traffic and the wave's extra state are not measured, so this is not a runtime claim.
- The wave's setup includes 30 Lanczos matvecs to estimate g*, which are counted. g* is overestimated on cycles (lam2_est/lam2_true up to 2.57), so the cycle result partly tests a larger damping than designed.
- `diff_fast` uses h = 1.8/lmax, near its stability limit, so it is a strong baseline. The wave's h was set by the same rule.

**No tested wave or Huber compute-efficiency claim advances.** Next, confirmation on fresh graphs would need a finer check interval and a cycle family whose targets are reachable.
