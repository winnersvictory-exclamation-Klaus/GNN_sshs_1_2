---
id: 008-ari-vs-ops
from: results
to: orchestrator
status: ok
tags: []
artifacts: [findings.md]
---
Checked 008 headline numbers against the CSVs; all match (did not re-run the script).
Result: wave at g* is no cheaper than fast diffusion (median ops ratio 1.14 clique/sbm, 0.96 cycle); local Huber is 2.3-3.3x costlier on sbm/cycle.
Caveats: 10-step check grid makes the 1.14 a one-step quantum; 8/16 cycle graphs are unreachable for every method; the 2g* cycle ratio (0.64) is an untuned lead only.
