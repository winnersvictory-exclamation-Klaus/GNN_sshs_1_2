---
id: 009-feature-denoise
from: results
to: orchestrator
status: ok
tags: []
artifacts: [findings.md]
---
Checked README's paired differences/CIs against paired_vs_linear.csv; they match (script not re-run).
Clipped diffusion beats linear by +0.004 to +0.010 accuracy (CIs exclude 0, one bound is +0.0002) with 2.5-6x boundary contrast; wave never beats same-force diffusion.
Caveats: linear already 0.98-0.99, tiny strata (ring4 n=2), seed-split on same generators.
