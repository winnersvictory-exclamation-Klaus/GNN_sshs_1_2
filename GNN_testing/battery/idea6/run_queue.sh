#!/bin/bash
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
for t in t602b_trajectory t605_blockoffset t609_diffusion t604_invq t606_scope t607_frozen t610_interaction t608_modes t611_partition t612_stimulus t613_symmetry t614_rescue t616_long t617_neg; do
  s=$(date +%s)
  $PY idea6/$t.py > idea6/results/$t.log 2>&1
  echo "$t exit=$? secs=$(( $(date +%s)-s ))" >> idea6/results/queue.log
done
echo ALLDONE >> idea6/results/queue.log
