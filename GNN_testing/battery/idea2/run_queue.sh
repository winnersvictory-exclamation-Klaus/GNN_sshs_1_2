#!/bin/bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea2
while pgrep -f t01_phase_diagram > /dev/null; do sleep 5; done
for t in t02_do_no_harm t09_symmetry t04_normalisation t06_energy_trigger t05_cfl_estimator t03_activation t07_minimal_rescue t11_stimulus t13_impulse t12_combinations t10_partition; do
  s=$(date +%s)
  $PY $t.py > results/${t}.log 2>&1
  echo "$t exit=$? $(( $(date +%s) - s ))s" >> results/queue.log
done
