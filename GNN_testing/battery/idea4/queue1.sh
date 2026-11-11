#!/bin/bash
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
for t in t404_gap_reshaping t405_equal_cost t406_nonlinear_bases t412_damping_rescue; do
  $PY $t.py > results/${t:0:4}.log 2>&1
done
echo queue1 done > results/queue1.done
