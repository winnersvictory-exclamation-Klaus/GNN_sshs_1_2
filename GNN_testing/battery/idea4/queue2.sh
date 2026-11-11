#!/bin/bash
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea4
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
while [ ! -f results/go2 ]; do sleep 3; done
for t in t411_partition_hold t414_long_horizon t415_negative_controls t413_generalisation; do
  $PY $t.py > results/${t:0:4}.log 2>&1
done
echo queue2 done > results/queue2.done
