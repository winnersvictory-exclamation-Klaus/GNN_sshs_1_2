#!/bin/bash
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
until grep -q ALLDONE idea6/results/queue.log; do sleep 10; done
for t in "$@"; do
  s=$(date +%s)
  $PY idea6/$t.py > idea6/results/$t.log 2>&1
  echo "$t exit=$? secs=$(( $(date +%s)-s ))" >> idea6/results/queue2.log
done
echo ALLDONE >> idea6/results/queue2.log
