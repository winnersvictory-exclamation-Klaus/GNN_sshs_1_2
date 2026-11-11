#!/bin/bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea2
while pgrep -f run_queue.sh > /dev/null; do sleep 5; done
$PY t09_symmetry.py > results/t09_symmetry.log 2>&1; echo "t09 rerun exit=$?" >> results/queue.log
$PY t04_normalisation.py > results/t04_normalisation.log 2>&1; echo "t04 rerun exit=$?" >> results/queue.log
$PY t03_activation.py > results/t03_activation.log 2>&1; echo "t03 rerun exit=$?" >> results/queue.log
$PY t08_bounded_garbage.py > results/t08_bounded_garbage.log 2>&1; echo "t08 exit=$?" >> results/queue.log
$PY t14_long_horizon.py > results/t14_long_horizon.log 2>&1; echo "t14 exit=$?" >> results/queue.log
