#!/bin/bash
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea5
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
for t in t12b; do
  f=$(ls ${t}_*.py | head -1)
  echo "=== $f small $(date +%T)" >> results/chain.log
  if $PY $f --small > results/${t}_small.log 2>&1; then
     echo "=== $f full $(date +%T)" >> results/chain.log
     ( time $PY $f ) > results/${t}.log 2>&1
     echo "=== $f done rc=$? $(date +%T)" >> results/chain.log
  else
     echo "=== $f SMALL FAILED $(date +%T)" >> results/chain.log
  fi
done
echo CHAIN2_DONE >> results/chain.log
