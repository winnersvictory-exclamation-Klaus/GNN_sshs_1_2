#!/bin/bash
# sequential chain (max 2 workers at any time): small sanity run, then full run if the small run succeeded
cd /Users/aidenkoh/Documents/GitHub/GNN_testing/battery/idea5
PY=/opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python
while [ ! -f results/t02_dispersion.csv ]; do sleep 5; done
sleep 5
for t in "$@"; do
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
echo CHAIN_DONE >> results/chain.log
