#!/bin/zsh
# Launch N search workers for H hours, keeping the Mac awake. Idempotent per worker id.
N=${1:-10}; H=${2:-10}
cd /Users/matveypro/kronos_lab/overnight
mkdir -p out logs
nohup caffeinate -i -s -t $(( H * 3600 + 900 )) > /dev/null 2>&1 &
for i in $(seq 0 $((N - 1))); do
  if ! pgrep -f "worker.py $i " > /dev/null; then
    nohup /Users/matveypro/.venvs/kronos/bin/python -W ignore worker.py $i $H > logs/w$i.log 2>&1 &
  fi
done
date '+%Y-%m-%d %H:%M:%S' > out/launched_at.txt
echo "launched $N workers for $H h"
