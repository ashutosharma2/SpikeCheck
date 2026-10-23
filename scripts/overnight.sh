#!/bin/bash
# Run the multi-seed training and evaluation unattended, optionally from a set time.
#   START_AT=22:00  wait until that time today before starting (optional)
#   RUNS=...        passed through to run_seeds.sh
#   RES=horizon     results folder under results/
# Example: START_AT=22:00 RUNS="fwdmax6:0 fwdmax6:1 fwdmax6:2" RES=horizon30 \
#          nohup caffeinate -i bash scripts/overnight.sh > overnight.log 2>&1 &
# caffeinate stops idle sleep from pausing the run; a closed lid still sleeps.
set -uo pipefail
source "$(dirname "$0")/common.sh"
RES="${RES:-horizon}"

if [ -n "${START_AT:-}" ]; then
  start=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $START_AT" +%s)
  now=$(date +%s)
  if [ "$now" -lt "$start" ]; then
    echo "waiting until $START_AT ($(( (start - now) / 60 )) min)"
    sleep $(( start - now ))
  fi
fi

echo "=== training start $(date)"
bash "$ROOT/scripts/run_seeds.sh" > "$DATA/run_seeds_$RES.log" 2>&1
grep "===" "$DATA/run_seeds_$RES.log"

echo "=== eval start $(date)"
mkdir -p "$ROOT/results/$RES"
"$PY" "$ROOT/scripts/eval_seeds.py" "$DATA/data.npz" "$DATA/models_seeds" "$ROOT/results/$RES" \
  > "$DATA/eval_seeds_$RES.log" 2>&1 || { echo "=== EVAL FAILED"; tail -20 "$DATA/eval_seeds_$RES.log"; }
for d in "$DATA"/models_seeds/*/; do
  n=$(basename "$d"); [ -f "$d/train_log.json" ] && cp "$d/train_log.json" "$ROOT/results/$RES/train_log_$n.json"
done
echo "=== finished $(date)"
