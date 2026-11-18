#!/bin/bash
# Train the Strategy 2 models of ADR-013: four variants, three seeds, one at a time.
#   START_AT="2026-11-17 00:00"  wait until that date and time before starting (optional)
#   MINUTES=60                   training cap per model, as scripts/run_seeds.sh
#   RUNS="single:0 binary:0"     pick runs (default: every variant for seed 0, then 1, then 2)
# Finished runs are skipped, so the script can be started again after an interruption.
# Example: START_AT="2026-11-17 00:00" nohup caffeinate -i bash strategy2/train_all.sh \
#          > data/train_s2.log 2>&1 &
# caffeinate stops idle sleep from pausing the run; a closed lid still sleeps.
set -uo pipefail
source "$(dirname "$0")/../scripts/common.sh"
MODELS="$DATA/models_s2"
mkdir -p "$MODELS"
# A relative PYTHON path would break after the cd below, so make it absolute first.
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; esac
cd "$ROOT/strategy2"

if [ -n "${START_AT:-}" ]; then
  start=$(date -j -f "%Y-%m-%d %H:%M" "$START_AT" +%s)
  now=$(date +%s)
  if [ "$now" -lt "$start" ]; then
    echo "waiting until $START_AT ($(( (start - now) / 60 )) min)"
    sleep $(( start - now ))
  fi
fi

DEFAULT=""
for s in 0 1 2; do for v in single binary proportional exponential; do DEFAULT="$DEFAULT $v:$s"; done; done

echo "=== training start $(date)"
for run in ${RUNS:-$DEFAULT}; do
  v=${run%%:*}; s=${run##*:}
  out="$MODELS/${v}_s${s}"
  if [ -f "$out/test_probs.npz" ]; then echo "=== skip $v seed $s"; continue; fi
  echo "=== train $v seed $s $(date +%H:%M)"
  "$PY" train_s2.py "$DATA/data.npz" "$UPSTREAM" "$v" "$out" "${MINUTES:-60}" "$s" || echo "=== FAILED $v seed $s"
done
echo "=== all done $(date)"
