#!/bin/bash
# Multi-seed runs for the horizon experiments. Pick runs with RUNS="mode:seed ...".
#   horizon:    RUNS="fwdmax:0 fwdmax:1 fwdmax:2 cbase:1 proposed:1 cbase:2 proposed:2"
#   horizon30:  RUNS="fwdmax6:0 fwdmax6:1 fwdmax6:2"
# Seed 0 of cbase and proposed is reused from the first replication (scripts/run_all.sh).
set -uo pipefail
source "$(dirname "$0")/common.sh"
MODELS="$DATA/models_seeds"
mkdir -p "$MODELS"
cd "$ROOT/scripts"

for m in cbase proposed; do
  if [ ! -e "$MODELS/${m}_s0" ] && [ -d "$DATA/models/$m" ]; then ln -s "$DATA/models/$m" "$MODELS/${m}_s0"; fi
done

for run in ${RUNS:-fwdmax:0 fwdmax:1 fwdmax:2 cbase:1 proposed:1 cbase:2 proposed:2}; do
  m=${run%%:*}; s=${run##*:}
  out="$MODELS/${m}_s${s}"
  if [ -f "$out/test_probs.npz" ]; then echo "=== skip $m seed $s"; continue; fi
  echo "=== train $m seed $s"
  "$PY" train.py "$DATA/data.npz" "$UPSTREAM" "$m" "$out" "${MINUTES:-60}" "$s" || echo "=== FAILED $m seed $s"
done
echo "=== all done"
