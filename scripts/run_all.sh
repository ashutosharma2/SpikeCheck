#!/bin/bash
# First replication: train the three targets (seed 0) and evaluate them.
# Writes into results/replication/. Run scripts/get_data.sh first.
set -euo pipefail
source "$(dirname "$0")/common.sh"
OUT="$ROOT/results/replication"
cd "$ROOT/scripts"

for m in cbase proposed proposed_full; do
  echo "=== train $m"
  "$PY" train.py "$DATA/data.npz" "$UPSTREAM" "$m" "$DATA/models/$m" "${MINUTES:-60}"
  cp "$DATA/models/$m/train_log.json" "$OUT/train_log_$m.json"
done
echo "=== eval"
"$PY" run_eval.py "$DATA/data.npz" "$UPSTREAM" "$DATA/models" "$OUT"
echo "=== done"
