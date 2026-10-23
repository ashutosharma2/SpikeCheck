#!/bin/bash
# Clone the authors' code at the exact commit the experiments used, then install packages.
set -euo pipefail
source "$(dirname "$0")/common.sh"
PIN=98f0d5103de9df075c0885c36325d2254e05142f

if [ ! -d "$UPSTREAM/.git" ]; then
  git clone https://github.com/FerranAgulloLopez/ResourceForecasting.git "$UPSTREAM"
fi
git -C "$UPSTREAM" checkout --quiet "$PIN"
echo "authors' code at $(git -C "$UPSTREAM" rev-parse --short HEAD) in $UPSTREAM"

"$PY" -m pip install -r "$ROOT/requirements.txt"
