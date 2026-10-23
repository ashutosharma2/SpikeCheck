#!/bin/bash
# Download and prepare the Alibaba 2018 trace.
# Streams the 29 GB container_usage archive without storing it (about 2.5 h at 4 MB/s),
# then builds the experiment dataset. Needs about 1 GB of free disk.
set -euo pipefail
source "$(dirname "$0")/common.sh"
URL=http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces

mkdir -p "$DATA"
if [ ! -f "$DATA/container_meta.csv" ]; then
  curl -sSf -o "$DATA/container_meta.tar.gz" "$URL/container_meta.tar.gz"
  tar -xzf "$DATA/container_meta.tar.gz" -C "$DATA"
fi
if [ ! -f "$DATA/alibaba_5min.npz" ]; then
  "$PY" -u "$ROOT/scripts/stream_alibaba.py" "$DATA/container_meta.csv" "$DATA"
fi
"$PY" "$ROOT/scripts/prep.py" "$DATA/alibaba_5min.npz" "$DATA/chosen_containers.csv" "$DATA/data.npz"
# Uncompressed copy of the inputs: train.py memory-maps it to keep RAM down.
"$PY" -c "import numpy as np, sys; np.save(sys.argv[2], np.load(sys.argv[1])['X'])" "$DATA/data.npz" "$DATA/X.npy"
