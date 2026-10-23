# Shared settings for the runner scripts. Source it; don't run it.
#   SPIKECHECK_DATA      where data, models and intermediate files live (default: ./data)
#   SPIKECHECK_UPSTREAM  the authors' ResourceForecasting checkout (default: ./upstream/ResourceForecasting)
#   PYTHON               the Python interpreter to use (default: python3)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${SPIKECHECK_DATA:-$ROOT/data}"
UPSTREAM="${SPIKECHECK_UPSTREAM:-$ROOT/upstream/ResourceForecasting}"
PY="${PYTHON:-python3}"
