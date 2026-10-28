#!/usr/bin/env bash
# Resume saved inputs from the main lab checkout; keep every artifact in its run.
set -euo pipefail
cd "$(dirname "$0")/../../.."
output=${1:?Pass the absolute experiment directory}
workers=${2:-3}
mkdir -p "$output/logs" "$output/raw_data/tmp" "$output/raw_data/tool_cache/matplotlib"
export TMPDIR="$output/raw_data/tmp"
export XDG_CACHE_HOME="$output/raw_data/tool_cache"
export MPLCONFIGDIR="$output/raw_data/tool_cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'code=$?; echo "$code" > "$output/logs/exit_code"' EXIT
.venv/bin/python -m _david.code.conformer.calibrate \
  --output "$output/raw_data" --resume --workers "$workers" \
  >> "$output/logs/run.log" 2>&1
.venv/bin/python -m _david.code.conformer.validate_calibration \
  --output "$output/raw_data" --replays 12 --end-to-end \
  > "$output/logs/validation.log" 2>&1
