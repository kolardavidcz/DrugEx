#!/usr/bin/env bash
# Whole scoring calls on a small frozen batch; no RNN training.
set -euo pipefail
cd "$(dirname "$0")/../../.."
run="$PWD/_david/output/20261029_0208_reward_cost"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$run/raw_data/tmp"
export MPLCONFIGDIR="$run/raw_data/tool_cache/matplotlib"
export XDG_CACHE_HOME="$run/raw_data/tool_cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$run/logs"
set +e
.venv/bin/python -m _david.code.conformer.diagnostics timing \
  --source "$PWD/_david/output/20261028_2145_joint_calibration/raw_data" \
  --output "$run" --batch-size 32 --repeats 2 > "$run/logs/batch_timing.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run/logs/timing_exit_code"
exit "$result"
