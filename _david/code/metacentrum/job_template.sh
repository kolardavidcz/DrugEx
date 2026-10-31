#!/usr/bin/env bash
# Submit through grid.py: arguments live in a JSON spec, never an eval string.
#PBS -N drugex_run
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=10:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
: "${PBS_JOBID:?Run this script as a PBS job}"
: "${SCRATCHDIR:?PBS scratch allocation is required}"
: "${STORAGE_DIR:?Persistent project directory is required}"
: "${RUN_SPEC:?A saved JSON run specification is required}"

# Prepare and validate this environment before submitting a research run.
PYTHON="$STORAGE_DIR/.venv/bin/python"
test -x "$PYTHON" || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }
EXP_NAME=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$RUN_SPEC")
SOURCE_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$RUN_SPEC")
export DRUGEX_SOURCE_REVISION
DRUGEX_SOURCE_REVISION=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$RUN_SPEC")
OUT_DIR="$SCRATCHDIR/output"
RESULTS_DIR="$STORAGE_DIR/_david/output/$EXP_NAME"

cleanup() {
    local status=$?
    trap - EXIT
    if test -d "$OUT_DIR"; then
        if ! mkdir -p "$RESULTS_DIR" || ! rsync -a "$OUT_DIR/" "$RESULTS_DIR/"; then
            echo "Result transfer failed; retaining scratch at $SCRATCHDIR" >&2
            exit 1
        fi
    fi
    if command -v clean_scratch >/dev/null 2>&1; then
        clean_scratch
    else
        rm -rf "$SCRATCHDIR"/*
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "$OUT_DIR" "$SCRATCHDIR/DrugEx"
rsync -a "$SOURCE_DIR/" "$SCRATCHDIR/DrugEx/"
cd "$SCRATCHDIR/DrugEx"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DRUGEX_MP_CONTEXT=forkserver
export PYTHONPATH="$SCRATCHDIR/DrugEx:${PYTHONPATH:-}"
cp "$RUN_SPEC" "$OUT_DIR/submission.json"

"$PYTHON" - "$RUN_SPEC" "$OUT_DIR" <<'PY' 2>&1 | tee "$OUT_DIR/execution.log"
import json
import subprocess
import sys

with open(sys.argv[1]) as handle:
    spec = json.load(handle)
subprocess.run(
    [sys.executable, "-u", "-m", "_david.code.pipeline", "--output-dir", sys.argv[2], *spec["args"]],
    check=True,
)
PY
