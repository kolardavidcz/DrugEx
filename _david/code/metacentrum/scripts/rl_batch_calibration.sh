#!/usr/bin/env bash
# ==============================================================================
# DrugEx — 1:1 Training-Interface Cost & Early-Accuracy Calibration
# Measures time x ROCS accuracy with the EXACT production scoring interface
# (full-epoch unique batches, ~500 molecules per env.getScores call) across
# conformer methods, plus 5-epoch RL pilots per arm through the real
# explorer.fit (sprint H1 parity: 2-objective [ROCS, SA], pareto_crowding,
# eps=0.05, tanimoto_combo, no QED).
# Arms are paired (identical frozen-prior inputs), isolated (fresh subprocess
# per arm), and resumable (per-arm complete.json markers).
# ==============================================================================
#PBS -N rl_batch_calib
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=15gb:spec=8.0:ngpus=1
#PBS -l walltime=25:00:00
#PBS -q gpu_long
#PBS -j oe

set -euo pipefail

# If invoked interactively on cluster frontend, submit to PBS Pro queue
if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting RL batch calibration job to MetaCentrum PBS Pro..."
    exec qsub "$0"
fi

: "${SCRATCHDIR:?PBS scratch allocation is required}"

# Resolve canonical storage directory (supporting both /storage/brno2/brno2 and /storage/brno2/home)
STORAGE_DIR="${STORAGE_DIR:-}"
if [[ -z "$STORAGE_DIR" ]]; then
    if [[ -d "/storage/brno2/brno2/kolardavidcz/DrugEx" ]]; then
        STORAGE_DIR="/storage/brno2/brno2/kolardavidcz/DrugEx"
    elif [[ -d "/storage/brno2/home/kolardavidcz/DrugEx" ]]; then
        STORAGE_DIR="/storage/brno2/home/kolardavidcz/DrugEx"
    else
        STORAGE_DIR="/storage/brno2/brno2/kolardavidcz/DrugEx"
    fi
fi
EXP_NAME="rl_batch_calibration"

PYTHON="$STORAGE_DIR/.venv/bin/python"
test -x "$PYTHON" || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }

# Resolve source code directory (direct repo or staged release)
SOURCE_DIR="${SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" ]]; then
    if [[ -d "$STORAGE_DIR/_david" ]]; then
        SOURCE_DIR="$STORAGE_DIR"
    elif [[ -d "$STORAGE_DIR/releases" ]]; then
        LATEST_REL="$(ls -td "$STORAGE_DIR"/releases/* 2>/dev/null | head -n 1 || true)"
        if [[ -n "$LATEST_REL" && -d "$LATEST_REL/_david" ]]; then
            SOURCE_DIR="$LATEST_REL"
        else
            SOURCE_DIR="$STORAGE_DIR"
        fi
    else
        SOURCE_DIR="$STORAGE_DIR"
    fi
fi

OUT_DIR="$SCRATCHDIR/output"
RESULTS_DIR="$STORAGE_DIR/_david/output/$EXP_NAME"

# Resume support: pull any previously completed results into scratch so the
# module's per-arm complete.json markers skip finished arms on resubmission
# (a walltime-killed job continues where it stopped instead of restarting).
if [[ -d "$RESULTS_DIR/raw_data" ]]; then
    mkdir -p "$OUT_DIR"
    rsync -a "$RESULTS_DIR/" "$OUT_DIR/" || {
        echo "Could not restore previous results from $RESULTS_DIR" >&2
        exit 1
    }
    echo "Resumed from previous results in $RESULTS_DIR"
fi

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
rsync -a --exclude='.venv' --exclude='_david/output' "$SOURCE_DIR/" "$SCRATCHDIR/DrugEx/"
cd "$SCRATCHDIR/DrugEx"

export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DRUGEX_MP_CONTEXT=forkserver
export PYTHONPATH="$SCRATCHDIR/DrugEx:${PYTHONPATH:-}"
# Scratch has no .git; record the staged release revision for manifests.
export DRUGEX_SOURCE_REVISION="$(basename "$SOURCE_DIR")"

# Pretrained Papyrus05.5 SMILES-RNN: resolved by the module via
# PROJECT_ROOT/tutorial/data/models/pretrained (rsynced with the source tree,
# same fallback the h1 sprint script relies on).

echo "Host: $(hostname) | CPUs: $(nproc) | Revision: $DRUGEX_SOURCE_REVISION"
echo "Starting RL batch calibration (6 arms, 500 samples, 3 repeats, 5 epochs)..."

"$PYTHON" -u -m _david.code.conformer.rl_batch_calibration \
    --output "$OUT_DIR" \
    --samples 500 \
    --repeats 3 \
    --epochs 5 \
    --gpu 0 \
    2>&1 | tee "$OUT_DIR/execution.log"
