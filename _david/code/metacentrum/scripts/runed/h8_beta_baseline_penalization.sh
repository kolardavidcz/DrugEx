#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum Job: Run H8 — Baseline Policy Penalization (beta=0.1)
# Hypothesis: Setting beta=0.1 penalizes generating molecules that are already
# probable under the prior baseline, reducing gradient variance and driving
# policy divergence toward novel chemical space.
# Uses default config.py (minimized conformers C=20, K=4 ensemble, Harrington).
# ==============================================================================
#PBS -N sprint_h8_beta_penalization
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting Sprint H8 (Beta Baseline Penalization) to MetaCentrum..."
    exec qsub "$0"
fi

: "${SCRATCHDIR:?PBS scratch allocation is required}"

STORAGE_DIR="${STORAGE_DIR:-}"
if [[ -z "$STORAGE_DIR" ]]; then
    if [[ -d "/storage/brno2/brno2/kolardavidcz/DrugEx" ]]; then
        STORAGE_DIR="/storage/brno2/brno2/kolardavidcz/DrugEx"
    elif [[ -d "/storage/brno2/home/kolardavidcz/DrugEx" ]]; then
        STORAGE_DIR="/storage/brno2/home/kolardavidcz/DrugEx"
    else
        STORAGE_DIR="/auto/vestec1-elixir/home/kolardavidcz/DrugEx"
    fi
fi

EXP_NAME="sprint_h8_beta_baseline_penalization"
PYTHON="$STORAGE_DIR/.venv/bin/python"
test -x "$PYTHON" || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }

SOURCE_DIR="$STORAGE_DIR"
OUT_DIR="$SCRATCHDIR/output"
RESULTS_DIR="$STORAGE_DIR/_david/output/$EXP_NAME"

cleanup() {
    local status=$?
    trap - EXIT
    if test -d "$OUT_DIR"; then
        mkdir -p "$RESULTS_DIR" || true
        rsync -a "$OUT_DIR/" "$RESULTS_DIR/" || echo "Result transfer failed" >&2
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

AGENT_PKG="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.pkg"
VOCAB_FILE="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.vocab"

echo "=== Launching Sprint H8: Beta Baseline Penalization (beta=0.1) ==="
"$PYTHON" -u -m _david.code.pipeline \
    --output-dir "$OUT_DIR" \
    --score-variant tanimoto_combo \
    --reward-scheme harrington \
    --beta 0.1 \
    --no-include-qed \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    2>&1 | tee "$OUT_DIR/execution.log"

echo "=== Sprint H8 Completed Successfully ==="

# ==============================================================================
# CONFIGURASION RUNED at DICHTATOR / METACENTRUM ID: METACENTRUM_25218574
# ==============================================================================
