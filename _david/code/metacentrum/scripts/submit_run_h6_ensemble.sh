#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum Job: Run H6 - Multi-Conformer Reference Ensemble (K=4)
# Hypothesis: Benchmarking multi-conformer reference ensemble (K=4) vs
# single-conformer reference (K=1, supervisor baseline) under eps=0.1 RL.
# ==============================================================================
#PBS -N h6_egcg_ensemble_k4
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting Run H6 (Multi-Conformer Ensemble) to MetaCentrum..."
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

EXP_NAME="run_h6_egcg_ensemble_k4"
PYTHON="$STORAGE_DIR/.venv/bin/python"
test -x "$PYTHON" || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }

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

mkdir -p "$OUT_DIR"
cd "$STORAGE_DIR"

echo "=== System Diagnostic ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $PBS_JOBID"
echo "GPU:"
nvidia-smi || true

REF_SDF="$STORAGE_DIR/_david/data/reference/EGCG_ensemble_4conf.sdf"
if [[ ! -f "$REF_SDF" ]]; then
    echo "Error: Reference SDF $REF_SDF not found!" >&2
    exit 1
fi

AGENT_PKG="$STORAGE_DIR/_david/data/models/Papyrus05.5_smiles_rnn_PT_model_SequenceRNN_000024.pkg"
VOCAB_FILE="$STORAGE_DIR/_david/data/models/Papyrus05.5_smiles_rnn_PT_model_SequenceRNN.vocab"

echo "=== Launching Run H6: Multi-Conformer Ensemble (K=4) ==="
"$PYTHON" _david/code/reproduce_supervisor_eps01.py \
    --output-dir "$OUT_DIR" \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    --reference-sdf "$REF_SDF" \
    --epsilon 0.1 \
    --epochs 50 \
    --n-samples 500 \
    --n-jobs 16 \
    --device cuda:0

echo "=== Run H6 Completed Successfully ==="

# ==============================================================================
# CONFIGURASION RUNED at DICHTATOR / METACENTRUM ID: METACENTRUM_25217210
# ==============================================================================
