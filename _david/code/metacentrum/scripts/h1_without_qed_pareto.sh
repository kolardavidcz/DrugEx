#!/usr/bin/env bash
# ==============================================================================
# DrugEx Sprint — Hypothesis 1 (Supervisor Replication Champion)
# Regime 2: K=1, C=30, unminimized ETKDGv3, eps=0.05, tanimoto_combo, ClippedScore(5,3)
# Multi-objective: without_qed + pareto_crowding
# ==============================================================================
#PBS -N h1_without_qed_pareto
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=10.0:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

# If invoked interactively on cluster frontend, submit to PBS Pro queue
if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting Hypothesis 1 job to MetaCentrum PBS Pro..."
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
EXP_NAME="sprint_h1_without_qed_pareto"

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

AGENT_PKG="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.pkg"
VOCAB_FILE="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.vocab"
if [[ ! -f "$AGENT_PKG" ]]; then
    FALLBACK_DIR="$SCRATCHDIR/DrugEx/tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
    if [[ -f "$FALLBACK_DIR/Papyrus05.5_smiles_rnn_PT.pkg" ]]; then
        AGENT_PKG="$FALLBACK_DIR/Papyrus05.5_smiles_rnn_PT.pkg"
        VOCAB_FILE="$FALLBACK_DIR/Papyrus05.5_smiles_rnn_PT.vocab"
    fi
fi

"$PYTHON" -u -m _david.code.pipeline \
    --output-dir "$OUT_DIR" \
    --epochs 200 \
    --min-epochs 50 \
    --patience 50 \
    --epsilon 0.05 \
    --score-variant tanimoto_combo \
    --no-include-qed \
    --reward-scheme pareto_crowding \
    --no-minimize-candidates \
    --conformers 30 \
    --reference-conformers 1 \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    2>&1 | tee "$OUT_DIR/execution.log"
