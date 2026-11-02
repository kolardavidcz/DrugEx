#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum Job: Run H9 — Dynamic Cosine Epsilon Decay (0.15 -> 0.01)
# Hypothesis: High early prior exploration (eps=0.15) stabilizes chemical diversity
# and grammar validity, while annealing to eps=0.01 over 120 epochs allows sharp
# exploitation that elevates the top 10% ROCS tail beyond 0.76.
# Uses default config.py (minimized conformers C=20, K=4 ensemble, Harrington).
# ==============================================================================
#PBS -N sprint_h9_epsilon_decay
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting Sprint H9 (Dynamic Epsilon Decay) to MetaCentrum..."
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

EXP_NAME="sprint_h9_dynamic_epsilon_decay"
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

echo "=== Launching Sprint H9: Dynamic Epsilon Decay (0.15 -> 0.01) ==="
"$PYTHON" -u -m _david.code.pipeline \
    --output-dir "$OUT_DIR" \
    --score-variant tanimoto_combo \
    --reward-scheme harrington \
    --use-decay \
    --epsilon 0.15 \
    --epsilon-end 0.01 \
    --decay-epochs 120 \
    --decay-type cosine \
    --no-include-qed \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    2>&1 | tee "$OUT_DIR/execution.log"

echo "=== Sprint H9 Completed Successfully ==="
