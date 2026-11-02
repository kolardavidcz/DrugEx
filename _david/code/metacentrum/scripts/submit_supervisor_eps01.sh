#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum Job: Supervisor Reproduction (EGCG ROCS RL, eps=0.1, 50 epochs)
# Reference: https://github.com/fulopjoz/DrugEx/blob/egcg-rocs-tutorial/tutorial/advanced/rocs/egcg/final_out/eps0.1/executed_eps0.1.ipynb
# ==============================================================================
#PBS -N supervisor_eps0.1
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q q_4h
#PBS -j oe

set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting supervisor reproduction job to MetaCentrum q_4h queue..."
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

EXP_NAME="supervisor_reproduction_eps0.1"
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

mkdir -p "$OUT_DIR" "$SCRATCHDIR/DrugEx"
rsync -a --exclude='.venv' --exclude='_david/output' "$STORAGE_DIR/" "$SCRATCHDIR/DrugEx/"
cd "$SCRATCHDIR/DrugEx"

export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DRUGEX_MP_CONTEXT=forkserver
export PYTHONPATH="$SCRATCHDIR/DrugEx:${PYTHONPATH:-}"

AGENT_PKG="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.pkg"
VOCAB_FILE="$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.vocab"
REF_SDF="$SCRATCHDIR/DrugEx/_david/data/reference/EGCG_reference.sdf"

"$PYTHON" -u -m _david.code.reproduce_supervisor_eps01 \
    --output-dir "$OUT_DIR" \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    --reference-sdf "$REF_SDF" \
    --epsilon 0.1 \
    --epochs 50 \
    --n-samples 500 \
    --n-jobs 14 \
    --device "cuda:0" \
    2>&1 | tee "$OUT_DIR/execution.log"
