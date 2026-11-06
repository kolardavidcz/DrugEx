#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum Job: EGCG K=4 C=20 Candidates Unminimized (ETKDGv3)
# Reference: MMFF94s eps=78.4 K=4 diverse ensemble (EGCG_ensemble_mmff94s_k4.sdf)
# Candidates: C=20 ETKDGv3 conformers per molecule UNMINIMIZED (--no-minimize-candidates)
# Training: eps=0.10, 50 epochs, 500 samples/epoch, 16 CPUs, fast CPU queue
# ==============================================================================
#PBS -N egcg_k4_c20_unmin
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submitting EGCG K=4 C=20 Unminimized job to MetaCentrum..."
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
EXP_NAME="egcg_k4_c20_candidates_unminimized"

PYTHON="$STORAGE_DIR/.venv/bin/python"
test -x "$PYTHON" || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }

SOURCE_DIR="${SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" ]]; then
    SOURCE_DIR="$STORAGE_DIR"
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

REF_SDF="$STORAGE_DIR/_david/data/reference/EGCG_ensemble_mmff94s_k4.sdf"
if [[ ! -f "$REF_SDF" ]]; then
    REF_SDF="$SCRATCHDIR/DrugEx/_david/data/reference/EGCG_ensemble_mmff94s_k4.sdf"
fi

echo "=== System Diagnostic ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Reference SDF: $REF_SDF"

"$PYTHON" -u -m _david.code.training.pipeline \
    --output-dir "$OUT_DIR" \
    --molecule "$REF_SDF" \
    --reference-conformers 4 \
    --conformers 20 \
    --no-minimize-candidates \
    --epsilon 0.10 \
    --epochs 50 \
    --min-epochs 50 \
    --patience 50 \
    --samples 500 \
    --score-variant tanimoto_combo \
    --no-include-qed \
    --reward-scheme pareto_crowding \
    --agent-path "$AGENT_PKG" \
    --vocab-path "$VOCAB_FILE" \
    --gpu cpu \
    --cpu-ratio 1.0 \
    2>&1 | tee "$OUT_DIR/execution.log"

echo "=== Job Completed Successfully ==="

# ==============================================================================
# CONFIGURASION RUNED at DICHTATOR / METACENTRUM ID: METACENTRUM_25223669
# ==============================================================================
