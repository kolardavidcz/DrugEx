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

# Pretrained Papyrus05.5 SMILES-RNN: the checkpoint is NOT tracked in git and
# is absent from both the storage repo and staged releases (verified live over
# SSH 2026-11-02), so stage it from the storage copy into the tutorial path the
# module's resolve_pretrained_paths() probes (same source as the h1 sprint
# script's $STORAGE_DIR/models fallback), then fail fast before any walltime
# is spent if the resolver still cannot find it.
MODEL_DIR="$SCRATCHDIR/DrugEx/tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
if [[ -f "$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.pkg" ]]; then
    mkdir -p "$MODEL_DIR"
    cp -f "$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.pkg" \
          "$STORAGE_DIR/models/Papyrus05.5_smiles_rnn_PT.vocab" "$MODEL_DIR/"
elif [[ ! -f "$MODEL_DIR/Papyrus05.5_smiles_rnn_PT.pkg" ]]; then
    echo "Pretrained Papyrus05.5 checkpoint missing: no $STORAGE_DIR/models copy and none in scratch" >&2
    exit 1
fi
"$PYTHON" -c "from _david.code.config import PipelineConfig; PipelineConfig().resolve_pretrained_paths()" \
    || { echo "Pretrained model unresolved in scratch" >&2; exit 1; }

echo "Host: $(hostname) | CPUs: $(nproc) | Revision: $DRUGEX_SOURCE_REVISION"
echo "Starting RL batch calibration (3 arms, 500 samples, 3 repeats, 5 epochs)..."

# Conservative arm set: the two production-decision cells plus the matched
# minimization pair (identical K/C, minimization only; reviewer C1). The
# k=all full-bank arms cost ~35-190 h each at 4 CPUs (reviewer D-2) and must
# be split into separate jobs (or run on ncpus=16) before being added here.
"$PYTHON" -u -m _david.code.conformer.rl_batch_calibration \
    --output "$OUT_DIR" \
    --arms etkdg:1:30,mmff_high_dielectric:4:20,mmff_high_dielectric:1:30 \
    --samples 500 \
    --repeats 3 \
    --epochs 5 \
    --gpu 0 \
    2>&1 | tee "$OUT_DIR/execution.log"
