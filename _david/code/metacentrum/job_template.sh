#!/usr/bin/env bash
# ==============================================================================
# MetaCentrum PBS Pro Job Script: DrugEx 3D De Novo Design Pipeline
# Usage: qsub -v EXP_NAME="run_name",EXP_ARGS="--epochs 100 ..." job_template.sh
# ==============================================================================

#PBS -N drugex_run
#PBS -l select=1:ncpus=16:mem=32gb:ngpus=1:gpu_cap=cuda80:scratch_local=15gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m ae

set -euo pipefail

echo "========================================================================"
echo "Job started on: $(hostname) at $(date)"
echo "Job ID: $PBS_JOBID"
echo "Scratch Directory: $SCRATCHDIR"
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo "========================================================================"

# Trap clean-up and copy results on exit
STORAGE_DIR="/storage/praha1/home/$USER/DrugEx"
RESULTS_DIR="$STORAGE_DIR/_david/output"

cleanup() {
    echo "Job finalizing at $(date). Transferring results..."
    if [ -d "$SCRATCHDIR/output" ]; then
        mkdir -p "$RESULTS_DIR"
        cp -r "$SCRATCHDIR/output/"* "$RESULTS_DIR/" || true
        echo "Results successfully transferred to $RESULTS_DIR"
    fi
    clean_scratch
}
trap 'cleanup' EXIT

# 1. Prepare fast local scratch execution space
cd "$SCRATCHDIR"
mkdir -p "$SCRATCHDIR/DrugEx"
mkdir -p "$SCRATCHDIR/output"

echo "Syncing DrugEx codebase and pretrained weights from persistent storage..."
rsync -a --exclude='.git' --exclude='_david/output' "$STORAGE_DIR/" "$SCRATCHDIR/DrugEx/"

cd "$SCRATCHDIR/DrugEx"

# 2. Environment modules and Python
echo "Loading environment modules..."
module load python/3.10-gcc || module load python || true
module load cuda/12.2 || module load cuda || true

# Check or install standalone uv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure virtualenv exists in scratch
echo "Synchronizing virtual environment..."
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install scipy seaborn ruff

# 3. Hardware check
nvidia-smi || true
lscpu | head -n 20 || true

# 4. Execute DrugEx pipeline
EXP_NAME="${EXP_NAME:-drugex_experiment}"
EXP_ARGS="${EXP_ARGS:- --epochs 100 --samples 500 --eval-samples 1000}"

echo "Starting Experiment: $EXP_NAME"
echo "Arguments: $EXP_ARGS"

OUT_DIR="$SCRATCHDIR/output/$EXP_NAME"
mkdir -p "$OUT_DIR"

python -u _david/code/pipeline.py \
    --output-dir "$OUT_DIR" \
    $EXP_ARGS 2>&1 | tee "$OUT_DIR/execution.log"

echo "Experiment completed successfully at $(date)!"
