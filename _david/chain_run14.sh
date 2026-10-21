#!/usr/bin/env bash
set -e

echo "[Run 14 Orchestrator] Verifying Run 13 peak agent checkpoint..."
PEAK_PKG="/home/kolarv/DrugEx/_david/output/overnight_runs/run13_harrington_desirability/models/david_agent_peak.pkg"
if [ ! -f "${PEAK_PKG}" ]; then
    echo "[ERROR] Checkpoint ${PEAK_PKG} not found!"
    exit 1
fi

echo "[Run 14 Orchestrator] Launching Run 14 (Harrington Desirability with Scheduled Epsilon Decay)..."
cd /home/kolarv/DrugEx
/home/kolarv/DrugEx/.venv/bin/python _david/code/pipeline.py \
    --detach \
    --output-dir _david/output/overnight_runs/run14_harrington_epsilon_decay \
    --agent-path "${PEAK_PKG}" \
    --batch-size 64 \
    --n-samples 192 \
    --val-samples 100 \
    --n-eval-samples 200 \
    --epochs 100 \
    --min-epochs 50 \
    --patience 25 \
    --epsilon 0.25 \
    --epsilon-end 0.05 \
    --decay-epochs 50 \
    --decay-type cosine \
    --beta 0.05 \
    --thresholds 0.70 0.50 0.50 \
    --criteria avg_amean \
    --reward-scheme harrington \
    --score-variant tversky_subshape

echo "[Run 14 Orchestrator] Run 14 spawned successfully."
