#!/usr/bin/env bash
set -e

echo "[Run 15 Orchestrator] Verifying Run 14 peak agent checkpoint..."
PEAK_PKG="/home/kolarv/DrugEx/_david/output/overnight_runs/run14_harrington_epsilon_decay/models/david_agent_peak.pkg"
if [ ! -f "${PEAK_PKG}" ]; then
    echo "[ERROR] Checkpoint ${PEAK_PKG} not found!"
    exit 1
fi

echo "[Run 15 Orchestrator] Launching Run 15 (Deep Conformer Sampling & Low-Noise Fine-Tuning)..."
cd /home/kolarv/DrugEx
/home/kolarv/DrugEx/.venv/bin/python _david/code/pipeline.py \
    --detach \
    --output-dir _david/output/overnight_runs/run15_harrington_deep_sampling \
    --agent-path "${PEAK_PKG}" \
    --batch-size 64 \
    --n-samples 192 \
    --val-samples 100 \
    --n-eval-samples 200 \
    --epochs 60 \
    --min-epochs 35 \
    --patience 20 \
    --epsilon 0.12 \
    --epsilon-end 0.02 \
    --decay-epochs 35 \
    --decay-type cosine \
    --max-conformers 50 \
    --max-reference-conformers 10 \
    --beta 0.05 \
    --thresholds 0.70 0.50 0.50 \
    --criteria avg_amean \
    --reward-scheme harrington \
    --score-variant tversky_subshape

echo "[Run 15 Orchestrator] Run 15 spawned successfully."
