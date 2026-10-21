#!/usr/bin/env bash
set -e

RUN11_PID=3258600
echo "[Chain Orchestrator] Monitoring Run 11 (PID ${RUN11_PID})..."

while kill -0 ${RUN11_PID} 2>/dev/null; do
    sleep 5
done

echo "[Chain Orchestrator] Run 11 has completed. Waiting 5s before launching Run 12..."
sleep 5

echo "[Chain Orchestrator] Launching Run 12 (Pareto + Threshold 0.70)..."
cd /home/kolarv/DrugEx
/home/kolarv/DrugEx/.venv/bin/python _david/code/pipeline.py \
    --detach \
    --output-dir _david/output/overnight_runs/run12_pareto_threshold_070 \
    --agent-path _david/output/overnight_runs/run10_chroman_exploration_160plus/models/david_agent_peak.pkg \
    --batch-size 64 \
    --n-samples 192 \
    --val-samples 100 \
    --n-eval-samples 200 \
    --epochs 35 \
    --min-epochs 20 \
    --patience 10 \
    --epsilon 0.22 \
    --beta 0.05 \
    --thresholds 0.70 0.50 0.50 \
    --criteria desired_ratio \
    --reward-scheme pareto \
    --score-variant tversky_subshape

echo "[Chain Orchestrator] Run 12 spawned successfully."
