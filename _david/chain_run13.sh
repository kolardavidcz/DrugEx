#!/usr/bin/env bash
set -e

RUN12_PID=204388
echo "[Chain Orchestrator] Monitoring Run 12 (PID ${RUN12_PID})..."

while kill -0 ${RUN12_PID} 2>/dev/null; do
    sleep 5
done

echo "[Chain Orchestrator] Run 12 has completed. Waiting 5s before launching Run 13..."
sleep 5

echo "[Chain Orchestrator] Launching Run 13 (Harrington Desirability)..."
cd /home/kolarv/DrugEx
/home/kolarv/DrugEx/.venv/bin/python _david/code/pipeline.py \
    --detach \
    --output-dir _david/output/overnight_runs/run13_harrington_desirability \
    --agent-path _david/output/overnight_runs/run12_pareto_threshold_070/models/david_agent_peak.pkg \
    --batch-size 64 \
    --n-samples 192 \
    --val-samples 100 \
    --n-eval-samples 200 \
    --epochs 35 \
    --min-epochs 20 \
    --patience 10 \
    --epsilon 0.20 \
    --beta 0.05 \
    --thresholds 0.70 0.50 0.50 \
    --criteria avg_amean \
    --reward-scheme harrington \
    --score-variant tversky_subshape

echo "[Chain Orchestrator] Run 13 spawned successfully."
