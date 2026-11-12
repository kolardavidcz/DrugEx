#!/usr/bin/env bash
# Hand-qsubbed triage rebuild: stop arms for the runs a walltime kill left unfinished.
#
# The collapse verdict reads only final_desired_ratio_at_stop.tsv, so one rebuilt arm
# upgrades an unfinished run from 'incomplete' to a real verdict. gmean stays missing --
# it is passive evidence. Two rebuilds share the node: 16 scoring workers each.
#
# qsub from the release root (submit.py syncs this file with the tree):
#   qsub -N rebuild_arms -l select=1:ncpus=32:mem=64gb:spec=8.0 \
#        -l walltime=12:00:00 -v RELEASE_DIR=$PWD,STORAGE_DIR=STORAGE \
#        _david/code/metacentrum/rebuild_arms.sh
set -uo pipefail

: "${RELEASE_DIR:?set by qsub -v}"
STORAGE_DIR="${STORAGE_DIR:-/storage/brno2/brno2/kolardavidcz/DrugEx}"
PYTHON="$STORAGE_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "missing validated environment: $PYTHON" >&2; exit 1; }

OUTPUT="$STORAGE_DIR/_david/output"
LOGS="$STORAGE_DIR/_david/output/rebuild_logs"
mkdir -p "$LOGS"
cd "$RELEASE_DIR"

RUNS=(
  "20261111_0443_p_m1e00r1c20q1"
  "20261111_0522_p_m0e00r1c30q0"
  "20261111_0540_c_e00fds20q1"
  "20261111_0625_c_e00fos10q0"
  "20261111_0625_c_e00fos10q1"
  "20261111_0444_p_m1e02r4c20q1"
  "20261111_0445_p_m1e02r1c20q0"
  "20261111_0506_p_m0e02r4c30q0"
  "20261111_0507_p_m0e02r1c30q0"
  "20261111_0443_p_m1e10r4c20q0"
  "20261111_0443_p_m1e10r4c20q1"
  "20261111_0444_p_m0e10r4c30q0"
  "20261111_0450_p_m0e10r1c30q0"
  "20261111_0441_p_m1e05r4c20q0"
  "20261111_0441_p_m1e05r4c20q1"
  "20261111_0444_w_e05q0"
  "20261111_0449_p_m0e05r4c30q0"
  "20261111_0454_p_m0e05r1c30q0"
  "20261111_0521_w_e05q1"
  "20261111_0522_c_e05fds10q0"
  "20261111_0522_c_e05fds20q0"
  "20261111_0522_c_e05fds20q1"
  "20261111_0522_c_e05fds30q0"
  "20261111_0522_c_e05fds30q1"
  "20261111_0522_c_e05fds50q0"
  "20261111_0523_c_e05fds50q1"
  "20261111_0525_c_e05fds10q1"
  "20261111_0529_c_e05ffs10q0"
  "20261111_0529_c_e05ffs50q1"
  "20261111_0530_c_e05ffs50q0"
  "20261111_0530_c_e05fos10q1"
  "20261111_0530_c_e05fos20q0"
  "20261111_0530_c_e05fos20q1"
  "20261111_0530_c_e05fos30q0"
  "20261111_0530_c_e05fos30q1"
  "20261111_0530_c_e05fos50q0"
  "20261111_0530_c_e05fos50q1"
  "20261111_0531_c_e05ffs10q1"
)

rebuild_one() {
  local run="$1"
  echo "[$(date +%H:%M:%S)] start $run"
  timeout 7200 "$PYTHON" -m _david.code.score_arms "$OUTPUT/$run" \
    --only desired_ratio_at_stop --n-jobs 16 \
    > "$LOGS/$run.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] done  $run" \
    || echo "[$(date +%H:%M:%S)] FAIL  $run (exit $?, see $LOGS/$run.log)"
}

# Two lanes, 16 workers each; the list is ordered so the interesting epsilons
# (e00, e02, e10) finish first even if the walltime cuts the job short.
for lane in 0 1; do (
  i=0
  for run in "${RUNS[@]}"; do
    (( i++ % 2 == lane )) || continue
    rebuild_one "$run"
  done
) & done
wait
echo "rebuild finished: $(grep -l desired_ratio_at_stop $OUTPUT/*/final_desired_ratio_at_stop.tsv 2>/dev/null | wc -l) stop arms present"
