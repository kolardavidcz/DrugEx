#!/usr/bin/env bash
set -euo pipefail
cd /home/kolarv/DrugEx
mkdir -p /home/kolarv/DrugEx/_david/output
exec /home/kolarv/DrugEx/.venv/bin/python -u /home/kolarv/DrugEx/_david/code/run_overnight_matrix.py 2>&1 | tee /home/kolarv/DrugEx/_david/output/overnight_matrix.log
