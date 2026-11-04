#!/usr/bin/env bash
# PBS launcher — the only shell left in a job.
#
# qsub executes this file; it checks the two variables submit.py passes and then
# replaces itself with the validated interpreter, which runs job.py. All job logic
# (run ids, release copy, argument forwarding, result transfer, signal exit codes)
# lives in job.py so it can be unit-tested instead of read as a bash trap.
#
# Do not add logic here. `submit.py` stages the release, writes the JSON spec and runs:
#     qsub -N <name> -l select=… -l walltime=… -v RUN_SPEC=…,STORAGE_DIR=… <this file>
# The #PBS lines below are fallback defaults for a hand-run `qsub`.
#PBS -N drugex_run
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
: "${STORAGE_DIR:?Set STORAGE_DIR (submit.py does this)}"
: "${RUN_SPEC:?Set RUN_SPEC (submit.py does this)}"

RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$STORAGE_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "Missing validated environment: $PYTHON" >&2; exit 1; }

exec "$PYTHON" "$RELEASE/_david/code/metacentrum/job.py" --release "$RELEASE"
