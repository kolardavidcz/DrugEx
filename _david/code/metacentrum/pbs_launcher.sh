#!/usr/bin/env bash
# PBS launcher: the only shell in a job, and it holds no logic.
#
# PBS spools a copy of this file outside the release, so it must not derive the release
# from its own path. `submit.py` passes RELEASE_DIR, RUN_SPEC and STORAGE_DIR through
# `qsub -v`, and this script only validates them and execs the interpreter on storage.
# Everything else -- run naming, the scratch copy, thread pinning, result transfer, signal
# exit codes -- lives in job.py, where it can be read and tested as Python.
#
# The #PBS lines are fallbacks for a hand-run qsub; submit.py passes the real resources.
#PBS -N drugex_v2
#PBS -l select=1:ncpus=16:mem=64gb:scratch_local=15gb:spec=8.0
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
: "${STORAGE_DIR:?set by submit.py}"
: "${RUN_SPEC:?set by submit.py}"
: "${RELEASE_DIR:?set by submit.py}"

PYTHON="$STORAGE_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "missing validated environment: $PYTHON" >&2; exit 1; }

exec "$PYTHON" "$RELEASE_DIR/_david/code/metacentrum/job.py" --release "$RELEASE_DIR"
