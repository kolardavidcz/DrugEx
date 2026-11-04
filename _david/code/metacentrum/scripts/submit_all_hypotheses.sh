#!/usr/bin/env bash
# ==============================================================================
# DrugEx Sprint — Batch Submitter for all 4 Hypotheses
# Submits H1, H2, H3, and H4 to MetaCentrum PBS Pro queue
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# If running directly on MetaCentrum frontend where qsub is present
if command -v qsub >/dev/null 2>&1; then
    echo "Submitting all 4 hypotheses via qsub on $(hostname)..."
    for script in "$SCRIPT_DIR"/h*.sh; do
        echo "Submitting $script..."
        qsub "$script"
    done
    exit 0
fi

# If running locally from repository with Python venv, use grid CLI submit-sprint
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    "$PROJECT_ROOT/.venv/bin/python" -m _david.code.metacentrum.grid submit-sprint "$@"
    exit 0
fi

echo "Error: Neither qsub on cluster frontend nor local project virtualenv was found." >&2
exit 1

# ==============================================================================
# CONFIGURASION RUNED at DICHTATOR / METACENTRUM ID: METACENTRUM_SPRINT_H1_H4_BATCH
# ==============================================================================
