#!/usr/bin/env bash
# Main-checkout execution only; no source worktrees or home-directory scratch.
set -euo pipefail
cd "$(dirname "$0")/../../.."
run="$PWD/_david/output/20261029_0250_reward_calibration"
panel="$PWD/_david/output/20261028_2145_joint_calibration/raw_data/panel.csv"
mode="${1:-pilot}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$run/raw_data/tmp" XDG_CACHE_HOME="$run/raw_data/tool_cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export DRUGEX_MP_CONTEXT=forkserver
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$run/logs"

if [[ "$mode" == pilot ]]; then
  .venv/bin/python -m _david.code.conformer.reward_calibration \
    --panel "$panel" --output "$run" --phase pilot --limit 2 \
    --candidates 10,20 --caps 1,4 --workers 2
  exit 0
fi
if [[ "$mode" != full ]]; then echo "Use pilot or full" >&2; exit 2; fi

# One bounded sample of free CPU and available RAM, with room for other users.
workers="${DRUGEX_STUDY_WORKERS:-$(.venv/bin/python - <<'PY'
from pathlib import Path
import os, time
def cpu():
    return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
before=cpu(); time.sleep(1); after=cpu()
total=sum(after)-sum(before)
idle=(after[3]+after[4]-before[3]-before[4])/max(1,total)
available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
print(max(1,min(32,int(os.cpu_count()*idle*.8),int(available*.7/(600*1024)))))
PY
)}"
printf 'Whole-input workers: %s\n' "$workers"
.venv/bin/python -m _david.code.conformer.reward_calibration \
  --panel "$panel" --output "$run" --phase development --workers "$workers"
.venv/bin/python -m _david.code.conformer.validate_reward_calibration --output "$run" --phase development
.venv/bin/python -m _david.code.conformer.reward_analysis --output "$run"
.venv/bin/python -m _david.code.conformer.reward_calibration \
  --panel "$panel" --output "$run" --phase confirmation --split confirmation \
  --settings "$run/raw_data/development/shortlist.json" --workers "$workers"
.venv/bin/python -m _david.code.conformer.validate_reward_calibration --output "$run" --phase confirmation
.venv/bin/python -m _david.code.conformer.reward_analysis --output "$run"
