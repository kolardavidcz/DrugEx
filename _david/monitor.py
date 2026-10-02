"""Root entrypoint for DrugEx LBDD pipeline telemetry monitor.

Usage:
    python _david/monitor.py            # Show telemetry snapshot of latest run
    python _david/monitor.py --watch    # Live real-time dashboard refresh
    python _david/monitor.py -d <path>  # Inspect specific run directory
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root and _david/code are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _david.code.monitor import main

if __name__ == "__main__":
    sys.exit(main())
