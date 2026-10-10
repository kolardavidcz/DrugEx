"""
====================================================================================================
BACHELOR THESIS RESEARCH STUDY: LIGAND-BASED DE NOVO MOLECULAR DESIGN IN DRUGEX
COMPARATIVE BENCHMARK OF 3 FUNDAMENTAL PHARMACOPHORIC HYPOTHESES (EGCG / DNMT1 MODEL)
====================================================================================================

Top-level alias forwarding to _david/code/reserch_test/run_hypotheses.py.
See reserch_test/run_hypotheses.py for complete academic notes and mathematical specifications.
"""
from pathlib import Path
import sys

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "drugex").exists() else Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _david.code.reserch_test.run_hypotheses import main

if __name__ == "__main__":
    main()
