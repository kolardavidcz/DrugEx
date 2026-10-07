"""Archived historical scripts and legacy components.

These modules are preserved strictly for historical reference and academic transparency:
- 1st_code_david.py: Initial September 2026 receptor-similar Papyrus exploration.
- 1st_code_receptor_similar.py: ChEMBL bioactivity and target deconvolution.
- SequenceRNN.py: 1D SMILES RNN generator with USRCAT reward.
- scaffold_funnel.py: Multi-stage scaffold filtering funnel.
- usrcat_scorer.py: 1D moment approximation scorer (USRCAT/ElectroShape).
- thesis_benchmark.py: Multi-model comparative benchmark suite.
- visualize_molecular_space.py: High-dimensional chemical space manifold analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add legacy scripts directory to sys.path if not present for relative imports
_legacy_dir = str(Path(__file__).resolve().parent)
if _legacy_dir not in sys.path:
    sys.path.insert(0, _legacy_dir)

try:
    from _david.output.archive.legacy_scripts.scaffold_funnel import ScaffoldFunnel
    from _david.output.archive.legacy_scripts.SequenceRNN import SequenceRNNOptimizer
    from _david.output.archive.legacy_scripts.usrcat_scorer import USRCATScorer
except (ImportError, ModuleNotFoundError):
    ScaffoldFunnel = None  # type: ignore
    SequenceRNNOptimizer = None  # type: ignore
    USRCATScorer = None  # type: ignore

__all__ = [
    "ScaffoldFunnel",
    "SequenceRNNOptimizer",
    "USRCATScorer",
]
