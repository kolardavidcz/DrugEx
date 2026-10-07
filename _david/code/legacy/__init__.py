"""Legacy experimental modules superseded by current Graph-Transformer architecture.

These modules are retained for historical comparison and methodological analysis:
1. SequenceRNNOptimizer (SequenceRNN.py): 1D autoregressive SMILES generator.
2. ScaffoldFunnel (scaffold_funnel.py): Bemis-Murcko clustering with fixed catechol filters.
3. USRCATScorer (usrcat_scorer.py): 1D moment approximation (USRCAT/ElectroShape).
"""
from __future__ import annotations

from _david.code.legacy.scaffold_funnel import ScaffoldFunnel
from _david.code.legacy.SequenceRNN import SequenceRNNOptimizer
from _david.code.legacy.usrcat_scorer import USRCATScorer

__all__: list[str] = [
    "ScaffoldFunnel",
    "SequenceRNNOptimizer",
    "USRCATScorer",
]
