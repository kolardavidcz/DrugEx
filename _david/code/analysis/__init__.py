"""Result analysis, Pareto frontier ranking, and aligned depictions for LBDD."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from _david.code.analysis.depiction import AlignedDepicter
from _david.code.analysis.pareto import ParetoFrontier
from _david.code.analysis.result_analyzer import (
    EGCG_SMILES,
    MY_MOLECULE,
    ResultAnalyzer,
)

if TYPE_CHECKING:
    from _david.code.analysis.manifold import MolecularSpacePipeline


def get_molecular_space_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Lazy loader for MolecularSpacePipeline avoiding eager openbabel/qsprpred C-extension conflicts."""
    from _david.code.analysis.manifold import MolecularSpacePipeline
    return MolecularSpacePipeline(*args, **kwargs)


__all__: list[str] = [
    "AlignedDepicter",
    "ParetoFrontier",
    "ResultAnalyzer",
    "MY_MOLECULE",
    "EGCG_SMILES",
    "get_molecular_space_pipeline",
]
