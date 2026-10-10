"""Backward-compatibility forwarding module for Molecular Space Visualizer."""
from __future__ import annotations

from typing import Any


def MolecularSpacePipeline(*args: Any, **kwargs: Any) -> Any:
    """Lazy loader for MolecularSpacePipeline to prevent C-level openbabel conflicts."""
    from _david.code.analysis.manifold import MolecularSpacePipeline as _Pipeline
    return _Pipeline(*args, **kwargs)


__all__: list[str] = ["MolecularSpacePipeline"]
