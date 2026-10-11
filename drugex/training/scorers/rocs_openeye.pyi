"""
OpenEye-ROCS based ROCS scorer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
OE_AVAILABLE: bool = ...

import os
import shutil
import subprocess
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from drugex.training.scorers.interfaces import ConformerGenerator
from drugex.training.scorers.interfaces import Scorer


@contextmanager
def _managed_tmpdir() -> Any:
    """Managed temporary directory with guaranteed cleanup."""

    ...


class OpenEyeROCSScorer(Scorer):
    """
    OpenEye ROCS scorer that shells out to the ROCS command-line binary.

    Uses the OpenEye ROCS CLI via ``subprocess`` for shape-based similarity
    scoring while relying on the OpenEye Python toolkits for reference validation
    and RDKit for molecule handling. Requires a valid OpenEye license with the
    ROCS binary available in ``PATH`` (or provided via ``binary_path``). If a
    future fastROCS implementation is added, this class will remain the CLI
    variant.

    Features:
    - Multiple reference group support (.sq or molecule files)
    - Best score selection across reference groups
    - Hybrid Python/CLI workflow for flexible integration

    Attributes:
        - conformer_generator: conformer generator used for generated molecules.
        - references: dict of reference files for ROCS scoring (.sq or molecule files).
            Dictionary keys are reference group names, values are file paths or lists
            of file paths. For each key (group), one score is returned per molecule.
            If a list of files is provided for a single group, the highest score across
            all reference files in that group is returned.
        - score_type: Type of scoring to use (e.g., TanimotoCombo)
        - shape_only: If True, only shape scoring is performed
        - optimize: If True, optimization is performed during scoring
        - color_optimize: If True, color optimization is performed
        - color_force_field: Force field to use for color optimization
        - rocs_binary: Name of the ROCS binary to use
        - binary_path: Path to the ROCS binary (if not in PATH)
        - show_progress: If True, progress is shown during scoring
        - name_suffix: Optional suffix for the scorer name (important if multiple
           OpenEyeROCSScorer scorers are used in the same environment)
    """

    def __init__(self, conformer_generator: ConformerGenerator, references: Dict[str, List[str] | str], score_type: str = ..., shape_only: bool = ..., optimize: bool = ..., color_optimize: bool = ..., color_force_field: str = ..., rocs_binary: str = ..., binary_path: str | None = ..., show_progress: bool = ..., n_jobs: int = ..., timeout: int = ..., mpi_np: int = ..., optimization_mode: str | None = ..., score_variant: str = ...) -> None:
        """
        Initialize the OpenEye ROCS scorer.

        Args:
            conformer_generator: generator used to produce generated molecule conformers
            references: dict mapping reference group names to SDF/sq reference file path(s).
                If multiple dict items are specified, the output will be a score per group.
                If a dict item has multiple file paths specified, molecules will be scored
                against each reference file but only the maximum score will be returned
                (i.e. the best match within that group).
            score_type: Type of scoring to use (e.g., TanimotoCombo), ignored
                if shape_only
            shape_only: If True, only shape scoring is performed
            optimize: turn optimizer on/off, if off score only
            color_optimize: If True, color optimization is performed
            color_force_field: Force field to use for color optimization
            rocs_binary: Name of the ROCS binary to use
            binary_path: Path to the ROCS binary (if not in PATH)
            show_progress: If True, progress is shown during scoring
            n_jobs: Accepted for API consistency with RDKit/CDPKit backends
                but has no effect (OpenEye ROCS runs as a single CLI subprocess)
            timeout: Timeout in seconds for the ROCS subprocess

        Raises:
            ImportError: If OpenEye toolkits are not available.
            FileNotFoundError: If rocs binary not found
        """

        ...

    def _validate_query_files(self) -> Any:
        """Validate all reference files exist and are readable"""

        ...

    def _convert_to_smiles(self, mols: List[Chem.Mol]) -> List[str | None]:
        """Convert various molecule types to SMILES"""

        ...

    @staticmethod
    def _deduplicate_smiles(smiles_list: List[Union[str, None]]) -> Tuple[List[str], Dict[int, List[int]]]:
        """Group identical SMILES to avoid redundant conformer generation."""

        ...

    def getScores(self, mols: Sequence[Union[str, Chem.Mol, None]], frags: Optional[Sequence[Union[str, None]]] = ...) -> np.ndarray:  # type: ignore[override]
        """Score molecules using one or more ROCS queries"""

        ...

    def _build_rocs_command(self, query_file: str, input_file: str, output_file: str) -> List[str]:
        """Build ROCS command matching cli_base_rocs.py exactly"""

        ...

    def _score(self, conf_file: Any) -> Dict[str, Any]:
        """
        Score molecules with ROCS

        Returns:
            dict: Dictionary with reference group names as keys and scores as values.
        """

        ...

    def _score_multi_query(self, conf_file: Any, query_files: Any) -> Dict[str, Any]:
        """Score against multiple reference files in a group, return best scores as dict"""

        ...

    def _score_single_query(self, conf_file: Any, query_file: str) -> Dict[str, Any]:
        """Score molecules against a single reference file and return as dict"""

        ...

    def _execute_rocs(self, query_file: str, input_file: str, tmpdir: str) -> str:
        """Execute ROCS CLI"""

        ...

    def _parse_results(self, output_file: str) -> Dict[str, float]:
        """Parse ROCS output and return scores as a dictionary"""

        ...

    def getKey(self) -> List[str]:
        """Return scorer identifier (suffixed by non-default score_variant so keys don't collide)."""

        ...

