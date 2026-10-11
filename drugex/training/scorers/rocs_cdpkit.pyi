"""
CDPKit-based ROCS scorer implementation.
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
CDPL_AVAILABLE: bool = ...

import os
import tempfile
from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing import TimeoutError as MPTimeoutError
from dataclasses import dataclass
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from drugex.training.scorers.interfaces import ConformerGenerator
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.oe_color_smarts import build_cdpkit_oe_pharm_generator
from drugex.training.scorers.parallel import pool_context
from drugex.training.scorers.parallel import capped_n_jobs
from drugex.training.scorers.parallel import score_timeout
from drugex.training.scorers.parallel import mol_timeout
from drugex.training.scorers.parallel import molecule_time_limit
from drugex.training.scorers.parallel import ScoreTimeout

MAX_OPTIMIZATION_ITERATIONS: Any = ...
OPTIMIZATION_STOP_GRADIENT: Any = ...
_DEFAULT_CDPKIT_GROUP_NAME: Any = ...
_VALID_OPT_MODES: Any = ...
_VALID_SCORE_VARIANTS: Any = ...
_VALID_COLOR_FFS: Any = ...
_OE_PHARM_GEN_CACHE: Any = ...

def _get_oe_pharm_generator() -> Any:
    """Return this process's OE ImplicitMillsDean pharmacophore generator (built once)."""

    ...


@dataclass
class CDPKitWorkerContext:
    """
    Immutable context for CDPKit scoring workers.

    This dataclass encapsulates all state needed by worker processes,
    replacing module-level global variables. It is sent once per worker
    via the Pool initializer, not with each task.

    Attributes:
        reference_shapes: List of CDPKit GaussianShape objects for alignment.
        group_to_indices: Mapping from group index to reference shape indices.
        conf_file: Path to the conformer SDF file to score.
        opt_mode: Optimization/score mode ("shape" | "combo" | "color").
        score_variant: Normalization ("tanimoto" | "tversky_ref" | "tversky_fit").
        color_ff: Color pharmacophore typing ("default" | "oe_mills_dean").
    """

    reference_shapes: List
    group_to_indices: List[List[int]]
    conf_file: str
    opt_mode: str = ...
    score_variant: str = ...
    color_ff: str = ...

class CDPKitScoringWorker:
    """
    Callable worker for scoring molecules in parallel.

    This class encapsulates the worker logic and holds the context
    that would otherwise be stored in global variables. When used
    with multiprocessing.Pool, the initializer sets up the context
    once per worker process.

    Usage with Pool:
        worker = CDPKitScoringWorker()
        with Pool(n_workers, initializer=worker.initialize,
                  initargs=(context,)) as pool:
            results = pool.map(worker, mol_ids)
    """

    _context: ClassVar[Optional[CDPKitWorkerContext]] = ...
    @staticmethod
    def initialize(context: CDPKitWorkerContext) -> None:
        """
        Initialize worker with shared context.

        Called once per worker process by Pool's initializer.
        Stores context at class level within the worker process.

        Args:
            context: The CDPKitWorkerContext containing reference shapes and config.
        """

        ...

    def __call__(self, mol_id: int) -> Tuple[int, List[float]]:
        """
        Score a single molecule.

        This method is called by pool.map() for each molecule ID.
        Accesses the context set up by initialize().

        Args:
            mol_id: Index of the molecule to score.

        Returns:
            Tuple of (mol_id, list of scores per group).
        """

        ...


def _generate_shape_helper(cdpkit_mol: Any, color_ff: str = ...) -> Any:
    """
    Generate Gaussian shape(s) for a molecule.

    Returns a list of shapes to ensure all conformers are considered.
    If no shapes can be generated, returns an empty list. When ``color_ff="oe_mills_dean"``
    the colored shape is typed with the OpenEye ImplicitMillsDean SMARTS (Lever 1) instead of
    CDPKit's default pharmacophore perception.
    """

    ...


def _select_cdpkit_score(alignment_result: Any, opt_mode: Any, score_variant: Any) -> Any:
    """
    Pick the CDPKit score function for (opt_mode x score_variant), from the module-level
    ``_CDPKIT_SCORE_FUNCS`` table (built once at import, not per call).

    score_variant:
      "tanimoto"    -> symmetric Tanimoto (default; the OE-parity metric).
      "tversky_ref" -> reference-weighted Tversky ("is the REFERENCE pharmacophore covered
                       by the candidate?"); forgives the candidate being larger than the
                       small CCR2 references (PheSA, Wahl 2024, 10.1021/acs.jcim.4c00516).
      "tversky_fit" -> aligned(fit)-weighted Tversky (candidate-centric).
    All are CDPKit-native and exact. (RDKit reaches the same Tversky on its native basis by
    recovering the overlap from its reported Tanimoto + sov/sof; OpenEye reads native Tversky
    report columns.) Values are validated in CDPKitROCSScorer.__init__; the ``.get`` fallbacks
    here are a defensive backstop for direct callers (tests).
    """

    ...


def _align_and_score_helper(query_shape: Any, ref_shape: Any, opt_mode: Any = ..., score_variant: Any = ...) -> Any:
    """
    Align two shapes and return the score for the requested optimization mode + variant.

    opt_mode:
      "shape" -> ShapeTanimoto (shape-center start seeding).
      "combo" -> TanimotoCombo (shape-center start seeding; production default).
      "color" -> ColorTanimoto with COLOR-feature-center start seeding. CDPKit's BFGS
                 optimizer has no color gradient (``calcColorOverlapGradient`` does not
                 exist), so color is approached by seeding alignment starts at
                 pharmacophore-feature centers and selecting the best color overlap --
                 the closest open-source approximation to OpenEye ROCS ``-optchem``.
    score_variant: "tanimoto" (default) | "tversky_ref" | "tversky_fit"; see
      ``_select_cdpkit_score``. The pose is optimized identically (Tversky changes only the
      reported normalization), so the best score is selected over the same alignment results.
    """

    ...


class CDPKitROCSScorer(Scorer):
    """
    ROCS-like scoring using CDPKit Gaussian shapes.

    Features:
    - Multi-reference support with automatic Gaussian shape generation.
    - Best-score selection across conformers and reference shapes.
    - Optional progress reporting and multiprocessing.
    - Dict-based reference grouping for multi-target optimization.
    - SMILES deduplication for efficient batch processing.
    - Memory-efficient worker design for scalability.

    Attributes:
        conformer_generator: 3D conformer generator used for query molecules.
        group_definitions: List of (name, paths) tuples defining reference groups.
        group_names: List of reference group names.
        shape_generator: CDPKit GaussianShapeGenerator instance.
        start_generator: CDPKit PrincipalAxesAlignmentStartGenerator instance.
        reference_mols: Loaded CDPKit molecules containing reference conformers.
        reference_shapes: Pre-computed Gaussian shapes for all references.
        group_to_indices: List mapping group indices to reference indices.
        show_progress: Whether to print progress and warnings.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _is_supermol: True when initialized with a single reference file.
    """

    def __init__(self, conformer_generator: ConformerGenerator, references: Union[str, List[str], Dict[str, List[str]]], show_progress: bool = ..., n_jobs: int = ..., opt_mode: str = ..., score_variant: str = ..., color_ff: str = ...) -> None:
        """
        Build a CDPKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used to produce query conformers.
            references: Path(s) to SDF references or a dict mapping group names
                to lists of reference paths.
            show_progress: Enables stdout progress indicators when True.
            n_jobs: Number of worker processes (-1 uses all available CPUs).
            color_ff: Color pharmacophore typing: "default" = CDPKit
                DefaultPharmacophoreGenerator (legacy), "oe_mills_dean" = OpenEye
                ImplicitMillsDean SMARTS port (Lever 1). OPT-IN: the default leaves color
                scoring unchanged. Recalibrate thresholds before making it the default
                (the OE-ground-truth Spearman GATE is a separate compute step).

        Raises:
            ImportError: If CDPKit bindings are not available.
            TypeError: If reference input types are unsupported.
            FileNotFoundError: If a reference path does not exist.
            ValueError: If reference shapes cannot be generated.
        """

        ...

    def _prepare_reference_groups(self, references: Union[str, List[str], Dict[str, List[str]]]) -> List[Tuple[str, List[str]]]:
        ...

    def _normalize_reference_list(self, refs: Union[str, List[str]]) -> List[str]:
        ...

    def _load_reference_groups(self) -> None:
        ...

    def _load_reference_molecules(self, filepath: str) -> List[Any]:
        ...

    def _generate_gaussian_shape(self, mol: Chem.Mol) -> Optional[object]:
        ...

    def getKey(self) -> List[str]:
        """Returns unique list of string objective keys identifying reference groups and scoring modes."""

        ...

    def create_progress_bar(self, total: Any, desc: Any) -> Any:
        ...

    def getScores(self, mols: Sequence[Union[str, Chem.Mol, None]], frags: Optional[Sequence[Union[str, None]]] = ...) -> np.ndarray:  # type: ignore[override]
        """
        Score molecules using CDPKit Gaussian shape alignment.

        Args:
            mols: List of molecules (RDKit Mol objects or SMILES strings).
            frags: Unused, kept for interface compatibility.

        Returns:
            Array of shape (len(mols), num_groups) with TanimotoCombo scores.
        """

        ...

    @staticmethod
    def _deduplicate_smiles(smiles_list: List[Optional[str]]) -> Tuple[List[str], Dict[int, List[int]]]:
        """Return unique SMILES plus mapping back to originals."""

        ...

