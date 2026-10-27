"""
RDKit-based ROCS scorer.
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
from drugex.training.generators.interfaces import Generator
OE_AVAILABLE: bool = ...
CDPL_AVAILABLE: bool = ...

import os
import tempfile
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from rdkit import RDConfig
from rdkit.Chem import rdShapeAlign
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdMolTransforms
from drugex.training.scorers.interfaces import ConformerGenerator
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.oe_color_smarts import OE_MILLS_DEAN_PATTERNS
from drugex.training.scorers.parallel import pool_context
from drugex.training.scorers.parallel import capped_n_jobs
from drugex.training.scorers.parallel import mol_timeout
from drugex.training.scorers.parallel import molecule_time_limit
from drugex.training.scorers.parallel import ScoreTimeout

_RDKIT_WORKER_SETTINGS: Dict[str, object]
_DEFAULT_RDKIT_GROUP_NAME: Any = ...
_VALID_SCORE_TYPES: Any = ...
_VALID_SCORE_VARIANTS: Any = ...
_VALID_COLOR_FFS: Any = ...
_TVERSKY_AB: Any = ...
_FAMILY_TO_COLOR: Any = ...
_AXIS_SIGN_FLIPS: List[Tuple[int, int, int]]

def _color_lines_from_fdef(mol: Chem.Mol) -> List[str]:
    """PUBCHEM_PHARMACOPHORE_FEATURES body lines from RDKit BaseFeatures.fdef typing."""

    ...


def _color_lines_from_oe(mol: Chem.Mol) -> List[str]:
    """
    PUBCHEM_PHARMACOPHORE_FEATURES body lines from the OpenEye ImplicitMillsDean SMARTS.

    Each match of an OE color SMARTS becomes one feature line: ``<n> <atom ids...> <color>``
    with 1-based SDF atom ids (the same scheme the fdef path and rdShapeAlign's 6-type reader
    use). The OE port (oe_color_smarts.py) supplies precompiled patterns per color name.
    """

    ...


def _annotate_color_features(mol: Chem.Mol, color_ff: str = ...) -> Chem.Mol:
    """
    Write PUBCHEM_PHARMACOPHORE_FEATURES so rdShapeAlign's color uses the full 6-type
    scheme (donor/acceptor/cation/anion/rings/hydrophobe), matching OpenEye ImplicitMillsDean.

    ``color_ff`` selects the pharmacophore typing:
      "default" / "fdef" -> RDKit's BaseFeatures.fdef (the original behavior).
      "oe_mills_dean"    -> OpenEye ImplicitMillsDean SMARTS (Lever 1, oe_color_smarts.py),
                            which (unlike fdef) types saturated rings as ``rings`` and alkyl
                            chains as ``hydrophobe`` exactly as OpenEye does.

    Mutates ``mol`` in place and returns it. No-op when the (fdef) factory is unavailable or the
    molecule yields no mappable features (an empty feature block crashes rdShapeAlign).

    PRECONDITIONS before enabling ``inject_color`` in production (code-review #2/#3, Task #15):
    1. Scheme consistency — if ONE molecule is annotated (6-type scheme) and the other is not
       (e.g. a featureless molecule, or a reference SDF that already ships a foreign
       PUBCHEM_PHARMACOPHORE_FEATURES), rdShapeAlign perceives the un-annotated one with its
       5-pattern fallback whose integer type IDs differ -> color is computed on mismatched
       schemes. The shipped CCR2 references have no pre-existing property and all map features,
       so this is latent for them; annotate references ONCE in __init__ (overwriting any
       foreign property) before enabling injection on arbitrary reference sets.
    2. Mutation — this writes onto the passed object; in the sequential path that is the
       scorer's reference Mol (private when loaded from an SDF path, but caller-shared if a
       Chem.Mol was passed directly). Annotating refs once in __init__ also removes the
       per-call shared mutation.
    """

    ...


def _tversky_from_tanimoto(tanimoto: float, self_a: float, self_b: float, alpha: float, beta: float) -> float:
    """
    Tversky index on rdShapeAlign's NATIVE first-order Gaussian basis.

    rdShapeAlign reports only Tanimoto ``T = O_ab/(sov_a + sov_b - O_ab)`` but exposes the
    native self-overlaps (``ShapeInput.sov``/``.sof``). We recover the native cross-overlap
    ``O_ab = T*(self_a + self_b)/(1 + T)`` and form the Tversky index
    ``O_ab / (O_ab + alpha*(self_a - O_ab) + beta*(self_b - O_ab))``. With ``alpha=beta=1`` and
    a per-component Tanimoto ``T in [0,1]`` this returns ``T`` exactly -- the proof that the
    recovery is on the native basis. (The round-trip identity holds only for ``T<=1``; do NOT
    feed a TanimotoCombo in [0,2] here -- call per component. The final clamp would cap such a
    value at 1.0.) Result is clamped to [0, 1]: the first-order Gaussian overlap is an L2 inner
    product, so O_ab can exceed a self-overlap under extreme size mismatch -- for reference-
    weighted Tversky that legitimately means "reference fully covered" (saturates at 1.0).
    ``alpha`` weights ``self_a`` (pass the REFERENCE self-overlap as ``self_a``).
    """

    ...


def _self_overlaps(mol: Chem.Mol, conf_id: int, opts: Any) -> Tuple[float, float]:
    """Native (sov, sof) self-overlaps for a conformer (pose-independent)."""

    ...


def _canonicalize_conformer_inplace(mol: Chem.Mol, conf_id: int) -> None:
    """
    Put one conformer of ``mol`` into its inertial frame in place (issue #8513 guard).

    No-op (swallowed) for degenerate geometries (e.g. <3 atoms, collinear) where the inertia
    tensor is singular; such molecules are pose-trivial anyway. ``conf_id`` of -1 selects the
    default conformer, matching rdShapeAlign's convention.
    """

    ...


def _flip_conformer(mol: Chem.Mol, conf_id: int, signs: Tuple[int, int, int]) -> Chem.Mol:
    """
    Return a copy of ``mol`` with conformer ``conf_id`` reflected by ``signs`` about the axes.

    ``signs`` must be a proper rotation (product of the three == +1) so chirality is preserved.
    """

    ...


def _score_single_reference(query_mol: Chem.Mol, ref_mol: Chem.Mol, score_type: str, use_colors: bool, inject_color: bool = ..., score_variant: str = ..., color_ff: str = ...) -> float:
    """
    Compute best alignment score between a query molecule and one reference.

    Uses rdShapeAlign.AlignMol which performs Gaussian shape overlay (same
    algorithm family as OpenEye ROCS and CDPKit GaussianShapeAlignment).

    ``color_ff`` selects the pharmacophore typing for the color channel (see
    ``_annotate_color_features``): "default"/"fdef" uses RDKit BaseFeatures.fdef,
    "oe_mills_dean" uses the OpenEye ImplicitMillsDean SMARTS port (Lever 1).
    Annotation is applied when ``inject_color`` is True OR ``color_ff`` is non-default
    (selecting an explicit FF implies injecting it); ``color_ff="default"`` with
    ``inject_color=False`` is the legacy no-op path.

    The alignment OPTIMIZATION objective is driven by ``score_type`` through
    AlignMol's ``opt_param`` (1.0 = shape-only, 0.5 = equal-weight shape+color
    ~ OpenEye ROCS ``-optchem`` TanimotoCombo, 0.0 = color-only). On flexible
    molecules this materially changes both the pose and the color score (an
    8-rotatable-bond probe vs a CCR2 reference gains ~0.10 ColorTanimoto under
    color optimization); it has little effect only on rigid references where
    the shape- and color-optimal poses coincide. (The earlier claim that
    opt_param has no effect held only at the rigid-reference / 200-conformer
    regime and is false for the flexible generated library.)
    """

    ...


def _rdkit_worker_init(reference_mols: List[Chem.Mol], group_to_indices: List[List[int]], score_type: str, use_colors: bool, inject_color: bool = ..., score_variant: str = ..., color_ff: str = ...) -> None:
    """Initializer to share immutable worker state."""

    ...


def _score_molecule_rdkit_worker(args: Tuple[int, List[Chem.Mol]]) -> Tuple[int, List[float]]:
    """Score a molecule in a worker process."""

    ...


class RDKitROCSScorer(Scorer):
    """
    ROCS-style scoring using RDKit shape alignment.

    Features:
    - Accepts references as files or RDKit molecules with conformers.
    - Supports shape-only, color-only, and combo scoring modes.
    - Optional multiprocessing and verbose progress reporting.
    - Dict-based reference grouping for multi-target optimization.
    - SMILES deduplication for efficient batch processing.
    - Auto-conformer generation for 2D reference molecules.

    Attributes:
        conformer_generator: Generator used to build query conformers.
        reference_mols: Normalized list of reference molecules.
        group_to_indices: List mapping group indices to reference indices.
        group_names: List of reference group names.
        score_type: Requested score mode (`shape`, `color`, or `TanimotoCombo`).
        use_colors: Whether to include pharmacophore colors in alignment.
        show_progress: Enables logging for long runs.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _single_reference: True when operating in single-reference mode.
    """

    def __init__(self, conformer_generator: ConformerGenerator, references: Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]], score_type: str = ..., use_colors: bool = ..., show_progress: bool = ..., n_jobs: int = ..., inject_color: bool = ..., score_variant: str = ..., color_ff: str = ...) -> None:
        """
        Initialize the RDKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used for query molecules.
            references: Reference source as SDF path(s), RDKit molecule(s),
                or a dict mapping group names to lists of references.
            score_type: Score variant (`TanimotoCombo`, `shape`, `color`).
            use_colors: Whether to include pharmacophore colors in alignments.
            show_progress: Enables stdout progress updates when True.
            n_jobs: Number of worker processes (-1 uses all available CPUs).
            inject_color: When True, annotate query + reference molecules with the
                6-type PUBCHEM_PHARMACOPHORE_FEATURES scheme (incl. hydrophobe) so
                rdShapeAlign's color matches OpenEye ImplicitMillsDean rather than its
                5-pattern fallback. Off by default (changes color scores; recalibrate
                thresholds before enabling in production).
            color_ff: Pharmacophore typing for the injected color features:
                "default"/"fdef" = RDKit BaseFeatures.fdef (legacy), "oe_mills_dean" =
                OpenEye ImplicitMillsDean SMARTS port (Lever 1). Selecting "oe_mills_dean"
                implies color injection with OE typing. OPT-IN: the default leaves color
                scoring bit-for-bit unchanged. Recalibrate thresholds before making it
                the default (the OE-ground-truth Spearman GATE is a separate compute step).

        Raises:
            TypeError: If reference inputs use unsupported types.
            FileNotFoundError: If a reference path is missing.
            ValueError: If reference molecules lack conformers or cannot load.
        """

        ...

    def _prepare_reference_groups(self, references: Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]]) -> List[Tuple[str, List[Chem.Mol]]]:
        ...

    def _normalize_reference_collection(self, refs: Union[str, List[str], Chem.Mol, List[Chem.Mol]]) -> List[Chem.Mol]:
        ...

    def _flatten_groups(self, groups: List[Tuple[str, List[Chem.Mol]]]) -> Tuple[List[Chem.Mol], List[List[int]]]:
        ...

    def _load_molecules_from_file(self, path: str) -> List[Chem.Mol]:
        ...

    def _ensure_reference_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        ...

    def _validate_references(self) -> Any:
        ...

    def _key_suffix(self) -> str:
        """Distinguish scorers by non-default flags so keys don't collide (defaults -> '')."""

        ...

    def getKey(self) -> List[str]:
        """Returns unique list of string objective keys identifying reference groups and scoring modes."""

        ...

    @staticmethod
    def _deduplicate_smiles(smiles_list: List[Union[str, None]]) -> Tuple[List[str], Dict[int, List[int]]]:
        """Group identical SMILES to avoid redundant conformer generation."""

        ...

    def _convert_to_smiles(self, mols: List[Chem.Mol]) -> List[Union[str, None]]:
        ...

    def _score_sequential(self, unique_count: int, conformers_by_mol: Dict[int, List[Chem.Mol]], num_groups: int) -> np.ndarray:
        """
        Score molecules sequentially without multiprocessing.

        Args:
            unique_count: Number of unique molecules to score.
            conformers_by_mol: Dictionary mapping molecule IDs to their conformer lists.
            num_groups: Number of reference groups.

        Returns:
            Array of shape (unique_count, num_groups) containing scores.
        """

        ...

    def getScores(self, mols: List[Chem.Mol], frags: Optional[List[Chem.Mol]] = ...) -> np.ndarray:  # type: ignore[override]
        """Generates 3D conformations via conformer_generator, superimposes against reference structures via rdShapeAlign, and returns max similarity scores."""

        ...

