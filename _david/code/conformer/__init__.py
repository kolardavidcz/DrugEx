"""Conformer ensembles, 3D shape/color scoring, and reproducible studies.

The package is organised around four scientific components:

``prepare``
    Standardize an input structure exactly the way the DrugEx reward pipeline
    does, enumerate stereoisomers, and register the result as canonical data.
``ensemble``
    Embed with ETKDGv3, relax with an explicit force field, then reduce the
    pool to a deployable reference bank (energy / Butina / balanced / diverse).
``scoring``
    TanimotoCombo (physical ``[0, 2]``) and asymmetric Tversky shape/color.
``studies``
    Declarative study matrices that turn the three layers above into tables.
``analysis``
    Publication figures built from those tables, one module per component.

Every expensive step is a pure function of a frozen specification, keyed by
:data:`~._david.code.conformer.core.contracts.ALGORITHM_VERSION` in the
content-addressed cache under ``_david/data/conformers/cache``.
"""

from __future__ import annotations

from _david.code.conformer.core import (
    ALGORITHM_VERSION,
    ArmSpec,
    ArtifactCache,
    CacheError,
    CandidateSpec,
    ConformerError,
    EmbedSpec,
    EmbeddingError,
    MinimizationError,
    MinimizeSpec,
    MoleculeSpec,
    PreparationError,
    ReferenceSpec,
    ScoreScale,
    ScoreSpec,
    ScoreVariant,
    SeedPlan,
    SelectSpec,
    SelectionError,
    SelectionMode,
    StudyError,
    StudySpec,
    arm,
    default_candidate_spec,
    default_reference_spec,
    fingerprint,
    jsonable,
)
from _david.code.conformer.ensemble import (
    CandidateEnsemble,
    EmbedResult,
    MinimizeResult,
    ReferenceBank,
    SelectionResult,
    build_candidate,
    build_reference,
    cluster_statistics,
    embed_ensemble,
    minimize_ensemble,
    pool_fingerprint,
    reference_from_records,
    select_ensemble,
    selection_histogram,
    selection_overlap,
)
from _david.code.conformer.prepare import (
    PreparedMolecule,
    molecule_properties,
    prepare_many,
    prepare_molecule,
)
from _david.code.conformer.scoring import (
    EnsembleScore,
    PairwiseTversky,
    align_pair_raw,
    grid_scores,
    pairwise_tversky,
    score_ensemble,
    score_self_recovery,
    score_tversky,
    self_volumes,
)

__all__ = [
    "ALGORITHM_VERSION",
    "ArmSpec",
    "ArtifactCache",
    "CacheError",
    "CandidateEnsemble",
    "CandidateSpec",
    "ConformerError",
    "EmbedResult",
    "EmbedSpec",
    "EmbeddingError",
    "EnsembleScore",
    "MinimizeResult",
    "MinimizeSpec",
    "MinimizationError",
    "MoleculeSpec",
    "PairwiseTversky",
    "PreparationError",
    "PreparedMolecule",
    "ReferenceBank",
    "ReferenceSpec",
    "ScoreScale",
    "ScoreSpec",
    "ScoreVariant",
    "SeedPlan",
    "SelectSpec",
    "SelectionError",
    "SelectionMode",
    "SelectionResult",
    "StudyError",
    "StudySpec",
    "align_pair_raw",
    "arm",
    "build_candidate",
    "build_reference",
    "cluster_statistics",
    "default_candidate_spec",
    "default_reference_spec",
    "embed_ensemble",
    "fingerprint",
    "grid_scores",
    "jsonable",
    "minimize_ensemble",
    "molecule_properties",
    "pairwise_tversky",
    "pool_fingerprint",
    "prepare_many",
    "prepare_molecule",
    "reference_from_records",
    "score_ensemble",
    "score_self_recovery",
    "score_tversky",
    "select_ensemble",
    "selection_histogram",
    "selection_overlap",
    "self_volumes",
]
