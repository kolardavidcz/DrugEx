"""Study matrices: optimization methods, clustering, color/shape, preparation."""

from __future__ import annotations

from _david.code.conformer.studies.base import (
    CellExecutor,
    CellRequest,
    StudyResult,
    load_study_frame,
    run_study,
    study_requests,
)
from _david.code.conformer.studies.clustering import (
    SELECTIONS,
    SELECTION_LABELS,
    cluster_summary,
)
from _david.code.conformer.studies.clustering import build_study as build_clustering_study
from _david.code.conformer.studies.color_shape import (
    DEFAULT_CANDIDATE,
    DEFAULT_REFERENCE,
    FALSE_POSITIVE_THRESHOLDS,
    BankKeys,
    load_pairwise,
    run_tversky_grid,
    selectivity_summary,
)
from _david.code.conformer.studies.measure import MeasureExecutor, STUDY_SCALE
from _david.code.conformer.studies.optimization_methods import (
    METHOD_LABELS,
    METHODS,
    SELECTION_CUTOFFS,
    SELECTION_MODES,
    STUDY_SCORE,
    MethodPair,
    build_arms,
    deployable_pairs,
    pareto_front,
    stratified_summary,
    symmetric_pairs,
)
from _david.code.conformer.studies.optimization_methods import (
    build_study as build_method_study,
)
from _david.code.conformer.studies.preparation import (
    AUDIT_EMBED,
    AUDIT_METHODS,
    PreparationExecutor,
    audit_arm,
    audit_molecules,
    survival_summary,
)

__all__ = [
    "AUDIT_EMBED",
    "AUDIT_METHODS",
    "BankKeys",
    "CellExecutor",
    "CellRequest",
    "DEFAULT_CANDIDATE",
    "DEFAULT_REFERENCE",
    "FALSE_POSITIVE_THRESHOLDS",
    "METHOD_LABELS",
    "METHODS",
    "MeasureExecutor",
    "MethodPair",
    "PreparationExecutor",
    "SELECTIONS",
    "SELECTION_CUTOFFS",
    "SELECTION_LABELS",
    "SELECTION_MODES",
    "STUDY_SCALE",
    "STUDY_SCORE",
    "StudyResult",
    "audit_arm",
    "audit_molecules",
    "build_arms",
    "build_clustering_study",
    "build_method_study",
    "cluster_summary",
    "deployable_pairs",
    "load_pairwise",
    "load_study_frame",
    "pareto_front",
    "run_study",
    "run_tversky_grid",
    "selectivity_summary",
    "stratified_summary",
    "study_requests",
    "survival_summary",
    "symmetric_pairs",
]
