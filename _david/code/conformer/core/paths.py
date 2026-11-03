"""Canonical filesystem locations for the conformer suite.

Layout
------
``_david/data/conformers/``
    Long-lived, reusable scientific data.  Everything here is content-addressed
    or explicitly registered, so it may safely be reused by later studies.

``_david/output/conformer/``
    Human-facing deliverables, one folder per component.  Regenerable from the
    data cache plus a study specification.
"""

from __future__ import annotations

from pathlib import Path

from _david.code.config import PROJECT_ROOT

DAVID_ROOT: Path = PROJECT_ROOT / "_david"

#: Long-lived canonical data (prepared molecules, ensemble cache, manifests).
DATA_ROOT: Path = DAVID_ROOT / "data" / "conformers"

#: Content-addressed cache for embedding, minimization, selection, and scoring.
CACHE_ROOT: Path = DATA_ROOT / "cache"

#: Registered, verified input molecules and their prepared forms.
CANONICAL_ROOT: Path = DATA_ROOT / "canonical"

#: Human-facing deliverables, one directory per component.
OUTPUT_ROOT: Path = DAVID_ROOT / "output" / "conformer"

COMPONENT_DIRS: dict[str, str] = {
    "optimization_methods": "01_optimization_methods",
    "color_shape": "02_color_shape_scoring",
    "clustering": "03_clustering_benchmark",
    "preparation": "04_molecule_preparation",
}


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_root() -> Path:
    """Return (and create) the content-addressed cache root."""
    return ensure_dir(CACHE_ROOT)


def canonical_root() -> Path:
    """Return (and create) the registered-molecule root."""
    return ensure_dir(CANONICAL_ROOT)


def component_dir(component: str) -> Path:
    """Return (and create) the output directory for one component."""
    try:
        folder = COMPONENT_DIRS[component]
    except KeyError as exc:
        raise ValueError(
            f"Unknown component {component!r}; expected one of {sorted(COMPONENT_DIRS)}"
        ) from exc
    return ensure_dir(OUTPUT_ROOT / folder)


__all__ = [
    "CACHE_ROOT",
    "CANONICAL_ROOT",
    "COMPONENT_DIRS",
    "DATA_ROOT",
    "DAVID_ROOT",
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "cache_root",
    "canonical_root",
    "component_dir",
    "ensure_dir",
]
