"""Typed exceptions for the conformer study suite.

Every failure mode that a caller may reasonably want to handle explicitly gets
its own type.  The study runners catch :class:`ConformerError` and record the
message in the run manifest instead of aborting the whole grid.
"""

from __future__ import annotations


class ConformerError(Exception):
    """Base class for every recoverable error raised by this package."""


class PreparationError(ConformerError):
    """The input molecule could not be standardized into a 3D-ready state."""


class EmbeddingError(ConformerError):
    """ETKDGv3 failed to produce at least one finite 3D conformer."""


class MinimizationError(ConformerError):
    """A force field could not be parameterized or converged."""


class SelectionError(ConformerError):
    """A reference/candidate selection specification is inconsistent."""


class CacheError(ConformerError):
    """A cache entry exists but its payload is corrupt or incompatible."""


class StudyError(ConformerError):
    """A study matrix is malformed or a required input is missing."""


__all__ = [
    "CacheError",
    "ConformerError",
    "EmbeddingError",
    "MinimizationError",
    "PreparationError",
    "SelectionError",
    "StudyError",
]
