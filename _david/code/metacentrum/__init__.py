"""MetaCentrum PBS Pro cluster grid utilities."""

from typing import Any

__all__ = [
    "GridJobSpec",
    "MetaCentrumGridManager",
    "generate_4hypothesis_grid",
    "generate_8arm_grid",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from _david.code.metacentrum import grid

        return getattr(grid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
