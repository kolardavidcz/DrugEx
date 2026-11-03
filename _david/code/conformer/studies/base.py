"""Declarative study matrices with crash-safe output and provenance.

A *study* is a Cartesian product of molecules x arms x seed repeats.  Each cell
is executed by a picklable callable, which makes the same code path usable
serially and through a process pool without shipping RDKit objects across a
pipe.  Workers rebuild chemistry from strings and share the content-addressed
cache on disk, so a crashed or interrupted run resumes instead of restarting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from _david.code.conformer.core import storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import ArmSpec, MoleculeSpec, StudySpec
from _david.code.conformer.core.errors import StudyError
from _david.code.conformer.core.provenance import build_manifest, utc_now


@dataclass(frozen=True)
class CellRequest:
    """One unit of work: a molecule, an arm, and a repeat index."""

    molecule: MoleculeSpec
    arm: ArmSpec
    repeat: int = 0

    @property
    def label(self) -> str:
        """Stable, human-readable cell identifier."""
        return f"{self.molecule.name}::{self.arm.name}::r{self.repeat}"


class CellExecutor(Protocol):
    """A picklable callable that executes one :class:`CellRequest`."""

    def __call__(self, request: CellRequest) -> Mapping[str, Any]:
        """Return a flat row of measurements for ``request``."""
        ...


@dataclass(frozen=True)
class StudyResult:
    """Everything one study run produced, in memory and on disk."""

    name: str
    frame: pd.DataFrame
    errors: tuple[dict[str, Any], ...]
    output_dir: Path
    manifest: dict[str, Any]
    artifacts: Mapping[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when at least one cell completed successfully."""
        return not self.frame.empty

    def save(self, frame: pd.DataFrame | None = None, *, name: str = "results") -> Path:
        """Write the (possibly extended) frame as CSV and return its path."""
        path = storage.write_csv(self.output_dir / f"{name}.csv", frame if frame is not None else self.frame)
        self.artifacts[f"{name}.csv"] = path
        return path


def study_requests(study: StudySpec) -> list[CellRequest]:
    """Expand a study specification into its ordered cell list."""
    requests: list[CellRequest] = []
    for repeat in study.seeds.repeats:
        for molecule in study.molecules:
            for arm in study.arms:
                requests.append(CellRequest(molecule=molecule, arm=arm, repeat=repeat))
    return requests


def _run_serial(
    requests: Sequence[CellRequest],
    executor: CellExecutor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for request in requests:
        try:
            rows.append({"cell": request.label, **dict(executor(request))})
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
            errors.append(
                {
                    "cell": request.label,
                    "molecule": request.molecule.name,
                    "arm": request.arm.name,
                    "repeat": request.repeat,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows, errors


@dataclass(frozen=True)
class _GuardedCell:
    """Picklable wrapper that converts an executor exception into a payload.

    A local closure cannot be sent to a process pool, so the guard must be a
    module-level object; it holds the executor and nothing else.
    """

    executor: CellExecutor

    def __call__(self, request: CellRequest) -> dict[str, Any]:
        try:
            return {"cell": request.label, "ok": True, **dict(self.executor(request))}
        except Exception as exc:  # noqa: BLE001 - serialized to the parent process
            return {
                "cell": request.label,
                "ok": False,
                "molecule": request.molecule.name,
                "arm": request.arm.name,
                "repeat": request.repeat,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }


def _run_pool(
    requests: Sequence[CellRequest],
    executor: CellExecutor,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from drugex.training.scorers.parallel import pool_context

    context = pool_context()
    with context.Pool(processes=workers) as pool:
        payloads = pool.map(_GuardedCell(executor), requests, chunksize=1)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.pop("ok"):
            rows.append(payload)
        else:
            errors.append(payload)
    return rows, errors


def run_study(
    study: StudySpec,
    executor: CellExecutor,
    *,
    cache: ArtifactCache | None = None,
    workers: int = 1,
    extra_manifest: Mapping[str, Any] | None = None,
) -> StudyResult:
    """Execute every cell of ``study`` and persist the table plus provenance.

    ``workers > 1`` runs cells in a process pool.  The executor must then be a
    module-level object holding only plain data (dataclasses and strings), which
    is why chemistry is always rebuilt from a :class:`MoleculeSpec` in the
    worker rather than pickled.
    """
    if not study.molecules:
        raise StudyError("Study has no molecules")
    if not study.arms:
        raise StudyError("Study has no arms")

    output_dir = Path(study.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = study_requests(study)
    if not requests:
        raise StudyError("Study expanded to zero cells")

    if workers > 1:
        rows, errors = _run_pool(requests, executor, workers)
    else:
        rows, errors = _run_serial(requests, executor)

    frame = pd.DataFrame(rows)
    error_frame = pd.DataFrame(errors)
    artifacts: dict[str, Path] = {}
    if not frame.empty:
        artifacts["results.csv"] = storage.write_csv(output_dir / "results.csv", frame)
    if not error_frame.empty:
        artifacts["errors.csv"] = storage.write_csv(output_dir / "errors.csv", error_frame)

    manifest = build_manifest(
        study=study.name,
        payload={
            "study": study,
            "cells": len(requests),
            "workers": workers,
        },
        inputs={
            "molecules": str(len(study.molecules)),
            "arms": str(len(study.arms)),
            "repeats": str(len(study.seeds.repeats)),
        },
        extra={
            "notes": study.notes,
            "completed_utc": utc_now(),
            "cells": len(requests),
            "workers": workers,
            "completed_cells": int(len(frame)),
            "failed_cells": int(len(error_frame)),
            "cache": cache.summary() if cache is not None else {"enabled": False},
            **(dict(extra_manifest) if extra_manifest else {}),
        },
    )
    artifacts["manifest.json"] = storage.write_json(output_dir / "manifest.json", manifest)
    return StudyResult(
        name=study.name,
        frame=frame,
        errors=tuple(errors),
        output_dir=output_dir,
        manifest=manifest,
        artifacts=artifacts,
    )


def load_study_frame(path: Path | str) -> pd.DataFrame:
    """Load a previously written study table."""
    return storage.read_csv(path)


__all__ = [
    "CellExecutor",
    "CellRequest",
    "StudyResult",
    "load_study_frame",
    "run_study",
    "study_requests",
]
