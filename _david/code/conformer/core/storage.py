"""Crash-safe serialization helpers.

Every writer in this module writes to a temporary sibling and then atomically
replaces the destination, so an interrupted study never leaves a half-written
CSV or SDF that a later run would trust.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.conformer.core import provenance


def _atomic_replace(target: Path, payload: bytes) -> Path:
    """Write ``payload`` to ``target`` through a temporary sibling file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with open(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def write_text(path: Path | str, text: str) -> Path:
    """Atomically write UTF-8 text."""
    return _atomic_replace(Path(path), text.encode("utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Atomically write an indented, deterministic JSON document."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    return write_text(path, text)


def read_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON document and require a top-level object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_csv(path: Path | str, frame: pd.DataFrame) -> Path:
    """Atomically write a CSV table without an index column."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return write_text(path, buffer.getvalue())


def read_csv(path: Path | str) -> pd.DataFrame:
    """Read a CSV table."""
    return pd.read_csv(path)


def write_sdf(path: Path | str, records: Iterable[Chem.Mol]) -> Path:
    """Atomically write an SDF, preserving float64 coordinates and all props."""
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    try:
        for record in records:
            if record is None:
                continue
            writer.write(record)
    finally:
        writer.close()
    return write_text(path, buffer.getvalue())


def read_sdf(path: Path | str) -> list[Chem.Mol]:
    """Read an SDF with explicit hydrogens, dropping unparseable records.

    A record is only dropped when RDKit itself returns ``None``; a valid record
    with zero conformers is returned so callers can decide what to do with it.
    """
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return []
    return [
        molecule
        for molecule in Chem.SDMolSupplier(str(target), removeHs=False)
        if molecule is not None
    ]


def write_npz(path: Path | str, payload: Mapping[str, np.ndarray]) -> Path:
    """Atomically write a compressed NumPy archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npz", dir=target.parent
    )
    try:
        with open(descriptor, "wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value) for key, value in payload.items()})
        Path(temporary_name).replace(target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Read a NumPy archive into a plain dictionary."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_table_bundle(directory: Path | str, frame: pd.DataFrame, name: str = "data") -> dict[str, Path]:
    """Write a CSV and a gzipped CSV copy; return both paths by role."""
    import gzip

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    payload = buffer.getvalue()
    csv_path = write_text(root / f"{name}.csv", payload)
    gzip_path = _atomic_replace(root / f"{name}.csv.gz", gzip.compress(payload.encode("utf-8")))
    return {"csv": csv_path, "csv_gz": gzip_path}


def file_record(path: Path | str) -> dict[str, Any]:
    """Return a size/digest record suitable for a manifest."""
    target = Path(path)
    return {
        "path": target.as_posix(),
        "bytes": target.stat().st_size if target.exists() else None,
        "sha256": provenance.sha256_file(target),
    }


def directory_records(paths: Sequence[Path | str]) -> dict[str, dict[str, Any]]:
    """Return a filename-keyed mapping of :func:`file_record`."""
    return {Path(path).name: file_record(path) for path in paths}


__all__ = [
    "directory_records",
    "file_record",
    "read_csv",
    "read_json",
    "read_npz",
    "read_sdf",
    "write_csv",
    "write_json",
    "write_npz",
    "write_sdf",
    "write_table_bundle",
    "write_text",
]
