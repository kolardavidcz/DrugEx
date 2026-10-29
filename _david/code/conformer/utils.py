"""Shared I/O, cryptographic provenance, SDF serialization, and fault-tolerant worker utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from rdkit import Chem

from _david.code.config import PROJECT_ROOT


def digest(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON file via a temporary sibling file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def save_sdf(path: Path | str, records: Iterable[Chem.Mol]) -> Path:
    """Write a sequence of RDKit molecule conformer records to an SDF file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(target)) as writer:
        for record in records:
            writer.write(record)
    return target


def load_sdf(path: Path | str) -> list[Chem.Mol]:
    """Load RDKit molecules with explicit hydrogens from an SDF file, returning [] if empty."""
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return []
    return [
        mol
        for mol in Chem.SDMolSupplier(str(target), removeHs=False)
        if mol is not None
    ]


def source_hashes(relative_paths: Iterable[Path | str]) -> dict[str, str]:
    """Compute SHA-256 digests for a collection of repository-relative source files."""
    return {
        str(rel_path): digest(PROJECT_ROOT / rel_path)
        for rel_path in relative_paths
    }


def collect_jobs(
    executor: Any,
    jobs: Sequence[Mapping[str, Any]],
    output: Path,
    manifest: MutableMapping[str, Any],
    worker_fn: Callable[[Mapping[str, Any], str], Any] | None = None,
) -> dict[str, str]:
    """Submit per-molecule jobs to ``executor`` and record completions and errors atomically.

    Writes per-input tracebacks to ``output.parent / 'logs' / 'input_<id>.error.log'``
    and updates ``output.parent / 'progress.json'`` without aborting remaining inputs.
    """
    fn = worker_fn or (lambda case, out: case["molecule_id"])
    futures = {
        executor.submit(fn, case, str(output)): case for case in jobs
    }
    errors: dict[str, str] = dict(manifest.get("errors", {}))
    logs_dir = output.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for future in as_completed(futures):
        case = futures[future]
        molecule_id = int(case["molecule_id"])
        try:
            future.result()
        except Exception as exc:
            errors[str(molecule_id)] = f"{type(exc).__name__}: {exc}"
            manifest["errors"] = errors
            manifest["state"] = "measuring_with_errors"
            (logs_dir / f"input_{molecule_id}.error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        else:
            errors.pop(str(molecule_id), None)
            if molecule_id not in manifest["completed_molecules"]:
                manifest["completed_molecules"].append(molecule_id)

        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "manifest.json", manifest)
        done = len(manifest["completed_molecules"])
        total = int(manifest.get("total_molecules", len(jobs)))
        atomic_json(
            output.parent / "progress.json",
            dict(
                state=manifest["state"],
                completed=done,
                total=total,
                errors=errors,
                updated_utc=manifest["updated_utc"],
            ),
        )
        print(
            f"{done}/{total} complete; {len(errors)} errors",
            flush=True,
        )
    return errors
