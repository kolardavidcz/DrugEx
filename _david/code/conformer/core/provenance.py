"""Deterministic provenance helpers for reproducible conformer studies.

The goal is that any figure or table can be traced back to (a) the exact source
tree, (b) the exact package versions, and (c) the exact specification payload
that produced it.  Nothing here is allowed to raise on a best-effort lookup:
missing git metadata or an unavailable package becomes ``None``/``"missing"``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from _david.code.conformer.core.contracts import jsonable

TRACKED_DISTRIBUTIONS = (
    "rdkit",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "torch",
    "drugex",
)


def utc_now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str | None:
    """Return the SHA-256 digest of a file, or ``None`` if it is unreadable."""
    target = Path(path)
    try:
        return sha256_bytes(target.read_bytes())
    except OSError:
        return None


def git_revision(repo_root: Path | str | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` outside a repository."""
    cwd = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def git_dirty(repo_root: Path | str | None = None) -> bool | None:
    """Return whether the working tree has uncommitted changes, or ``None``."""
    cwd = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing the package."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str | None]:
    """Capture the numerical stack versions that can change a score."""
    versions: dict[str, str | None] = {
        name: _distribution_version(name) for name in TRACKED_DISTRIBUTIONS
    }
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    versions["cpu_count"] = str(os.cpu_count() or 0)
    return versions


def build_manifest(
    *,
    study: str,
    payload: Any,
    inputs: dict[str, str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a compact provenance manifest for one study run."""
    manifest: dict[str, Any] = {
        "study": study,
        "generated_utc": utc_now(),
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "versions": runtime_versions(),
        "spec": jsonable(payload),
    }
    if inputs:
        manifest["inputs"] = dict(inputs)
    if extra:
        manifest.update(extra)
    return manifest


__all__ = [
    "build_manifest",
    "git_dirty",
    "git_revision",
    "runtime_versions",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "utc_now",
]
