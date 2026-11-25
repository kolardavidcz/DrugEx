"""Run provenance fingerprinting, atomic file writes, and manifest persistence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from importlib import import_module
from pathlib import Path

KERNEL_FILES = (
    "drugex/training/scorers/rocs_rdkit.py",
    "drugex/training/scorers/conformer_generators.py",
    "drugex/training/scorers/properties.py",
    "drugex/training/rewards.py",
)
LIBRARIES = (
    ("rdkit", "rdkit.rdBase", "rdkitVersion"),
    ("torch", "torch", "__version__"),
    ("numpy", "numpy", "__version__"),
    ("pandas", "pandas", "__version__"),
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "prior.tsv": "Scored pre-training baseline",
    "final_desired_ratio_at_stop.tsv": "Samples from the policy at the epoch training stopped",
    "rl_fit.tsv": "Per-epoch training diagnostics",
    "training_samples.tsv": "Scored policy-gradient samples by epoch",
    "rl_smiles.tsv": "Scored held-out evaluation samples by epoch",
    "reference.sdf": "Reference conformer bank",
    "config.json": "Resolved configuration and final summary",
    "report.md": "Run summary",
}
TRACKED_ENV = (
    "DRUGEX_MP_CONTEXT",
    "DRUGEX_SCORE_NJOBS",
    "DRUGEX_SCORE_MOL_TIMEOUT",
    "DRUGEX_SOURCE_REVISION",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PBS_JOBID",
    "SCRATCHDIR",
)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` atomically via a PID-scoped sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def digest(path: Path) -> str | None:
    """Hex SHA-256 of ``path``, or ``None`` if absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_digests(package: Path) -> dict[str, str]:
    """SHA-256 of every ``.py`` and ``.sh`` file under ``package``."""
    return {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.suffix in {".py", ".sh"} and p.is_file() and "__pycache__" not in p.parts
    }


def kernel_fingerprint(root: Path) -> dict[str, str]:
    """SHA-256 of upstream DrugEx modules that can directly affect molecule scores."""
    return {rel: digest(root / rel) or "missing" for rel in KERNEL_FILES}


def _git(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def source_revision(root: Path) -> dict[str, object]:
    """Current git commit (or staged release env var) for ``root``."""
    if declared := os.environ.get("DRUGEX_SOURCE_REVISION"):
        return {"revision": declared, "source": "DRUGEX_SOURCE_REVISION", "dirty": None}
    rev = _git(root, "rev-parse", "HEAD")
    if rev is None:
        return {"revision": None, "source": "unavailable", "dirty": None}
    return {"revision": rev, "source": "git", "dirty": bool(_git(root, "status", "--porcelain", "-uno"))}


def imported_drugex() -> str | None:
    """Resolved package directory of the imported ``drugex`` module."""
    import drugex

    return str(Path(drugex.__file__).resolve().parent)


def describe(root: Path) -> dict[str, object]:
    """Snapshot host, library versions, git/release revision, and source digests."""
    versions = {"python": platform.python_version()} | {
        name: str(getattr(import_module(mod), attr)) for name, mod, attr in LIBRARIES
    }
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
        "source": source_revision(root),
        "kernel_sha256": kernel_fingerprint(root),
        "drugex_imported_from": imported_drugex(),
        "kernel_hashed_from": str(root),
        "source_sha256": source_digests(PACKAGE_ROOT),
        "determinism": "Seeded RNGs; deterministic GPU kernels are not enforced",
        "environment": {name: os.environ.get(name) for name in TRACKED_ENV},
    }


class Manifest:
    """Run state on disk, rewritten atomically after every stage."""

    def __init__(self, path: Path, **fields: object) -> None:
        self.path = path
        self.data: dict[str, object] = {"state": "running", **fields}
        self.update()

    def update(self, **fields: object) -> None:
        """Merge ``fields`` into the manifest and persist atomically."""
        self.data.update(fields)
        atomic_write_text(self.path, json.dumps(self.data, indent=2, default=str) + "\n")
