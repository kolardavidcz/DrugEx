from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from importlib import import_module
from pathlib import Path

# Hashed per run, so an edit to DrugEx shows up in the manifest, not in the results.
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
ARTIFACTS = {
    "prior.tsv": "Scored pre-training baseline",
    "final_desired_ratio.tsv": "Samples from the best desired-ratio checkpoint",
    "final_desired_ratio_at_stop.tsv": "Samples from the policy at the freeze epoch",
    "final_gmean.tsv": "Samples from the best geometric-mean checkpoint",
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


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_digests(package: Path) -> dict[str, str]:
    return {
        str(path.relative_to(package)): digest(path)
        for path in sorted(package.rglob("*"))
        if path.suffix in {".py", ".sh"} and path.is_file() and "__pycache__" not in path.parts
    }


def kernel_fingerprint(root: Path) -> dict[str, str]:
    return {relative: digest(root / relative) or "missing" for relative in KERNEL_FILES}


def _version(module: str, attribute: str) -> str:
    try:
        return str(getattr(import_module(module), attribute))
    except Exception:
        return "unavailable"


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_revision(root: Path) -> dict[str, object]:
    # A cluster release has no .git, so it declares its revision through the environment.
    declared = os.environ.get("DRUGEX_SOURCE_REVISION")
    if declared:
        return {"revision": declared, "source": "DRUGEX_SOURCE_REVISION", "dirty": None}
    revision = _git(root, "rev-parse", "HEAD")
    if revision is None:
        return {"revision": None, "source": "unavailable", "dirty": None}
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"revision": revision, "source": "git", "dirty": bool(dirty)}


def imported_drugex() -> str | None:
    try:
        import drugex

        return str(Path(drugex.__file__).resolve().parent)
    except Exception:
        return None


def describe(root: Path) -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {"python": platform.python_version()}
        | {name: _version(module, attribute) for name, module, attribute in LIBRARIES},
        "source": source_revision(root),
        "kernel_sha256": kernel_fingerprint(root),
        "drugex_imported_from": imported_drugex(),
        "kernel_hashed_from": str(root),
        "source_sha256": source_digests(Path(__file__).parent),
        "determinism": "Seeded RNGs; deterministic GPU kernels are not enforced",
        "environment": {name: os.environ.get(name) for name in TRACKED_ENV},
    }


class Manifest:
    """A run's state on disk, rewritten whole after every stage."""

    def __init__(self, path: Path, **fields: object) -> None:
        self.path = path
        self.data: dict[str, object] = {"state": "running", **fields}
        self.update()

    def update(self, **fields: object) -> None:
        self.data.update(fields)
        # Through a temporary file: a half-written manifest would read as a finished run.
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        temporary.replace(self.path)
