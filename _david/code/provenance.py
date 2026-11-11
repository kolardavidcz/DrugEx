"""What a run needs to be re-creatable later: versions, revision, host, and a state.

``config.json`` records what was asked for. This records what actually ran, which is a
different question and the one a thesis has to answer two years later. The three parts
that matter:

* **the native kernel's fingerprint.** Every number in this project comes out of
  ``drugex/training/scorers/rocs_rdkit.py`` and its two neighbours. The archived conformer
  study is only comparable with a run whose kernel is byte-identical to the one it
  measured, and "same commit" does not prove that in a tree someone edited by hand. A
  SHA-256 per file does.
* **a state.** A run killed at walltime leaves a directory that looks complete. Without a
  state a reader cannot tell a finished run from a corpse, and the analysis will happily
  plot half of one.
* **the environment.** ``DRUGEX_MP_CONTEXT`` and the thread caps change timing, not
  results; the library versions change results.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

#: The native modules every score depends on. A change in any of them invalidates a
#: comparison against an earlier run, so each one is hashed by content.
KERNEL_FILES = (
    "drugex/training/scorers/rocs_rdkit.py",
    "drugex/training/scorers/conformer_generators.py",
    "drugex/training/scorers/properties.py",
    "drugex/training/rewards.py",
)

#: What each file a run writes is for, so a reader two years later does not have to guess
#: which table a number came from. Keys are run-directory-relative; a missing file is
#: simply absent from the manifest.
ARTIFACTS = {
    "prior.tsv": "pre-training baseline arm: the prior's own output, scored, thresholds applied",
    "final_desired_ratio.tsv": "arm sampled from the best-desired_ratio checkpoint",
    "final_desired_ratio_at_stop.tsv": "arm sampled from the policy live at the freeze epoch",
    "final_gmean.tsv": "arm sampled from the best-avg_gmean checkpoint",
    "rl_fit.tsv": "per-epoch training trace; longitudinal diagnostics only, never pooled "
    "into arm statistics",
    "training_samples.tsv": "every molecule the policy gradient was trained on, with its "
    "scores, by epoch -- 10x more scored molecules than the evaluation arm and the "
    "widest pool of candidates the run produced",
    "rl_smiles.tsv": "the held-out evaluation arm of each epoch (n_eval molecules), by "
    "epoch -- NOT the n_samples molecules the policy gradient trained on, which are "
    "scored and discarded",
    "reference.sdf": "the EGCG reference bank every score is measured against",
    "config.json": "the resolved configuration, including thresholds and final summary",
    "report.md": "generated summary of the arms, thresholds and settings",
}

#: Environment variables that change how a run executes. Recorded, never set from here.
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
    """SHA-256 of one file, or None when it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digests(package: Path) -> dict[str, str]:
    """SHA-256 of every source file of this package, keyed by name.

    The git revision says which commit the code came from; this says what the files
    actually contained when the run started, which is the stronger claim and the only one
    that survives an uncommitted edit.
    """
    package = Path(package)
    # Recursive, and keyed by the path relative to the package: a flat glob would silently
    # stop covering anything that moves into a subpackage, leaving most of the code
    # unhashed while the manifest still looked complete.
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts
    }


def _versions() -> dict[str, str]:
    """Version of every library whose behaviour lands in a score."""
    versions = {"python": platform.python_version()}
    for name, module, attribute in (
        ("rdkit", "rdkit.rdBase", "rdkitVersion"),
        ("torch", "torch", "__version__"),
        ("numpy", "numpy", "__version__"),
        ("pandas", "pandas", "__version__"),
    ):
        try:
            imported = __import__(module, fromlist=[attribute])
            versions[name] = str(getattr(imported, attribute))
        except Exception:  # a version string is never worth failing a run over
            versions[name] = "unavailable"
    return versions


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_revision(root: Path) -> dict[str, object]:
    """The commit this code came from, and whether the tree had edits on top of it.

    ``DRUGEX_SOURCE_REVISION`` wins when set, because a cluster release is an extracted
    ``git archive`` with no ``.git`` directory to ask.
    """
    declared = os.environ.get("DRUGEX_SOURCE_REVISION")
    if declared:
        return {"revision": declared, "source": "DRUGEX_SOURCE_REVISION", "dirty": None}
    revision = _git(root, "rev-parse", "HEAD")
    if revision is None:
        return {"revision": None, "source": "unavailable", "dirty": None}
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"revision": revision, "source": "git", "dirty": bool(status)}


def kernel_fingerprint(root: Path) -> dict[str, str]:
    """SHA-256 of each native scoring module, so "same kernel" is checkable, not assumed."""
    digests = {}
    for relative in KERNEL_FILES:
        path = root / relative
        if not path.is_file():
            digests[relative] = "missing"
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def imported_drugex() -> str | None:
    """Where ``import drugex`` actually resolved.

    ``kernel_sha256`` hashes the files in the release directory, which is only the truth
    if those are the files Python imported. On the cluster the release is prepended to
    both ``cwd`` and ``PYTHONPATH`` so it wins over anything installed in the venv -- but
    recording the resolved path is what makes that checkable rather than assumed.
    """
    try:
        import drugex

        return str(Path(drugex.__file__).resolve().parent)
    except Exception:
        return None


def describe(root: Path) -> dict[str, object]:
    """Everything about the machine and the code that a rerun would have to match."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": _versions(),
        "source": source_revision(root),
        "kernel_sha256": kernel_fingerprint(root),
        "drugex_imported_from": imported_drugex(),
        "kernel_hashed_from": str(root),
        "source_sha256": source_digests(Path(__file__).parent),
        "determinism": (
            "RNGs are seeded from Config.seed; RDKit ETKDG uses DrugEx's own seed. "
            "Deterministic GPU kernels are not enforced, so a CUDA run is reproducible "
            "only up to float non-associativity."
        ),
        "environment": {name: os.environ.get(name) for name in TRACKED_ENV},
    }


class Manifest:
    """``manifest.json``, rewritten in place as the run moves through its states.

    States are ``running`` -> ``complete`` or ``failed``. The file exists from before the
    first expensive stage, so a run killed at walltime leaves ``running`` behind and any
    reader can tell it apart from a finished one.
    """

    def __init__(self, path: Path, **fields: object) -> None:
        self.path = Path(path)
        self.data: dict[str, object] = {"state": "running", **fields}
        self.write()

    def update(self, **fields: object) -> None:
        self.data.update(fields)
        self.write()

    def finish(self, state: str, **fields: object) -> None:
        self.data.update(fields)
        self.data["state"] = state
        self.write()

    def write(self) -> None:
        # Written through a temporary file: a SIGTERM mid-write would otherwise leave
        # truncated JSON, which is worse than a stale manifest because it cannot be read.
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        temporary.replace(self.path)
