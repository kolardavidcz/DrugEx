"""Immutable, hashable contracts for conformer preparation and study design.

The whole package is built around one idea: every expensive step is a pure
function of an explicit, serializable specification.  ``contracts`` defines
those specifications.  Nothing here touches RDKit, the filesystem, or the
network, which keeps the domain model reviewable on its own.

Conventions
-----------
* All ``float`` fields are finite and validated in ``__post_init__``.
* ``Optimization`` comes from :mod:`_david.code.chemistry` and is already a
  frozen dataclass, so it hashes and compares by value.
* ``cap = 0`` means *retain the whole pool* (the historical ``all`` arm).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Literal

from _david.code.chemistry import Optimization

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

#: Bump when a change to preparation/embedding/minimization/selection alters the
#: numerical result of a cache entry.  Cache keys embed this string, so stale
#: artifacts are never silently reused across algorithm changes.
ALGORITHM_VERSION = "conformer-algorithm-2026-11-03"


class SelectionMode(str, Enum):
    """How a reference pool is reduced to a deployable bank."""

    ENERGY = "energy"
    """Keep the lowest-energy conformers, round-robin across stereoisomers."""

    BUTINA = "butina"
    """Butina RMSD clusters, lowest-energy representative per cluster."""

    BALANCED = "balanced"
    """Butina RMSD clusters, then round-robin across stereoisomers."""

    DIVERSE = "diverse"
    """Farthest-first shape+color diversity seeded by lowest energy."""


class ScoreVariant(str, Enum):
    """Supported 3D similarity families."""

    TANIMOTO = "tanimoto"
    """Classic TanimotoCombo on the physical ``[0, 2]`` scale."""

    TVERSKY = "tversky"
    """Asymmetric Tversky shape/color with explicit ``(alpha, beta)``."""


class ScoreScale(str, Enum):
    """Output scale requested from the scorer."""

    PHYSICAL = "physical"
    """Shape + color TanimotoCombo, ``[0, 2]``."""

    NORMALIZED = "normalized"
    """Half of the physical score, ``[0, 1]``; matches the RL reward axis."""


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    """Validate that ``value`` is a finite float above an optional bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _positive_int(value: int, name: str) -> int:
    """Validate a strictly positive integer that is not a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def jsonable(value: Any) -> Any:
    """Recursively convert a contract into a deterministic JSON-compatible value.

    Dataclasses become mappings, enums become their string values, tuples become
    lists, and paths become POSIX strings.  Mapping keys are sorted by the json
    serializer configured in :func:`fingerprint`.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 digest of any contract payload."""
    import hashlib

    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Molecule identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeSpec:
    """One named molecular input before any chemical transformation."""

    name: str
    smiles: str
    source: str = "user"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MoleculeSpec.name must not be empty")
        if not self.smiles.strip():
            raise ValueError("MoleculeSpec.smiles must not be empty")

    @property
    def identity(self) -> str:
        """Human-stable identity used in tables and manifests."""
        return self.name


# ---------------------------------------------------------------------------
# Ensemble specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedSpec:
    """ETKDGv3 budget for one stereoisomer of a prepared molecule.

    ``isomer_cap`` defaults to 1: prepare exactly the identities the input
    already specifies.  Studies that need to enumerate unassigned centres must
    raise the cap explicitly (see :func:`default_reference_spec`), so
    enumeration cost is never paid -- or silently truncated -- by accident.
    """

    attempts: int = 100
    isomer_cap: int = 1
    seed: int = 104729
    stereo_seed: int = 42
    rmsd: float | None = 0.35
    timeout: float = 30.0
    threads: int = 1
    protonate: bool = False
    canonicalize_tautomer: bool = True
    heavy_atom_cap: int | None = None
    rotatable_bond_cap: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.attempts, "EmbedSpec.attempts")
        _positive_int(self.isomer_cap, "EmbedSpec.isomer_cap")
        if self.rmsd is not None:
            _finite(self.rmsd, "EmbedSpec.rmsd", minimum=0.0)
        _finite(self.timeout, "EmbedSpec.timeout", minimum=0.0)
        if self.threads < 0:
            raise ValueError("EmbedSpec.threads must be >= 0")

    @property
    def label(self) -> str:
        """Compact identifier used in arm names and cache manifests."""
        return f"a{self.attempts}i{self.isomer_cap}s{self.seed}"


@dataclass(frozen=True)
class MinimizeSpec:
    """Force-field relaxation policy and post-relaxation energy window."""

    profile: Optimization | None = Optimization("MMFF94s", 78.4)
    energy_window: float | None = 8.0

    def __post_init__(self) -> None:
        if self.energy_window is not None:
            _finite(self.energy_window, "MinimizeSpec.energy_window", minimum=0.0)
        if self.profile is not None and self.profile.method not in {
            "none",
            "UFF",
            "MMFF94",
            "MMFF94s",
        }:
            raise ValueError(f"Unsupported force field: {self.profile.method!r}")

    @property
    def enabled(self) -> bool:
        """True when a real force field will be executed."""
        return self.profile is not None and self.profile.method != "none"

    @property
    def label(self) -> str:
        """Compact identifier used in arm names."""
        if not self.enabled:
            return "raw"
        profile = self.profile
        assert profile is not None
        dielectric = f"{profile.dielectric:g}"
        window = "none" if self.energy_window is None else f"{self.energy_window:g}"
        return f"{profile.method.lower()}e{dielectric}w{window}"


@dataclass(frozen=True)
class SelectSpec:
    """How to reduce a conformer pool to a reference bank."""

    mode: SelectionMode = SelectionMode.DIVERSE
    cap: int = 4
    rmsd_cutoff: float = 0.35
    balance_isomers: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.cap, bool) or not isinstance(self.cap, int) or self.cap < 0:
            raise ValueError("SelectSpec.cap must be an integer >= 0 (0 = keep all)")
        _finite(self.rmsd_cutoff, "SelectSpec.rmsd_cutoff", minimum=0.0)

    @property
    def label(self) -> str:
        """Compact identifier used in arm names."""
        cap = "all" if self.cap == 0 else str(self.cap)
        return f"{self.mode.value}-{cap}-{self.rmsd_cutoff:g}"


@dataclass(frozen=True)
class ScoreSpec:
    """3D shape/color similarity specification."""

    variant: ScoreVariant = ScoreVariant.TANIMOTO
    scale: ScoreScale = ScoreScale.PHYSICAL
    alpha: float = 1.0
    beta: float = 1.0
    shape_weight: float = 0.5
    color_weight: float = 0.5

    def __post_init__(self) -> None:
        _finite(self.alpha, "ScoreSpec.alpha", minimum=0.0)
        _finite(self.beta, "ScoreSpec.beta", minimum=0.0)
        _finite(self.shape_weight, "ScoreSpec.shape_weight", minimum=0.0)
        _finite(self.color_weight, "ScoreSpec.color_weight", minimum=0.0)
        if self.alpha + self.beta <= 0:
            raise ValueError("ScoreSpec requires alpha + beta > 0")
        if self.shape_weight + self.color_weight <= 0:
            raise ValueError("ScoreSpec requires shape_weight + color_weight > 0")
        if self.variant is ScoreVariant.TANIMOTO and (
            abs(self.alpha - 1.0) > 1e-9 or abs(self.beta - 1.0) > 1e-9
        ):
            # Classic TanimotoCombo *is* symmetric Tversky.  Accepting an
            # asymmetric (alpha, beta) here would silently ignore it and report
            # a different metric than the one requested.
            raise ValueError(
                "ScoreSpec.variant='tanimoto' requires alpha=beta=1; "
                "use ScoreVariant.TVERSKY for an asymmetric setting"
            )

    @property
    def label(self) -> str:
        """Compact identifier used in tables."""
        if self.variant is ScoreVariant.TANIMOTO:
            return "tanimoto_combo"
        return (
            f"tversky-a{self.alpha:g}-b{self.beta:g}"
            f"-s{self.shape_weight:g}-c{self.color_weight:g}"
        )


def _reference_embed_spec() -> EmbedSpec:
    """Production reference embedding: 100 attempts, I=8, RMSD pruning 0.35 A."""
    return EmbedSpec(
        attempts=100,
        isomer_cap=8,
        seed=104729,
        stereo_seed=42,
        rmsd=0.35,
        timeout=60.0,
    )


@dataclass(frozen=True)
class ReferenceSpec:
    """Everything needed to build one deployable reference bank."""

    embed: EmbedSpec = field(default_factory=_reference_embed_spec)
    minimize: MinimizeSpec = field(default_factory=MinimizeSpec)
    select: SelectSpec = field(default_factory=SelectSpec)

    @property
    def label(self) -> str:
        return f"ref[{self.embed.label}|{self.minimize.label}|{self.select.label}]"


@dataclass(frozen=True)
class CandidateSpec:
    """Everything needed to build one candidate ensemble."""

    embed: EmbedSpec = field(default_factory=lambda: EmbedSpec(attempts=20, isomer_cap=4, seed=42, rmsd=0.5))
    minimize: MinimizeSpec = field(default_factory=lambda: MinimizeSpec(profile=None, energy_window=None))

    @property
    def label(self) -> str:
        return f"cand[{self.embed.label}|{self.minimize.label}]"


@dataclass(frozen=True)
class ArmSpec:
    """One row of a study design: a reference bank, a candidate ensemble, a scorer."""

    name: str
    reference: ReferenceSpec
    candidate: CandidateSpec
    score: ScoreSpec = field(default_factory=ScoreSpec)
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ArmSpec.name must not be empty")


@dataclass(frozen=True)
class SeedPlan:
    """Independent RNG streams for a study.

    Reference and candidate streams are deliberately disjoint.  The historical
    head-over-heels bug was a multiplicative collision (``43 * 42 == 42 * 43``)
    between reference ordinal 42 and candidate ordinal 43; keeping the bases
    distinct and asserting it in :func:`_david.code.chemistry.verify_disjoint_conformer_seeds`
    prevents a silent recurrence.
    """

    reference_base: int = 104729
    candidate_base: int = 42
    stereo: int = 42
    repeats: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        for name in ("reference_base", "candidate_base", "stereo"):
            _positive_int(getattr(self, name), f"SeedPlan.{name}")
        if not self.repeats:
            raise ValueError("SeedPlan.repeats must not be empty")


@dataclass(frozen=True)
class StudySpec:
    """A complete, declarative study matrix."""

    name: str
    molecules: tuple[MoleculeSpec, ...]
    arms: tuple[ArmSpec, ...]
    output_dir: Path
    seeds: SeedPlan = field(default_factory=SeedPlan)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("StudySpec.name must not be empty")
        if not self.molecules:
            raise ValueError("StudySpec requires at least one molecule")
        if not self.arms:
            raise ValueError("StudySpec requires at least one arm")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError(f"Arm names must be unique, got duplicates in {names}")


# ---------------------------------------------------------------------------
# Convenience constructors for the four canonical components
# ---------------------------------------------------------------------------


def default_reference_spec(
    *,
    method: Optimization | None = Optimization("MMFF94s", 78.4),
    attempts: int = 100,
    isomer_cap: int = 8,
    window: float | None = 8.0,
    mode: SelectionMode = SelectionMode.DIVERSE,
    cap: int = 4,
    rmsd_cutoff: float = 0.35,
) -> ReferenceSpec:
    """Reference bank matching the deployable ``K=4`` production default."""
    return ReferenceSpec(
        embed=EmbedSpec(
            attempts=attempts,
            isomer_cap=isomer_cap,
            seed=104729,
            stereo_seed=42,
            rmsd=0.35,
            timeout=60.0,
        ),
        minimize=MinimizeSpec(profile=method, energy_window=window),
        select=SelectSpec(mode=mode, cap=cap, rmsd_cutoff=rmsd_cutoff),
    )


def default_candidate_spec(
    *,
    method: Optimization | None = None,
    attempts: int = 20,
    isomer_cap: int = 4,
    seed: int = 42,
    rmsd: float | None = 0.5,
) -> CandidateSpec:
    """Candidate ensemble matching the deployable ``C=20, I=4`` default."""
    return CandidateSpec(
        embed=EmbedSpec(
            attempts=attempts,
            isomer_cap=isomer_cap,
            seed=seed,
            stereo_seed=42,
            rmsd=rmsd,
            timeout=30.0,
        ),
        minimize=MinimizeSpec(profile=method, energy_window=None),
    )


def arm(
    name: str,
    *,
    reference_method: Optimization | None = Optimization("MMFF94s", 78.4),
    candidate_method: Optimization | None = None,
    reference_attempts: int = 100,
    reference_isomers: int = 8,
    reference_cap: int = 4,
    reference_window: float | None = 8.0,
    mode: SelectionMode = SelectionMode.DIVERSE,
    rmsd_cutoff: float = 0.35,
    candidate_attempts: int = 20,
    candidate_isomers: int = 4,
    candidate_seed: int = 42,
    candidate_rmsd: float | None = 0.5,
    score: ScoreSpec | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> ArmSpec:
    """Build an :class:`ArmSpec` without repeating the nested constructors."""
    return ArmSpec(
        name=name,
        reference=default_reference_spec(
            method=reference_method,
            attempts=reference_attempts,
            isomer_cap=reference_isomers,
            window=reference_window,
            mode=mode,
            cap=reference_cap,
            rmsd_cutoff=rmsd_cutoff,
        ),
        candidate=default_candidate_spec(
            method=candidate_method,
            attempts=candidate_attempts,
            isomer_cap=candidate_isomers,
            seed=candidate_seed,
            rmsd=candidate_rmsd,
        ),
        score=score or ScoreSpec(),
        description=description,
        tags=tags,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "ArmSpec",
    "CandidateSpec",
    "EmbedSpec",
    "MinimizeSpec",
    "MoleculeSpec",
    "ReferenceSpec",
    "ScoreScale",
    "ScoreSpec",
    "ScoreVariant",
    "SeedPlan",
    "SelectSpec",
    "SelectionMode",
    "StudySpec",
    "arm",
    "default_candidate_spec",
    "default_reference_spec",
    "fingerprint",
    "jsonable",
]
