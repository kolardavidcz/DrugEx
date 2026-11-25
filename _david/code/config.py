"""Experiment configuration: every hyperparameter of one DrugEx RL run in one place."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from drugex.training.rewards import (
    ParetoCrowdingDistance,
    ParetoTanimotoDistance,
    WeightedSum,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EGCG = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

#: DrugEx reward schemes by short name (Pareto ignores thresholds; WS converts them to weights).
SCHEMES = {
    "PRCD": ParetoCrowdingDistance,
    "PRTD": ParetoTanimotoDistance,
    "WS": WeightedSum,
}
#: Published TanimotoCombo ceiling [0, 2]; rewards are scaled onto [0, 1].
ROCS_PUBLISHED_MAX = 2.0
#: Objectives scored from 2D structure alone (reference value can stand in for a prior percentile).
MOLECULE_LEVEL = ("SA", "QED")

_T = TypeVar("_T")


@dataclass(frozen=True)
class Config:
    """Immutable settings for one DrugEx RL run, conformer generator, and ROCS scorer."""

    reference_smiles: str = EGCG  # query molecule; its 3D conformer bank is built from this

    # Reinforcement learning
    epochs: int = 70  # hard cap; patience normally stops earlier
    patience: int = 20  # epochs without improvement before early stopping
    min_epochs: int = 40  # earliest epoch early stopping may trigger
    n_samples: int = 1000  # molecules scored per epoch for the policy gradient
    n_eval: int = 100  # held-out eval size (retained for manifest/legacy compatibility)
    batch_size: int = 128  # molecules per policy-gradient step
    epsilon: float = 0.0  # per-token probability of mutating from the frozen prior
    scheme: str = "WS"  # PRCD | PRTD | WS
    beta: float = 0.0  # reward baseline subtracted in the policy gradient

    # Objectives & thresholds
    alignment_shape_weight: float = 0.5  # rdShapeAlign pose-optimizer weight [0, 1]
    color_ff: str = "default"  # default | fdef | oe_mills_dean
    include_qed: bool = False  # add QED as a third objective alongside ROCS and SA
    sa_reward_zero: float = 5.0  # raw SA >= this maps to desirability 0.0
    sa_reward_one: float = 3.0  # raw SA <= this maps to desirability 1.0
    fixed_thresholds: tuple[float, ...] = ()  # explicit thresholds overriding percentile calibration
    threshold_percentile: float = 85.0  # prior percentile used to calibrate thresholds
    threshold_from_reference: tuple[str, ...] = ("SA", "QED")  # objectives calibrated to reference
    n_prior: int = 5000  # prior molecules sampled to calibrate thresholds
    n_final: int = 5000  # molecules sampled from the final policy checkpoint

    # Conformers & MMFF94s relaxation
    minimize: bool = True  # relax reference and candidate conformers with MMFF94s
    dielectric: float = 78.4  # solvent dielectric constant (78.4 = water)
    dielectric_model: int = 1  # 1 = constant, 2 = distance-dependent
    max_iterations: int = 500  # MMFF94s minimization iteration cap per conformer
    max_conformers: int = 20  # candidate conformers per stereoisomer
    max_isomers: int = 4  # enumerated unassigned stereoisomers cap
    max_heavy_atoms: int = 45  # candidates above this are skipped (score 0)
    max_rotatable_bonds: int = 15  # candidates above this are skipped (score 0)
    timeout: int = 30  # per-isomer ETKDG timeout in seconds (0 = unlimited)

    # Reference bank (Rule B: ascending MMFF94s energy with heavy-atom RMSD > reference_rmsd)
    reference_attempts: int = 200  # initial ETKDGv3 conformers embedded for the reference
    reference_conformers: int = 4  # poses kept in reference.sdf
    reference_rmsd: float = 0.5  # angstrom; heavy-atom RMSD cutoff against kept poses

    # Execution
    seed: int = 42  # Python / NumPy / PyTorch seed
    n_jobs: int = -1  # -1 = all allocated cores (capped by DRUGEX_SCORE_NJOBS)
    use_prior_cache: bool = True  # reuse cached reference.sdf + raw prior scores when axes match
    track_conformer_basins: bool = False  # log conformer RMSD/energy basins during scoring
    output_root: Path = PROJECT_ROOT / "_david" / "output"

    def __post_init__(self) -> None:
        if self.dielectric_model not in (1, 2):
            raise ValueError(f"dielectric_model must be 1 or 2, not {self.dielectric_model!r}")
        if not all(map(math.isfinite, self.fixed_thresholds)):
            raise ValueError(f"fixed_thresholds must be finite, not {self.fixed_thresholds!r}")
        if not 0 <= self.alignment_shape_weight <= 1:
            raise ValueError("alignment_shape_weight must be in [0, 1]")
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be in (0, 100)")
        if min(self.n_samples, self.n_eval, self.epochs) < 1:
            raise ValueError("n_samples, n_eval and epochs must be positive")
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs cannot exceed epochs")
        if self.sa_reward_one >= self.sa_reward_zero:
            raise ValueError("sa_reward_one must be below sa_reward_zero")

    @classmethod
    def from_dict(cls, recorded: Mapping[str, Any]) -> Config:
        """Restore a :class:`Config` from a recorded ``config.json`` dictionary."""
        return dataclass_from_dict(cls, recorded)

    def pretrained(self) -> tuple[Path, Path]:
        """Return ``(checkpoint_path, vocab_path)`` for the Papyrus pretrained SMILES-RNN."""
        base = PROJECT_ROOT / "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        paths = base / f"{base.name}.pkg", base / f"{base.name}.vocab"
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing pretrained model: " + ", ".join(missing))
        return paths


def coerce_field_value(field: dataclasses.Field[Any], raw: str) -> Any:
    """Coerce a CLI ``name=value`` string into ``field``'s declared type."""
    ann = str(field.type)
    if "None" in ann and raw.lower() in ("none", "null", ""):
        return None
    if "bool" in ann:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"{field.name} is a boolean; got {raw!r}")
        return raw.lower() in ("true", "1")
    if "tuple" in ann:
        inner = float if "float" in ann else int if "int" in ann else str
        text = raw.strip().strip("()[]")
        return tuple(inner(p.strip()) for p in text.split(",") if p.strip()) if text else ()
    for name, cast in (("int", int), ("float", float), ("Path", Path)):
        if name in ann:
            return cast(raw)
    return raw


def parse_dataclass_overrides(cls: type[_T], overrides: Sequence[str]) -> _T:
    """Construct dataclass ``cls`` from repeatable ``name=value`` CLI overrides."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    values: dict[str, Any] = {}
    for item in overrides:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        if name not in fields:
            raise SystemExit(f"unknown setting {name!r}. Valid: {', '.join(sorted(fields))}")
        values[name] = coerce_field_value(fields[name], raw.strip())
    return cls(**values)


def dataclass_from_dict(cls: type[_T], recorded: Mapping[str, Any]) -> _T:
    """Restore dataclass ``cls`` from a JSON-like mapping, converting ``Path`` and ``tuple``."""
    values: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        value = recorded.get(field.name)
        if value is None:
            continue
        ann = str(field.type)
        if "Path" in ann:
            value = Path(value)
        elif "tuple" in ann:
            value = tuple(value)
        values[field.name] = value
    return cls(**values)
