from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EGCG = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"


@dataclass(frozen=True)
class Config:
    reference_smiles: str = EGCG

    epochs: int = 70
    patience: int = 20
    min_epochs: int = 40
    n_samples: int = 1000
    n_eval: int = 100
    batch_size: int = 128
    epsilon: float = 0.0
    beta: float = 0.0
    scheme: str = "WS"

    alignment_shape_weight: float = 0.5
    color_ff: str = "default"
    include_qed: bool = False
    sa_reward_zero: float = 5.0
    sa_reward_one: float = 3.0
    fixed_thresholds: tuple[float, ...] = ()
    threshold_percentile: float = 85.0
    threshold_from_reference: tuple[str, ...] = ("SA", "QED")
    n_prior: int = 5000
    n_final: int = 5000

    minimize: bool = True
    dielectric: float = 78.4
    dielectric_model: int = 1
    max_iterations: int = 500
    max_conformers: int = 20
    max_isomers: int = 4
    max_heavy_atoms: int = 45
    max_rotatable_bonds: int = 15
    timeout: int = 30
    reference_attempts: int = 200
    reference_conformers: int = 2
    reference_rmsd: float = 0.5

    seed: int = 42
    n_jobs: int = -1

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
            raise ValueError("sa_reward_one must be below sa_reward_zero: lower SA is better")

    def pretrained(self) -> tuple[Path, Path]:
        base = PROJECT_ROOT / "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        paths = base / f"{base.name}.pkg", base / f"{base.name}.vocab"
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing pretrained model: " + ", ".join(missing))
        return paths
