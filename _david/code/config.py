from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Canonical default target query molecule (Epigallocatechin gallate - EGCG)
DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical single source of truth for all LBDD hyperparameters and execution settings."""

    # Reinforcement Learning Exploration
    epochs: int = 35
    patience: int = 10
    min_epochs: int = 25
    n_samples: int = 200
    val_samples: int = 100
    n_eval_samples: int = 200
    criteria: str = "avg_amean"
    epsilon: float = 0.2
    beta: float = 0.0
    thresholds: tuple[float, float] = (0.5, 0.5)

    # 3D Conformer & Chemistry Settings
    max_conformers: int = 30
    energy_window_kcal: float = 8.0
    timeout_seconds: int = 60
    random_seed: int = 42

    # Hardware & Multiprocessing
    gpu: int | str = "auto"
    cpu_ratio: float | None = None


@dataclass(frozen=True)
class RunPaths:
    """Canonical directory layout and target paths for a pipeline run."""

    root: Path
    models_dir: Path
    raw_data_dir: Path
    logs_dir: Path

    def __post_init__(self) -> None:
        """Ensure all required target directories exist on disk immediately upon initialization."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve(
        cls,
        base_dir: Path | str | None = None,
        run_id: str | None = None,
    ) -> RunPaths:
        """Resolve authoritative output targets directly in _david/output/<timestamp>."""
        rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(base_dir) if base_dir else PROJECT_ROOT / "_david" / "output" / rid
        models_dir = root / "models"
        raw_data_dir = root / "raw_data"
        logs_dir = raw_data_dir / "logs"
        return cls(root=root, models_dir=models_dir, raw_data_dir=raw_data_dir, logs_dir=logs_dir)
