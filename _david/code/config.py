from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical Single Source of Truth for all LBDD hyperparameters and execution settings.

    WHY:
    Eliminates duplicated and conflicting default parameters previously scattered across
    run.py, pipeline.py, models.py, and environment.py.

    WHAT:
    - Reinforcement Learning hyperparameters (epochs, patience, samples, criteria, epsilon, thresholds).
    - 3D conformer and scoring environment settings (max conformers, energy window, timeout).
    - Dynamic hardware preferences (GPU device, CPU core ratio).
    """

    # Reinforcement Learning Exploration
    epochs: int = 35
    patience: int = 25
    min_epochs: int = 5
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
    render_pareto: bool = False


@dataclass(frozen=True)
class RunPaths:
    """Canonical directory layout and target paths for a pipeline run.

    WHY:
    Direct-to-disk architecture eliminates 25+ lines of fragile post-hoc shutil.move
    cleanups by ensuring all components know their exact destination upfront.

    WHAT:
    - root: Presentation deliverables (PNGs, summary.md, top_prioritized_leads.tsv).
    - models_dir: Checkpoint weights (david_agent_peak.pkg, david_agent.pkg).
    - raw_data_dir: Conformer SDFs, seed fragments, intermediate TSVs (fit, smiles, candidates).
    - logs_dir: Process PID, daemon logs, and pipeline_status.json.
    """

    root: Path
    models_dir: Path
    raw_data_dir: Path
    logs_dir: Path

    @classmethod
    def resolve(
        cls,
        base_dir: Path | str | None = None,
        phase: str | None = None,
        run_id: str | None = None,
    ) -> RunPaths:
        """Resolve authoritative output targets in a phase-agnostic manner."""
        rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_dir:
            root = Path(base_dir)
        elif phase:
            root = PROJECT_ROOT / "_david" / "output" / phase / rid
        else:
            root = PROJECT_ROOT / "_david" / "output" / rid

        models_dir = root / "models"
        raw_data_dir = root / "raw_data"
        logs_dir = raw_data_dir / "logs"
        return cls(root=root, models_dir=models_dir, raw_data_dir=raw_data_dir, logs_dir=logs_dir)

    def create_directories(self) -> None:
        """Ensure all required target directories exist on disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
