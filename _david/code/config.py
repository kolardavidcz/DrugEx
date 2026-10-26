from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _david.code.schema import std

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Canonical default target query molecule (Epigallocatechin gallate - EGCG)
DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical single source of truth for all LBDD hyperparameters and execution settings."""

    # Reinforcement Learning Exploration
    epochs: int = 50
    patience: int = 51  # Decoupled (> epochs) to prevent premature stopping
    min_epochs: int = 50

    n_samples: int = 500  # Molecule samples generated per training epoch
    val_samples: int = 100
    n_eval_samples: int = 1000  # Statistical evaluation sample pool size
    batch_size: int = 64

    criteria: str = std.AVG_AMEAN
    beta: float = 0.0  # Baseline policy penalization (0.0 = unpenalized)

    # Exploration Noise (Epsilon)
    epsilon: float = 0.20
    epsilon_end: float = 0.05
    decay_epochs: int = 50
    decay_type: str = "cosine"  # "cosine" or "linear"
    use_epsilon_decay: bool = False

    # Multi-Objective Desirability Thresholds (ROCS, SA, QED)
    thresholds: tuple[float, float, float] = (0.70, 0.27, 0.50)

    # Scoring & Rewards
    score_variant: str = "tanimoto_combo"  # "tanimoto_combo" or "tversky_subshape"
    reward_scheme: str = (
        "weighted_sum"  # "weighted_sum", "pareto_crowding", "harrington"
    )
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25)

    # 3D Conformer & Chemistry Settings
    max_conformers: int = 30
    max_reference_conformers: int = 3  # Hawkins low-energy basin ensemble size
    energy_window_kcal: float = 8.0
    timeout_seconds: int = 60
    random_seed: int = 42

    # Hardware & Multiprocessing (Authoritative 0.90 CPU allocation)
    gpu: int | str = "auto"
    cpu_ratio: float = 0.90  # 90% of available CPU cores by default
    conformer_threads: int = 0

    # Checkpoint Paths
    agent_path: Path | str | None = None
    vocab_path: Path | str | None = None

    def resolve_pretrained_paths(self) -> tuple[Path, Path]:
        """Resolve paths to pretrained SMILES-RNN checkpoint and vocabulary."""
        if self.agent_path and self.vocab_path:
            p_agent, p_voc = Path(self.agent_path), Path(self.vocab_path)
            if p_agent.is_file() and p_voc.is_file():
                return p_agent, p_voc

        candidates = [
            PROJECT_ROOT
            / "tutorial"
            / "data"
            / "models"
            / "pretrained"
            / "smiles-rnn"
            / "Papyrus05.5_smiles_rnn_PT",
            Path(
                "/home/kolarv/DrugEx/tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
            ),
            Path(
                "/home/kolar/DrugEx/tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
            ),
        ]
        for cand in candidates:
            pkg = cand / "Papyrus05.5_smiles_rnn_PT.pkg"
            voc = cand / "Papyrus05.5_smiles_rnn_PT.vocab"
            if pkg.is_file() and voc.is_file():
                return pkg, voc

        raise FileNotFoundError(
            "Could not locate pretrained Papyrus SMILES-RNN checkpoint (.pkg) and vocabulary (.vocab)."
        )


@dataclass(frozen=True)
class RunPaths:
    """Canonical directory layout and target paths for a pipeline run."""

    root: Path
    models_dir: Path
    raw_data_dir: Path
    logs_dir: Path

    @property
    def figures_dir(self) -> Path:
        """Directory target for all publication figures and visual diagnostics."""
        return self.root / "figures"

    def create_directories(self) -> None:
        """Ensure all required target directories exist on disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        self.create_directories()

    @classmethod
    def resolve(
        cls,
        base_dir: Path | str | None = None,
        run_id: str | None = None,
    ) -> RunPaths:
        """Resolve authoritative output targets directly in _david/output/<timestamp>."""
        rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(base_dir) if base_dir else PROJECT_ROOT / "_david" / "output" / rid
        return cls(
            root=root,
            models_dir=root / "models",
            raw_data_dir=root / "raw_data",
            logs_dir=root / "logs",
        )
