from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _david.code.schema import std

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Canonical default target query molecule (Epigallocatechin gallate - EGCG)
DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)

# Pinned SHA-256 hashes of standard Papyrus SMILES-RNN pretrained checkpoint
PRETRAINED_PKG_SHA256: str = (
    "34dbc312454635e374dee6ce61897fc4889b2598cddaf292fd243f41c2369f7f"
)
PRETRAINED_VOCAB_SHA256: str = (
    "5da364a8b64662462c5b7a7b29c3cbe3c4aad4aaf982ec1ba339ffa6735154ce"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical single source of truth for all LBDD hyperparameters and execution settings."""

    # Reinforcement Learning Exploration
    epochs: int = 50
    patience: int = 51  # decoupled: > epochs avoids premature early stopping
    min_epochs: int = (
        50  # decoupled: matches epochs so threshold does not select weights
    )
    n_samples: int = 500  # molecules sampled per RL training epoch
    batch_size: int = 64
    criteria: str = std.AVG_AMEAN

    # Exploration Noise (Epsilon)
    epsilon: float = 0.20  # constant baseline exploration rate
    epsilon_end: float = 0.05  # terminal exploration rate for scheduled decay
    decay_epochs: int = 50  # decay duration if scheduled exploration is enabled
    # ~ [Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017]
    decay_type: str = "cosine"  # "cosine" or "linear" schedule
    use_epsilon_decay: bool = (
        False  # False for Phase 1 baseline; True for future decay runs
    )

    # Policy Penalization & Rewards
    beta: float = 0.0  # baseline policy penalization (0.0 = unpenalized)
    reward_scheme: str = "weighted_sum"  # "weighted_sum" (Phase 1 default), "pareto_crowding", "harrington"
    weights: tuple[float, float, float] = (
        0.50,
        0.25,
        0.25,
    )  # ROCS, SA, QED for WeightedSum

    # Reporting Desirability Thresholds (Used for Desired ratio metrics, NOT for reward gradients in WeightedSum)
    # ROCS 0.70 (~56% of empirical ~1.26 ceiling), SA 0.27 (equals raw SAscore 4.5 via SmoothClippedScore), QED 0.50
    thresholds: tuple[float, float, float] = (0.70, 0.27, 0.50)

    # 3D Shape & Chemistry Settings
    score_variant: str = (
        "tanimoto_combo"  # "tanimoto_combo" (default) or "tversky_subshape"
    )
    max_reference_conformers: int = 3  # Hawkins low-energy basin ensemble size
    max_conformers: int = 30  # query conformers per candidate (empirical knee)
    energy_window_kcal: float = 8.0  # physiological conformer energy filter window
    timeout_seconds: int = 60
    random_seed: int = 42

    # Hardware & Multiprocessing
    gpu: int | str = "auto"
    cpu_ratio: float | None = None
    conformer_threads: int = (
        0  # 0 = all cores for parallel conformer generation in RDKit
    )

    # Evaluation Arm Sample Size
    n_eval_samples: int = (
        1000  # molecules sampled before and after training for rigorous statistics
    )

    # Checkpoint Paths
    agent_path: Path | str | None = None
    vocab_path: Path | str | None = None

    def resolve_pretrained_paths(self) -> tuple[Path, Path]:
        """Resolve authoritative paths to Papyrus SMILES-RNN pretrained checkpoint and vocabulary.

        Returns:
            tuple[Path, Path]: (pkg_path, vocab_path)
        Raises:
            FileNotFoundError: If checkpoint or vocab files cannot be located on host.
        """
        pkg: Path | None = Path(self.agent_path) if self.agent_path else None
        voc: Path | None = Path(self.vocab_path) if self.vocab_path else None

        if pkg and voc and pkg.is_file() and voc.is_file():
            return pkg, voc

        candidates: list[Path] = [
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
            cand_pkg = cand / "Papyrus05.5_smiles_rnn_PT.pkg"
            cand_voc = cand / "Papyrus05.5_smiles_rnn_PT.vocab"
            if cand_pkg.is_file() and cand_voc.is_file():
                return cand_pkg, cand_voc

        raise FileNotFoundError(
            "Could not locate Papyrus05.5_smiles_rnn_PT (.pkg / .vocab). "
            f"Searched candidates: {[str(c) for c in candidates]}"
        )


@dataclass(frozen=True)
class RunPaths:
    """Canonical directory layout and target paths for a pipeline run."""

    root: Path
    models_dir: Path
    raw_data_dir: Path
    logs_dir: Path
    figures_dir: Path

    def create_directories(self) -> None:
        """Ensure all required target directories exist on disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Ensure directories exist immediately upon initialization."""
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
        models_dir = root / "models"
        raw_data_dir = root / "raw_data"
        logs_dir = raw_data_dir / "logs"
        figures_dir = root / "figures"
        return cls(
            root=root,
            models_dir=models_dir,
            raw_data_dir=raw_data_dir,
            logs_dir=logs_dir,
            figures_dir=figures_dir,
        )
