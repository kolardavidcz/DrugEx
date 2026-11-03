from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from _david.code.chemistry import Optimization
from _david.code.schema import std

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Canonical default target query molecule (Epigallocatechin gallate - EGCG)
DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical single source of truth for all LBDD hyperparameters and execution settings.

    Downstream constructors consume the config object directly
    (``PrepareMolecule(config=...)``, ``create_rocs_scorer(cfg=...)``,
    ``explorer.fit(epochs=cfg.epochs, ...)``). There are no kwargs-projection
    helpers: every field is read where it is used, so renaming or adding one is a
    single mechanical edit instead of four silent projections that can drift.
    """

    # Reinforcement Learning Exploration
    # DrugEx v1 (Liu et al., 2019, J. Cheminform. 11:35) converges around ~200 epochs
    # with 8 Adam gradient steps/epoch (lr=1e-3). Defaulting to 200 epochs with min_epochs=50 and patience=50.
    epochs: int = 200  # ^ [250, 200, 100]
    patience: int = 50  # Early-stopping patience (min 50 epochs before triggering)
    min_epochs: int = 50

    n_samples: int = 500  # Molecule samples generated per training epoch # ^ [500, 1000]
    val_samples: int = 100
    # Gold standard (2026-11-03): the final evaluation arm samples 5x more
    # molecules than the 1,000 used by every historical run, so the reported
    # property distributions and Pareto skies are not dominated by sampling
    # noise. Cost is linear in this number; it only affects Stage 3, never
    # training.
    n_eval_samples: int = 5000  # Statistical evaluation sample pool size # ^ [1000, 5000]
    batch_size: int = 64  # Mini-batch size per Adam policy-gradient step # ^ [64, 128]

    # Gold standard (2026-11-03): checkpoint selection uses the geometric mean
    # of the per-objective desirability ratios, which penalizes an arm that
    # collapses one objective to buy another. The arithmetic mean let a run
    # with one dead objective outrank a balanced one.
    criteria: str = std.AVG_GMEAN
    beta: float = 0.0  # Baseline policy penalization

    # Exploration Noise (Epsilon)
    epsilon: float = 0.05  # Prior-mutation exploration rate
    epsilon_end: float = 0.0  # Terminal exploration floor
    decay_start_epoch: int = 1  # Epoch where decay begins (stays at epsilon until this epoch)
    decay_epochs: int = 50  # Epoch where decay finishes reaching epsilon_end
    decay_type: str = "cosine"  # ^ ["cosine", "linear"]
    use_epsilon_decay: bool = False  # ^ [True, False]

    # Multi-Objective Desirability Thresholds (ROCS, SA[, QED])
    include_qed: bool = False  # Toggle QED inclusion in multi-objective RL environment
    thresholds: tuple[float, ...] | None = None  # Auto-calibrates to (0.35, 0.50) or (0.35, 0.50, 0.50)
    # Empirical baseline percentile (e.g. 75.0 = Q3) to auto-calibrate from Stage 1 prior distribution
    threshold_percentile: float | None = None  # ^ [75.0, None]

    # Scoring & Rewards (calibrated on the 121x121 = 14,641 drug-pair x 162-cell
    # selectivity grid, see _david/code/conformer/results.md).
    # Gold standard (2026-11-03): Surový (raw) ROCS TanimotoCombo on [0, 2],
    # ShapeTanimoto + ColorTanimoto, computed with use_colors=True. Raw means
    # `rocs_use_modifier=False` below and no ClippedScore re-scaling; the value
    # entering the reward is exactly the sum of the two similarities. Production
    # scoring is canonical ROCS TanimotoCombo only. The asymmetric
    # Tversky grid, its weights and its Gaussian re-weighters are a *measured
    # calibration*, not a runtime knob: they live in
    # _david/code/to_do/rocs_multimetric.py and are never reachable from training.
    # `score_variant` is kept as a single-choice field so historical submission
    # specs (which pass `--score-variant tanimoto_combo`) still replay verbatim.
    score_variant: str = std.VARIANT_TANIMOTO_COMBO
    reward_scheme: str = std.REWARD_PARETO  # ^ ["pareto_crowding", "harrington", "weighted_sum"]
    weights: tuple[float, ...] | None = None  # Auto-calibrates to (0.75, 0.25) or (0.50, 0.25, 0.25)

    # Scorer Modifier Bounds (ROCS raw/clipped & SA ClippedScore)
    rocs_use_modifier: bool = False  # False preserves raw TanimotoCombo [0, 2]; True applies ClippedScore [0, 1]
    rocs_clip_upper: float = 1.8  # Raw TanimotoCombo upper bound when rocs_use_modifier=True
    sa_score_lower: float = 5.0  # Raw SA >= 5.0 -> 0.0 reward
    sa_score_upper: float = 3.0  # Raw SA <= 3.0 -> flat 1.0 reward (stops artificial shrinkage)
    sa_smooth_lower: float = 5.0  # Backward-compatibility alias
    sa_smooth_upper: float = 3.0  # Backward-compatibility alias

    # 3D Conformer & Chemistry Settings (Calibrated on 128-drug panel: C=20 minimized beats C=75 unminimized)
    max_conformers: int = 20  # ^ [20, 30] (C=20 with MMFF94s optimization reaches >=0.95 ceiling at minimum latency)
    max_isomers: int = 4  # Independent cap; enumerate only unresolved stereochemistry.
    max_heavy_atoms: int | None = 45  # Pre-filter limit for candidate conformer generation (None disables)
    max_rotatable_bonds: int | None = 15  # Pre-filter limit for candidate rotatable bonds (None disables)

    # Note: ``optimization`` defines the force-field profile (method, dielectric, max_iterations)
    # shared by reference curation and candidate minimization; ``minimize_candidates`` toggles
    # whether RL candidates execute that minimization at runtime.
    minimize_candidates: bool = True  # ^ [True, False]
    optimization: Optimization | None = Optimization(
        "MMFF94s", 78.4
    )  # ^ [Optimization("MMFF94s", 78.4), Optimization("MMFF94s", 6.5)]
    reference_attempts: int = 100
    reference_rmsd_cutoff: float | None = 0.35  # Butina RMSD cutoff in Å # ^ [0.35, 0.50]
    reference_random_seed: int | None = None  # None uses disjoint DEFAULT_REFERENCE_SEED (104729).
    max_reference_conformers: int = 4  # Low-energy reference ensemble cap K # ^ [3, 4, 5]
    max_reference_isomers: int = 1
    reference_selection_mode: Literal["rmsd", "diverse"] = "diverse"  # ^ ["diverse", "rmsd"]
    energy_window_kcal: float = 8.0
    timeout_seconds: int = (
        #! I switch from  60 to 30 based on my past experience
        30  # Per-molecule ETKDG cap in seconds # ^ [60, 30, 20, 10, 0] (60s bounds pathological macrocycle hangs in RL; 0 = unlimited)
    )
    random_seed: int = 42
    training_seed: int | None = (
        None  # Optional Python/NumPy/Torch seed; recorded per run.
    )

    # Hardware & Multiprocessing (Authoritative 0.99 CPU allocation)
    gpu: int | str = "auto"
    cpu_ratio: float = 0.99  # 99% of available CPU cores by default
    conformer_threads: int = 0  # RDKit C++ EmbedMultipleConfs threads per molecule (0 = all cores, 1 = single-thread inside worker pool)

    # Checkpoint Paths
    agent_path: Path | str | None = None
    vocab_path: Path | str | None = None

    def __post_init__(self) -> None:
        valid_schemes = (std.REWARD_PARETO, std.REWARD_HARRINGTON, std.REWARD_WEIGHTED_SUM)
        if self.reward_scheme not in valid_schemes:
            raise ValueError(
                f"Invalid reward_scheme '{self.reward_scheme}'. Must be one of canonical schema standards: {valid_schemes}"
            )
        if self.score_variant != std.VARIANT_TANIMOTO_COMBO:
            raise ValueError(
                f"Invalid score_variant '{self.score_variant}'. Production training scores with "
                f"canonical ROCS TanimotoCombo ('{std.VARIANT_TANIMOTO_COMBO}') only: the 162-cell "
                "asymmetric Tversky calibration in _david/code/conformer/ is a measurement, not a "
                "runtime knob."
            )
        if self.thresholds is None:
            default_thresh = (0.35, 0.50, 0.50) if self.include_qed else (0.35, 0.50)
            object.__setattr__(self, "thresholds", default_thresh)
        if self.weights is None:
            default_weights = (0.50, 0.25, 0.25) if self.include_qed else (0.75, 0.25)
            object.__setattr__(self, "weights", default_weights)
        if self.sa_smooth_lower != 5.0 and self.sa_score_lower == 5.0:
            object.__setattr__(self, "sa_score_lower", self.sa_smooth_lower)
        if self.sa_smooth_upper != 3.0 and self.sa_score_upper == 3.0:
            object.__setattr__(self, "sa_score_upper", self.sa_smooth_upper)

    @property
    def minimize_method(self) -> Optimization | None:
        """Alias for ``optimization`` force-field profile."""
        return self.optimization

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
