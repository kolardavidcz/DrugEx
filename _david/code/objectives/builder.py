"""Objective and Environment Builder for Ligand-Based De Novo Drug Design.

Constructs multi-objective DrugEx environments with semantic objective roles,
eliminating manual column-index coupling and providing standard thesis presets
as well as a first-principles modular ablation framework (2D vs 3D, reward schemes,
SA gating vs objective vs disabled, and dynamic property envelopes).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from drugex.training.environment import DrugExEnvironment
from drugex.training.interfaces import RewardScheme
from drugex.training.rewards import ParetoCrowdingDistance, ParetoTanimotoDistance, WeightedSum
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.similarity import TverskyFingerprintSimilarity

from _david.code.objectives.conformer import (
    ROCS_COLOR_FF,
    ROCS_MAX_CONFORMERS,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    create_rocs_scorer,
)
from _david.code.objectives.envelopes import create_dynamic_property_scorers
from _david.code.objectives.rewards import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    GeometricMeanReward,
)

SA_FLOOR: float = 5.50
SA_CEILING: float = 6.00
DEFAULT_ROCS_THRESHOLD: float = 0.25
DEFAULT_2D_THRESHOLD: float = 0.40
DEFAULT_PROP_THRESHOLD: float = 0.25
DEFAULT_SA_THRESHOLD: float = 0.25


class ObjectivePreset(str, Enum):
    """Standard objective presets for multi-objective RL experiments."""

    PURE_3D_SA = "pure_3d_sa"
    GATED_ENVELOPE = "gated_envelope"


class SimilarityType(str, Enum):
    """Molecular similarity evaluation strategy."""

    ROCS_3D = "3d_rocs"
    TANIMOTO_2D = "2d_tanimoto"
    HYBRID = "hybrid"
    NONE = "none"


class RewardSchemeType(str, Enum):
    """Multi-objective reward aggregation and scalarization scheme."""

    ANTI_COLLAPSE_SOFT_GATE = "anti_collapse_soft_gate"
    WEIGHTED_SUM = "weighted_sum"
    PARETO_CROWDING = "pareto_crowding"
    PARETO_TANIMOTO = "pareto_tanimoto"
    CHEBYSHEV = "chebyshev"
    GEOMETRIC_MEAN = "geometric_mean"


class SAMode(str, Enum):
    """Synthesizability Accessibility (SA) constraint enforcement strategy."""

    GATE = "gate"            # Non-linear soft gating multiplier in reward calculation
    OBJECTIVE = "objective"  # Standard evaluated objective column in multi-objective vector
    OFF = "off"              # Completely disabled (useful for proving synthetic collapse)


@dataclass
class ObjectiveConfig:
    """Configuration class specifying modular objective settings for RL environments.

    Attributes:
        similarity_type: Similarity strategy ('3d_rocs', '2d_tanimoto', 'hybrid', 'none').
        fp_type: Fingerprint type for 2D similarity (default 'ECFP6').
        reward_type: Multi-objective scalarization / anti-cheating scheme.
        sa_mode: Synthesizability enforcement mode (GATE, OBJECTIVE, or OFF).
        enable_envelope: Whether to construct dynamic physicochemical property envelopes.
        include_hb: Whether to include hydrogen bond donor/acceptor envelopes.
        rocs_threshold: Desirability threshold for 3D ROCS shape/color similarity.
        sim_2d_threshold: Desirability threshold for 2D fingerprint Tanimoto similarity.
        prop_threshold: Desirability threshold for dynamic property envelopes.
        sa_threshold: Desirability threshold for synthetic accessibility score.
        w_similarity: Reward weight assigned to similarity objective.
        w_props: Reward weight assigned to property envelope objectives.
        tau_similarity: Soft gating steepness parameter for similarity.
        sa_floor: Hard boundary where synthesizability is considered acceptable.
        sa_ceiling: Hard boundary where synthesizability is completely unacceptable.
        chebyshev_rho: Augmentation factor for Chebyshev scalarization.
        geometric_eps: Epsilon offset for Geometric Mean reward scheme.
    """

    similarity_type: SimilarityType = SimilarityType.ROCS_3D
    fp_type: str = "ECFP6"
    reward_type: RewardSchemeType = RewardSchemeType.ANTI_COLLAPSE_SOFT_GATE
    sa_mode: SAMode = SAMode.GATE
    enable_envelope: bool = False
    include_hb: bool = False
    rocs_threshold: float = DEFAULT_ROCS_THRESHOLD
    sim_2d_threshold: float = DEFAULT_2D_THRESHOLD
    prop_threshold: float = DEFAULT_PROP_THRESHOLD
    sa_threshold: float = DEFAULT_SA_THRESHOLD
    w_similarity: float = 0.70
    w_props: float = 0.30
    tau_similarity: float = 0.60
    sa_floor: float = SA_FLOOR
    sa_ceiling: float = SA_CEILING
    chebyshev_rho: float = 0.05
    geometric_eps: float = 1e-4


class ObjectiveBuilder:
    """Builds DrugExEnvironment with semantic objective roles and anti-cheating schemes.

    Prevents column index mismatch by programmatically assembling scorer lists,
    determining index mappings, and injecting them into reward schemes.
    Supports both legacy presets and fully granular modular ablation configurations.
    """

    def __init__(
        self,
        config: ObjectiveConfig | None = None,
        preset: ObjectivePreset | str | None = None,
        similarity_type: SimilarityType | str | None = None,
        reward_type: RewardSchemeType | str | None = None,
        sa_mode: SAMode | str | None = None,
        enable_envelope: bool | None = None,
        rocs_threshold: float = DEFAULT_ROCS_THRESHOLD,
        sim_2d_threshold: float = DEFAULT_2D_THRESHOLD,
        prop_threshold: float = DEFAULT_PROP_THRESHOLD,
        sa_threshold: float = DEFAULT_SA_THRESHOLD,
        w_3d: float = 0.70,
        w_props: float = 0.30,
        tau_3d: float = 0.60,
        sa_floor: float = SA_FLOOR,
        sa_ceiling: float = SA_CEILING,
        include_hb: bool = False,
    ) -> None:
        """Initialize ObjectiveBuilder with either an ObjectiveConfig or granular parameters."""
        if config is not None:
            self.config: ObjectiveConfig = config
            return

        # Initialize defaults from parameters
        sim_type = SimilarityType.ROCS_3D
        rew_type = RewardSchemeType.ANTI_COLLAPSE_SOFT_GATE
        s_mode = SAMode.GATE
        env_enabled = False

        if preset is not None:
            if isinstance(preset, str):
                preset = ObjectivePreset(preset.lower())
            if preset == ObjectivePreset.GATED_ENVELOPE:
                env_enabled = True

        if similarity_type is not None:
            sim_type = SimilarityType(similarity_type.lower()) if isinstance(similarity_type, str) else similarity_type

        if reward_type is not None:
            rew_type = RewardSchemeType(reward_type.lower()) if isinstance(reward_type, str) else reward_type

        if sa_mode is not None:
            s_mode = SAMode(sa_mode.lower()) if isinstance(sa_mode, str) else sa_mode

        if enable_envelope is not None:
            env_enabled = bool(enable_envelope)

        self.config = ObjectiveConfig(
            similarity_type=sim_type,
            reward_type=rew_type,
            sa_mode=s_mode,
            enable_envelope=env_enabled,
            include_hb=bool(include_hb),
            rocs_threshold=float(rocs_threshold),
            sim_2d_threshold=float(sim_2d_threshold),
            prop_threshold=float(prop_threshold),
            sa_threshold=float(sa_threshold),
            w_similarity=float(w_3d),
            w_props=float(w_props),
            tau_similarity=float(tau_3d),
            sa_floor=float(sa_floor),
            sa_ceiling=float(sa_ceiling),
        )

    @property
    def preset(self) -> ObjectivePreset:
        """Backward-compatibility preset property."""
        if self.config.enable_envelope:
            return ObjectivePreset.GATED_ENVELOPE
        return ObjectivePreset.PURE_3D_SA

    @property
    def rocs_threshold(self) -> float:
        return self.config.rocs_threshold

    @property
    def prop_threshold(self) -> float:
        return self.config.prop_threshold

    @property
    def sa_threshold(self) -> float:
        return self.config.sa_threshold

    @property
    def sa_floor(self) -> float:
        return self.config.sa_floor

    @property
    def sa_ceiling(self) -> float:
        return self.config.sa_ceiling

    def build_environment(
        self,
        ref_sdf_path: Path | str,
        seed_smiles: str,
        reward_scheme: RewardScheme | None = None,
        custom_scorers: Sequence[Scorer] | None = None,
    ) -> DrugExEnvironment:
        """Construct multi-objective DrugExEnvironment according to active configuration.

        Args:
            ref_sdf_path: Path to 3D conformer ensemble reference SDF.
            seed_smiles: SMILES string of query seed molecule.
            reward_scheme: Optional explicit reward scheme override (e.g. for ablated runs).
            custom_scorers: Optional extra custom scorers to append.

        Returns:
            Fully initialized DrugExEnvironment with consistent semantic column mappings.
        """
        ref_path_str = str(ref_sdf_path)
        scorers: list[Scorer] = []
        thresholds: list[float] = []

        similarity_col_indices: list[int] = []
        primary_sim_idx: int = 0

        # 1. Similarity Scorers (3D ROCS, 2D Tanimoto, Hybrid, or None)
        if self.config.similarity_type in (SimilarityType.ROCS_3D, SimilarityType.HYBRID):
            rocs_scorer = create_rocs_scorer(
                references=ref_path_str,
                max_conformers=ROCS_MAX_CONFORMERS,
                max_heavy_atoms=ROCS_MAX_HEAVY_ATOMS,
                max_rotatable_bonds=ROCS_MAX_ROTATABLE_BONDS,
                timeout=ROCS_TIMEOUT,
                score_type=ROCS_SCORE_TYPE,
                color_ff=ROCS_COLOR_FF,
            )
            rocs_scorer.modifier = ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)
            similarity_col_indices.append(len(scorers))
            scorers.append(rocs_scorer)
            thresholds.append(self.config.rocs_threshold)

        if self.config.similarity_type in (SimilarityType.TANIMOTO_2D, SimilarityType.HYBRID):
            # [PACKAGE: drugex] WHY: Native 2D fingerprint similarity scorer evaluated against query seed. WHAT: Tversky with alpha=beta=1.0 is exact Tanimoto. HOW: TverskyFingerprintSimilarity.
            sim_2d_scorer = TverskyFingerprintSimilarity(
                smiles=seed_smiles,
                fp_type=self.config.fp_type,
                alpha=1.0,
                beta=1.0,
                modifier=ClippedScore(lower_x=0.0, upper_x=1.0, low_score=0.0, high_score=1.0),
            )
            similarity_col_indices.append(len(scorers))
            scorers.append(sim_2d_scorer)
            thresholds.append(self.config.sim_2d_threshold)

        if similarity_col_indices:
            primary_sim_idx = similarity_col_indices[0]

        # 2. Dynamic Property Envelopes
        prop_indices: list[int] = []
        if self.config.enable_envelope:
            dynamic_scorers = create_dynamic_property_scorers(seed_smiles, include_hb=self.config.include_hb)
            for s in dynamic_scorers:
                prop_indices.append(len(scorers))
                scorers.append(s)
                thresholds.append(self.config.prop_threshold)

        # 3. Optional custom scorers
        if custom_scorers:
            for s in custom_scorers:
                scorers.append(s)
                thresholds.append(self.config.prop_threshold)

        # 4. Synthesizability Accessibility (SA) Scorer
        sa_col_idx: int = -1
        if self.config.sa_mode in (SAMode.GATE, SAMode.OBJECTIVE):
            sa_scorer = Property(
                prop="SA",
                modifier=ClippedScore(
                    lower_x=self.config.sa_ceiling,
                    upper_x=self.config.sa_floor,
                    low_score=0.0,
                    high_score=1.0,
                ),
            )
            sa_col_idx = len(scorers)
            scorers.append(sa_scorer)
            thresholds.append(self.config.sa_threshold)

        # Ensure environment has at least one scorer
        if not scorers:
            raise ValueError("ObjectiveBuilder assembled an empty scorers list. At least one scorer must be enabled.")

        # 5. Programmatic Reward Scheme Assembly
        if reward_scheme is not None:
            scheme = reward_scheme
        else:
            rew_choice = self.config.reward_type
            if rew_choice == RewardSchemeType.ANTI_COLLAPSE_SOFT_GATE:
                w_props = self.config.w_props if (self.config.enable_envelope and prop_indices) else 0.0
                effective_sa_idx = sa_col_idx if (self.config.sa_mode == SAMode.GATE and sa_col_idx >= 0) else len(scorers) - 1
                effective_sa_floor = self.config.sa_floor if (self.config.sa_mode == SAMode.GATE) else 999.0
                effective_sa_ceil = self.config.sa_ceiling if (self.config.sa_mode == SAMode.GATE) else 1000.0

                scheme = AntiCollapseSoftGateReward(
                    rocs_col_idx=primary_sim_idx,
                    sa_col_idx=effective_sa_idx,
                    w_3d=self.config.w_similarity,
                    w_props=w_props,
                    tau_3d=self.config.tau_similarity,
                    normalize_rocs=False,
                    sa_floor_val=effective_sa_floor,
                    sa_ceil_val=effective_sa_ceil,
                )
            elif rew_choice == RewardSchemeType.WEIGHTED_SUM:
                scheme = WeightedSum()
            elif rew_choice == RewardSchemeType.PARETO_CROWDING:
                scheme = ParetoCrowdingDistance()
            elif rew_choice == RewardSchemeType.PARETO_TANIMOTO:
                scheme = ParetoTanimotoDistance()
            elif rew_choice == RewardSchemeType.CHEBYSHEV:
                scheme = AugmentedChebyshevReward(thresholds=thresholds, rho=self.config.chebyshev_rho)
            elif rew_choice == RewardSchemeType.GEOMETRIC_MEAN:
                scheme = GeometricMeanReward(thresholds=thresholds, eps=self.config.geometric_eps)
            else:
                scheme = WeightedSum()

        return DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=scheme,
        )
