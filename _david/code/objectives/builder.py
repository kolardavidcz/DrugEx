"""Objective and Environment Builder for Ligand-Based De Novo Drug Design.

Constructs multi-objective DrugEx environments with semantic objective roles,
eliminating manual column-index coupling and providing standard thesis presets.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from drugex.training.environment import DrugExEnvironment
from drugex.training.interfaces import RewardScheme
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property

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
from _david.code.objectives.rewards import AntiCollapseSoftGateReward

SA_FLOOR: float = 5.50
SA_CEILING: float = 6.00
DEFAULT_ROCS_THRESHOLD: float = 0.25
DEFAULT_PROP_THRESHOLD: float = 0.25
DEFAULT_SA_THRESHOLD: float = 0.25


class ObjectivePreset(str, Enum):
    """Standard objective presets for multi-objective RL experiments."""

    PURE_3D_SA = "pure_3d_sa"
    GATED_ENVELOPE = "gated_envelope"


class ObjectiveBuilder:
    """Builds DrugExEnvironment with semantic objective roles and anti-cheating schemes.

    Prevents column index mismatch by programmatically assembling scorer lists,
    determining index mappings, and injecting them into reward schemes.
    """

    def __init__(
        self,
        preset: ObjectivePreset | str = ObjectivePreset.PURE_3D_SA,
        rocs_threshold: float = DEFAULT_ROCS_THRESHOLD,
        prop_threshold: float = DEFAULT_PROP_THRESHOLD,
        sa_threshold: float = DEFAULT_SA_THRESHOLD,
        w_3d: float = 0.70,
        w_props: float = 0.30,
        tau_3d: float = 0.60,
        sa_floor: float = SA_FLOOR,
        sa_ceiling: float = SA_CEILING,
        include_hb: bool = False,
    ) -> None:
        """Initialize ObjectiveBuilder with thresholds, weights, and preset."""
        if isinstance(preset, str):
            preset = ObjectivePreset(preset.lower())
        self.preset: ObjectivePreset = preset
        self.rocs_threshold: float = float(rocs_threshold)
        self.prop_threshold: float = float(prop_threshold)
        self.sa_threshold: float = float(sa_threshold)
        self.w_3d: float = float(w_3d)
        self.w_props: float = float(w_props)
        self.tau_3d: float = float(tau_3d)
        self.sa_floor: float = float(sa_floor)
        self.sa_ceiling: float = float(sa_ceiling)
        self.include_hb: bool = bool(include_hb)

    def build_environment(
        self,
        ref_sdf_path: Path | str,
        seed_smiles: str,
        reward_scheme: RewardScheme | None = None,
        custom_scorers: Sequence[Scorer] | None = None,
    ) -> DrugExEnvironment:
        """Construct multi-objective DrugExEnvironment according to active preset and parameters.

        Args:
            ref_sdf_path: Path to 3D conformer ensemble reference SDF.
            seed_smiles: SMILES string of query seed molecule.
            reward_scheme: Optional explicit reward scheme override (e.g. WeightedSum for ablation).
            custom_scorers: Optional extra custom scorers to append.

        Returns:
            Fully initialized DrugExEnvironment with consistent column mappings.
        """
        ref_path_str = str(ref_sdf_path)

        # 1. Primary 3D ROCS Scorer (Always Column 0)
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

        scorers: list[Scorer] = [rocs_scorer]
        thresholds: list[float] = [self.rocs_threshold]

        # 2. Dynamic Property Envelopes (Included only if preset == GATED_ENVELOPE)
        if self.preset == ObjectivePreset.GATED_ENVELOPE:
            dynamic_scorers = create_dynamic_property_scorers(seed_smiles, include_hb=self.include_hb)
            scorers.extend(dynamic_scorers)
            thresholds.extend([self.prop_threshold] * len(dynamic_scorers))

        # 3. Optional custom scorers
        if custom_scorers:
            scorers.extend(custom_scorers)
            thresholds.extend([self.prop_threshold] * len(custom_scorers))

        # 4. Synthesizability Feasibility Floor (Always the Final Column)
        sa_scorer = Property(
            prop="SA",
            modifier=ClippedScore(
                lower_x=self.sa_ceiling,
                upper_x=self.sa_floor,
                low_score=0.0,
                high_score=1.0,
            ),
        )
        scorers.append(sa_scorer)
        thresholds.append(self.sa_threshold)

        sa_col_idx = len(scorers) - 1

        # 5. Programmatic Reward Scheme Initialization
        if reward_scheme is not None:
            scheme = reward_scheme
        else:
            w_props = self.w_props if self.preset == ObjectivePreset.GATED_ENVELOPE else 0.0
            scheme = AntiCollapseSoftGateReward(
                rocs_col_idx=0,
                sa_col_idx=sa_col_idx,
                w_3d=self.w_3d,
                w_props=w_props,
                tau_3d=self.tau_3d,
                normalize_rocs=False,
                sa_floor_val=self.sa_floor,
                sa_ceil_val=self.sa_ceiling,
            )

        return DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=scheme,
        )
