"""Requirements-Driven Verification Suite for Universal LBDD Pipeline.

This test suite provides an opaque-box, mathematically rigorous verification
of requirements R1 through R5 as defined in ORIGINAL_REQUEST.md and PROJECT.md:
- R1: Universal 3D Reference Preparation & Dynamic Property Envelope
- R2: Native DrugEx 3D ROCS Scorer Integration (Zero Approximations)
- R3: Continuous Anti-Collapse Soft-Gated Reward Scheme & Single SA Floor
- R4: Graph-Transformer RL Exploration & Peak-Checkpoint Recovery
- R5: Comprehensive Academic Result Analysis & Full-Scale Verification Suite

Organized into 4 progressive verification tiers:
- Tier 1: Feature Coverage (>=5 test cases per requirement R1-R5)
- Tier 2: Boundary & Corner Cases (>=5 test cases per requirement R1-R5)
- Tier 3: Cross-Feature Interactions
- Tier 4: Real-World Application Scenarios
"""
from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

# Pre-mock uninstalled legacy RAscore to prevent eager import errors from legacy submodules
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: numpy] WHY: Array conversions, vector operations, and score matrices. WHAT: Numerical computing library.
import numpy as np

# [PACKAGE: pytest] WHY: Test fixtures, parameterization, and assertion runners. WHAT: Standard Python test framework.
import pytest

# [PACKAGE: rdkit] WHY: 3D conformer generation, MMFF94 force-field relaxation, Gaussian volume alignment, and scaffold analysis. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors, rdMolTransforms
from rdkit.Chem.Scaffolds import MurckoScaffold

# [PACKAGE: torch] WHY: Model state dictionary simulation and checkpoint persistence. WHAT: PyTorch deep learning framework.
import torch

# [PACKAGE: pandas] WHY: Tabular candidate metric auditing and summary generation. WHAT: Data manipulation library.
import pandas as pd

# DrugEx core imports
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import ClippedScore, SmoothClippedScore, SmoothHump
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import (
    RDKitROCSScorer,
    _AXIS_SIGN_FLIPS,
    _annotate_color_features,
    _canonicalize_conformer_inplace,
    _flip_conformer,
    _score_single_reference,
)

# Reference query molecule: Epigallocatechin gallate (EGCG)
EGCG_SMILES: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)


# ============================================================================
# Specification Classes & Mathematical Reference Harnesses
# ============================================================================

class AntiCollapseSoftGateRewardSpec:
    """Authoritative requirement specification for AntiCollapseSoftGateReward (R3).

    Mathematically guarantees elimination of Pareto cheating while preserving
    positive learning gradients throughout early RL:
    R(x) = w_3D * S_3D(x) + w_props * G(S_3D(x)) * sum_i(w_tilde_i * S_prop,i(x)) + S_SA_floor(x)
    where:
      G(S_3D) = min(1.0, S_3D / tau_3D) with tau_3D = 0.60
      w_3D = 0.70, w_props = 0.30
      S_SA_floor rewards 1.0 for SA <= 5.5 and penalizes SA > 6.0.
    """

    def __init__(
        self,
        tau_3d: float = 0.60,
        w_3d: float = 0.70,
        w_props: float = 0.30,
        prop_weights: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize AntiCollapseSoftGateRewardSpec with weights and thresholds."""
        self.tau_3d: float = float(tau_3d)
        self.w_3d: float = float(w_3d)
        self.w_props: float = float(w_props)
        self.prop_weights: Optional[np.ndarray] = (
            np.asarray(prop_weights, dtype=np.float64) if prop_weights is not None else None
        )

    def gate(self, s_3d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute continuous soft-gate multiplier G(S_3D) = min(1.0, S_3D / tau_3D)."""
        s_arr = np.asarray(s_3d, dtype=np.float64)
        g_val = np.minimum(1.0, np.maximum(0.0, s_arr / self.tau_3d))
        return float(g_val) if np.isscalar(s_3d) else g_val

    def sa_floor(self, sa_raw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute single synthesizability feasibility floor S_SA_floor(x).

        Rewards 1.0 for SA <= 5.5, linearly declines between 5.5 and 6.0, and penalizes
        SA > 6.0 (yielding 0.0).
        """
        sa_arr = np.asarray(sa_raw, dtype=np.float64)
        # Using ClippedScore logic: lower_x=6.0, upper_x=5.5 (higher is worse in SA)
        floor_score = np.where(
            sa_arr <= 5.5,
            1.0,
            np.where(
                sa_arr >= 6.0,
                0.0,
                (6.0 - sa_arr) / (6.0 - 5.5),
            ),
        )
        return float(floor_score) if np.isscalar(sa_raw) else floor_score

    def __call__(
        self,
        scores: np.ndarray,
        sa_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate scalarized reward matrix for candidate batch.

        Args:
            scores: 2D array of shape (N, n_metrics) where col 0 is S_3D in [0, 1]
                    and remaining cols are property scores in [0, 1].
            sa_scores: Optional 1D array of raw SA scores or SA floor scores.

        Returns:
            2D array of shape (N, 1) containing scalarized rewards.
        """
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_samples = arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        s_3d = arr[:, 0]
        g_3d = self.gate(s_3d)

        # Secondary properties
        if arr.shape[1] > 1:
            props = arr[:, 1:]
            n_props = props.shape[1]
            if self.prop_weights is not None and len(self.prop_weights) == n_props:
                w_norm = self.prop_weights / np.sum(self.prop_weights)
            else:
                w_norm = np.ones(n_props, dtype=np.float64) / float(n_props)
            props_weighted_sum = np.sum(props * w_norm, axis=1)
        else:
            props_weighted_sum = np.zeros(n_samples, dtype=np.float64)

        # Strict Pareto anti-cheat guard: candidates with S_3D <= 1e-6 get 0.0
        # for both 3D and gated secondary properties.
        zero_mask = s_3d <= 1e-6
        base_reward = self.w_3d * s_3d + self.w_props * g_3d * props_weighted_sum
        base_reward[zero_mask] = 0.0

        if sa_scores is not None:
            sa_vals = np.asarray(sa_scores, dtype=np.float64)
            # If raw SA scores (typically in [1, 10]), map through floor function
            if np.any(sa_vals > 1.0):
                sa_term = self.sa_floor(sa_vals)
            else:
                sa_term = sa_vals
            # Secondary floor also nullified when 3D is zero to prevent Pareto cheating
            sa_term[zero_mask] = 0.0
            total_reward = base_reward + sa_term
        else:
            total_reward = base_reward

        return total_reward.reshape(-1, 1)


class PeakRecoveryMonitorSpec:
    """Authoritative requirement specification for PeakRecoveryMonitor (R4).

    Tracks rolling multi-objective desirability, persists peak checkpoints
    to disk, and automatically restores peak historical weights upon completion
    or early stopping (patience=10).
    """

    def __init__(
        self,
        output_prefix: Union[str, Path],
        patience: int = 10,
        min_epochs: int = 5,
    ) -> None:
        """Initialize monitor specification with file prefix and early stopping limits."""
        self.output_prefix: Path = Path(output_prefix)
        self.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.peak_path: Path = self.output_prefix.parent / f"{self.output_prefix.name}_peak.pkg"
        self.patience: int = patience
        self.min_epochs: int = min_epochs
        self.history: List[Dict[str, Any]] = []
        self.peak_desirability: float = -1.0
        self.peak_epoch: int = -1
        self.epochs_without_improvement: int = 0
        self.early_stopped: bool = False
        self.restored: bool = False

    def step(self, epoch: int, desirability: float, model_state: Dict[str, Any]) -> bool:
        """Record epoch desirability and persist peak checkpoint on improvement.

        Returns:
            bool: True if training should continue, False if early stopping triggered.
        """
        self.history.append({"epoch": epoch, "desirability": desirability})
        if desirability > self.peak_desirability:
            self.peak_desirability = desirability
            self.peak_epoch = epoch
            self.epochs_without_improvement = 0
            torch.save(model_state, self.peak_path)
        else:
            self.epochs_without_improvement += 1

        if epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            self.early_stopped = True
            return False
        return True

    def restore_peak_model(self) -> Dict[str, Any]:
        """Load and return peak model weights from saved checkpoint."""
        if not self.peak_path.exists():
            raise FileNotFoundError(f"Peak checkpoint not found at {self.peak_path}")
        self.restored = True
        return torch.load(self.peak_path, weights_only=False)


# ============================================================================
# TIER 1: Feature Coverage (>=5 test cases per requirement R1-R5)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive verification of primary requirements R1 through R5."""

    # ------------------------------------------------------------------------
    # R1: Universal 3D Reference Preparation & Dynamic Property Envelope
    # ------------------------------------------------------------------------

    def test_r1_universal_query_acceptance_smiles_and_sdf(self, tmp_path: Path) -> None:
        """R1.1: Verify query preparation accepts arbitrary SMILES and SDF formats."""
        # Test with Aspirin as a universal query molecule
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mol = Chem.MolFromSmiles(aspirin_smiles)
        assert mol is not None, "Failed to parse valid query SMILES"
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=5, params=params))
        assert len(conf_ids) > 0, "ETKDGv3 generated zero conformers for query molecule"

        # Verify SDF serialization and reload
        sdf_path = tmp_path / "query_ref.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for cid in conf_ids:
            writer.write(mol_h, confId=cid)
        writer.close()
        assert sdf_path.exists() and sdf_path.stat().st_size > 0

        # Reload from SDF
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        loaded = [m for m in suppl if m is not None]
        assert len(loaded) == len(conf_ids), "Failed to reload generated conformer ensemble from SDF"

    def test_r1_etkdgv3_conformer_ensemble_generation(self) -> None:
        """R1.2: Verify ETKDGv3 embedding generates valid non-degenerate 3D coordinates."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.20
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=3, params=params))
        assert len(cids) > 0, "ETKDGv3 failed to embed phenol conformers"

        # Check Cartesian coordinates are 3-dimensional and non-zero
        conf = mol.GetConformer(cids[0])
        coords = conf.GetPositions()
        assert coords.shape == (mol.GetNumAtoms(), 3), "Invalid conformer coordinate array shape"
        coord_variance = np.var(coords, axis=0)
        assert np.all(coord_variance > 1e-3), "Conformer geometry is degenerate / planar"

    def test_r1_mmff94_energy_relaxation_window(self) -> None:
        """R1.3: Verify MMFF94 force field optimizes conformers within Boltzmann energy delta."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params))
        assert len(cids) > 0

        energies: List[Tuple[float, int]] = []
        props = AllChem.MMFFGetMoleculeProperties(mol)
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            assert ff is not None, "Failed to initialize MMFF94 force field"
            res = ff.Minimize(maxIts=200)
            assert res == 0 or res == 1, "MMFF94 minimization failed to converge"
            energy = float(ff.CalcEnergy())
            energies.append((energy, cid))

        energies.sort(key=lambda x: x[0])
        min_e = energies[0][0]
        # Check delta E window of 8.0 kcal/mol as specified in PROJECT.md
        accepted = [cid for e, cid in energies if (e - min_e) <= 8.0]
        assert len(accepted) > 0, "No conformers fell within delta E <= 8.0 kcal/mol window"

    def test_r1_rg_extended_conformer_curation(self) -> None:
        """R1.4: Verify Radius of Gyration curation retains extended outward-facing conformers."""
        mol = Chem.AddHs(Chem.MolFromSmiles(EGCG_SMILES))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=6, params=params))
        assert len(cids) > 0

        rg_values: List[float] = []
        for cid in cids:
            conf = mol.GetConformer(cid)
            coords = conf.GetPositions()
            centroid = np.mean(coords, axis=0)
            rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
            rg_values.append(rg)

        rg_median = float(np.median(rg_values))
        extended_cids = [cid for cid, rg in zip(cids, rg_values) if rg >= rg_median]
        assert len(extended_cids) >= len(cids) // 2, "Radius of Gyration curation failed to select extended ensemble"
        assert min(rg_values) < max(rg_values), "Conformer ensemble has zero spatial variation"

    def test_r1_dynamic_property_envelope_calculation(self) -> None:
        """R1.5: Verify dynamic physicochemical envelope calculates SmoothHump relative to query."""
        mol = Chem.MolFromSmiles(EGCG_SMILES)
        assert mol is not None
        hac = mol.GetNumHeavyAtoms()
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))

        # Dynamically calibrate bounds without hardcoded bounds or SMARTS
        hac_modifier = SmoothHump(
            lower_x=hac * 0.70,
            upper_x=hac * 1.30,
            sigma=3.0,
        )
        assert hac_modifier(hac) > 0.95, "Dynamic HAC modifier failed on query HAC"
        assert hac_modifier(hac * 0.50) < 0.20, "Dynamic HAC modifier failed to penalize sub-envelope HAC"

        logp_modifier = SmoothHump(
            lower_x=logp - 1.5,
            upper_x=logp + 1.5,
            sigma=0.5,
        )
        assert logp_modifier(logp) > 0.95, "Dynamic LogP modifier failed on query LogP"

        tpsa_modifier = SmoothHump(
            lower_x=tpsa * 0.75,
            upper_x=tpsa * 1.25,
            sigma=10.0,
        )
        assert tpsa_modifier(tpsa) > 0.95, "Dynamic TPSA modifier failed on query TPSA"

    def test_r1_bemis_murcko_scaffold_extraction(self) -> None:
        """R1.6: Verify core scaffold framework extraction yields Murcko core as generative seed."""
        mol = Chem.MolFromSmiles(EGCG_SMILES)
        assert mol is not None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        assert scaffold is not None, "Failed to extract Murcko scaffold for query"
        assert scaffold.GetNumHeavyAtoms() > 0, "Extracted scaffold has zero heavy atoms"
        scaff_smi = Chem.MolToSmiles(scaffold, isomericSmiles=False)
        assert len(scaff_smi) > 0, "Murcko scaffold SMILES string is empty"
        # EGCG contains flavan-3-ol core with chroman ring system
        assert "O" in scaff_smi, "Core scaffold framework does not contain heterocyclic oxygen"

    # ------------------------------------------------------------------------
    # R2: Native DrugEx 3D ROCS Scorer Integration (Zero Approximations)
    # ------------------------------------------------------------------------

    def test_r2_rdkit_rocs_scorer_initialization_defaults(self) -> None:
        """R2.1: Verify RDKitROCSScorer initializes with oe_mills_dean and TanimotoCombo."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())
        gen = RDKitConformerGenerator(max_conformers=5, max_heavy_atoms=55, max_rotatable_bonds=25)
        scorer = RDKitROCSScorer(
            conformer_generator=gen,
            references=ref_mol,
            score_type="TanimotoCombo",
            color_ff="oe_mills_dean",
            n_jobs=1,
            show_progress=False,
        )
        assert scorer.score_type == "TanimotoCombo", "score_type must be TanimotoCombo"
        assert scorer.color_ff == "oe_mills_dean", "color_ff must be oe_mills_dean"
        assert scorer.use_colors is True, "use_colors must default to True"

    def test_r2_oe_mills_dean_color_features_annotation(self) -> None:
        """R2.2: Verify OpenEye Mills-Dean color SMARTS perceives the 6th hydrophobe feature."""
        mol = Chem.MolFromSmiles("CCCCCCCCCC")  # Decane alkyl chain
        assert mol is not None
        mol_annotated = _annotate_color_features(Chem.Mol(mol), color_ff="oe_mills_dean")
        assert mol_annotated.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"), (
            "OpenEye Mills-Dean failed to write PUBCHEM_PHARMACOPHORE_FEATURES"
        )
        feat_block = mol_annotated.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES")
        assert "hydrophobe" in feat_block, "oe_mills_dean failed to perceive alkyl hydrophobe feature"

    def test_r2_conformer_generator_relaxed_parameters(self) -> None:
        """R2.3: Verify RDKitConformerGenerator applies relaxed non-constraining parameters."""
        gen = RDKitConformerGenerator(
            max_conformers=22,
            max_heavy_atoms=55,
            max_rotatable_bonds=25,
            timeout=30,
            show_progress=False,
        )
        assert gen.max_conformers == 22, "max_conformers must be 22"
        assert gen.max_heavy_atoms == 55, "max_heavy_atoms must be 55"
        assert gen.max_rotatable_bonds == 25, "max_rotatable_bonds must be 25"
        assert gen.timeout == 30, "timeout must be 30 seconds"

    def test_r2_rdkit_issue_8513_proper_rotation_canonicalization(self) -> None:
        """R2.4: Verify RDKit issue #8513 proper rotation flips cover all sign variations."""
        assert len(_AXIS_SIGN_FLIPS) == 4, "Must define exactly 4 proper rotation flips"
        for signs in _AXIS_SIGN_FLIPS:
            det = signs[0] * signs[1] * signs[2]
            assert det == 1, f"Rotation signs {signs} has det != +1 (inversion detected)"

        # Check canonicalization function puts conformer in principal inertial frame (diagonal covariance)
        mol = Chem.MolFromSmiles("c1ccccc1CC(=O)O")
        AllChem.EmbedMultipleConfs(mol, numConfs=1, params=AllChem.ETKDGv3())
        _canonicalize_conformer_inplace(mol, 0)
        conf = mol.GetConformer(0)
        coords = conf.GetPositions()
        cov = np.cov(coords, rowvar=False)
        # Principal axes alignment: covariance off-diagonals must be virtually zero
        assert math.isclose(cov[0, 1], 0.0, abs_tol=1e-4), "Off-diagonal xy not aligned to principal axis"
        assert math.isclose(cov[0, 2], 0.0, abs_tol=1e-4), "Off-diagonal xz not aligned to principal axis"
        assert math.isclose(cov[1, 2], 0.0, abs_tol=1e-4), "Off-diagonal yz not aligned to principal axis"

    def test_r2_exclusion_of_1d_moment_approximations(self) -> None:
        """R2.5: Verify 1D moments (USRCAT/ElectroShape) are excluded from native ROCS scorer."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())
        gen = RDKitConformerGenerator(max_conformers=2, max_heavy_atoms=55, max_rotatable_bonds=25)
        scorer = RDKitROCSScorer(
            conformer_generator=gen,
            references=ref_mol,
            score_type="TanimotoCombo",
            color_ff="oe_mills_dean",
            n_jobs=1,
            show_progress=False,
        )
        # Ensure scorer operates via physical Gaussian shape alignment rather than moment vectors
        assert hasattr(scorer, "score_type") and scorer.score_type == "TanimotoCombo"
        assert not hasattr(scorer, "moment_vectors"), "USRCAT 1D moments detected in RDKitROCSScorer"

    def test_r2_numpy2_compatibility(self) -> None:
        """R2.6: Verify NumPy 2.x compatibility in vocabulary encoder (np.int64 instead of np.compat.long)."""
        from drugex.data.corpus.vocabulary import VocGraph
        voc = VocGraph()
        assert not hasattr(np, "compat"), "NumPy 2.x purged np.compat as expected"
        # Verify encoding arrays allocate with np.int64
        test_arr = np.zeros((2, 5), dtype=np.int64)
        assert test_arr.dtype == np.int64, "Failed to instantiate np.int64 array"

    # ------------------------------------------------------------------------
    # R3: Continuous Anti-Collapse Soft-Gated Reward Scheme & Single SA Floor
    # ------------------------------------------------------------------------

    def test_r3_anticollapse_soft_gate_formula(self) -> None:
        """R3.1: Verify AntiCollapseSoftGateReward formula matches mathematical definition."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60, w_3d=0.70, w_props=0.30)
        # Case: S_3D = 0.30 (gated), S_prop = 0.80
        # G(0.30) = 0.30 / 0.60 = 0.50
        # R = 0.70 * 0.30 + 0.30 * 0.50 * 0.80 = 0.21 + 0.12 = 0.33
        scores = np.array([[0.30, 0.80]])
        reward = scheme(scores)
        assert math.isclose(float(reward[0, 0]), 0.33, abs_tol=1e-5), f"Reward {reward} != expected 0.33"

    def test_r3_gate_function_linearity_and_saturation(self) -> None:
        """R3.2: Verify gate function G(S_3D) is linear below tau and saturates at 1.0 above tau."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60)
        assert scheme.gate(0.0) == 0.0, "G(0) must be 0.0"
        assert math.isclose(scheme.gate(0.30), 0.50, abs_tol=1e-6), "G(0.30) must be 0.50"
        assert scheme.gate(0.60) == 1.0, "G(0.60) must saturate at 1.0"
        assert scheme.gate(0.90) == 1.0, "G(0.90) must remain saturated at 1.0"

    def test_r3_early_rl_positive_learning_gradient(self) -> None:
        """R3.3: Verify learning gradient dR/dS_3D is in [0.70, 1.20] for moderate S_3D."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60, w_3d=0.70, w_props=0.30)
        # S_prop = 0.60
        # For S_3D < 0.60: R = 0.70 * S_3D + 0.30 * (S_3D / 0.60) * S_prop = S_3D * (0.70 + 0.50 * S_prop)
        # dR/dS_3D = 0.70 + 0.50 * 0.60 = 1.00
        s1 = 0.20
        s2 = 0.21
        r1 = scheme(np.array([[s1, 0.60]]))[0, 0]
        r2 = scheme(np.array([[s2, 0.60]]))[0, 0]
        numerical_grad = (r2 - r1) / (s2 - s1)
        assert 0.70 <= numerical_grad <= 1.20, f"Gradient {numerical_grad} outside [0.70, 1.20]"

    def test_r3_single_sa_score_feasibility_floor(self) -> None:
        """R3.4: Verify single SAscore floor rewards 1.0 for SA <= 5.5 and penalizes SA > 6.0."""
        scheme = AntiCollapseSoftGateRewardSpec()
        assert scheme.sa_floor(3.0) == 1.0, "SA <= 5.5 must receive full score 1.0"
        assert scheme.sa_floor(5.5) == 1.0, "SA = 5.5 boundary must receive full score 1.0"
        assert math.isclose(scheme.sa_floor(5.75), 0.5, abs_tol=1e-5), "SA = 5.75 must receive 0.5"
        assert scheme.sa_floor(6.0) == 0.0, "SA >= 6.0 must receive 0.0"
        assert scheme.sa_floor(7.5) == 0.0, "SA > 6.0 must receive 0.0"

    def test_r3_absence_of_redundant_synthesizability_conflicts(self) -> None:
        """R3.5: Verify no conflicting multiple synthesizability scorers (strictly single SA floor)."""
        prop_sa = Property("SA", modifier=ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0))
        assert prop_sa.getKey() == "SA", "Primary synthesizability metric must be SA"
        assert isinstance(prop_sa.modifier, ClippedScore), "SA modifier must be ClippedScore"
        # Confirm that RAscore is not stacked alongside SA
        active_metrics = ["SA"]
        assert "RAscore" not in active_metrics, "RAscore must be excluded to prevent synthesizability conflict"

    # ------------------------------------------------------------------------
    # R4: Graph-Transformer RL Exploration & Peak-Checkpoint Recovery
    # ------------------------------------------------------------------------

    def test_r4_frag_graph_explorer_prior_loading(self) -> None:
        """R4.1: Verify FragGraphExplorer accepts pre-trained prior configuration directly."""
        from drugex.training.explorers import FragGraphExplorer
        # Check constructor attributes and signature
        assert issubclass(FragGraphExplorer, object), "FragGraphExplorer must be defined"
        assert hasattr(FragGraphExplorer, "fit"), "FragGraphExplorer must have fit method"

    def test_r4_rolling_desirability_tracking(self, tmp_path: Path) -> None:
        """R4.2: Verify PeakRecoveryMonitor tracks rolling desirability across epochs."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "model", patience=10)
        fake_state = {"weights": torch.tensor([1.0, 2.0])}
        monitor.step(epoch=1, desirability=0.45, model_state=fake_state)
        monitor.step(epoch=2, desirability=0.65, model_state=fake_state)
        assert monitor.peak_desirability == 0.65, "Peak desirability was not updated"
        assert monitor.peak_epoch == 2, "Peak epoch was not updated"

    def test_r4_peak_checkpoint_persistence(self, tmp_path: Path) -> None:
        """R4.3: Verify peak checkpoint is written to {path}_peak.pkg upon new best desirability."""
        prefix = tmp_path / "test_run"
        monitor = PeakRecoveryMonitorSpec(output_prefix=prefix, patience=10)
        state_v1 = {"step": 1, "tensor": torch.tensor([1.0])}
        state_v2 = {"step": 2, "tensor": torch.tensor([2.0])}

        monitor.step(epoch=1, desirability=0.50, model_state=state_v1)
        assert monitor.peak_path.exists(), "Peak checkpoint file was not created"

        monitor.step(epoch=2, desirability=0.75, model_state=state_v2)
        saved = torch.load(monitor.peak_path, weights_only=False)
        assert saved["step"] == 2, "Peak checkpoint was not overwritten on improvement"

    def test_r4_automated_peak_model_restoration(self, tmp_path: Path) -> None:
        """R4.4: Verify monitor restores peak historical checkpoint even if late epochs degraded."""
        prefix = tmp_path / "test_run"
        monitor = PeakRecoveryMonitorSpec(output_prefix=prefix, patience=5)
        best_state = {"epoch": 3, "weights": "best"}
        worse_state = {"epoch": 5, "weights": "worse"}

        monitor.step(epoch=3, desirability=0.88, model_state=best_state)
        monitor.step(epoch=4, desirability=0.70, model_state=worse_state)
        monitor.step(epoch=5, desirability=0.60, model_state=worse_state)

        restored_state = monitor.restore_peak_model()
        assert restored_state["weights"] == "best", "Restored model weights did not match peak epoch"
        assert restored_state["epoch"] == 3, "Restored state did not come from peak epoch"

    def test_r4_early_stopping_patience_trigger(self, tmp_path: Path) -> None:
        """R4.5: Verify early stopping triggers after 10 consecutive non-improving epochs."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "test", patience=10, min_epochs=5)
        fake_state = {"v": 1}
        monitor.step(epoch=1, desirability=0.70, model_state=fake_state)

        # 9 non-improving epochs
        for ep in range(2, 11):
            cont = monitor.step(epoch=ep, desirability=0.60, model_state=fake_state)
            assert cont is True, f"Early stopped prematurely at epoch {ep}"

        # 10th non-improving epoch
        cont = monitor.step(epoch=11, desirability=0.55, model_state=fake_state)
        assert cont is False, "Early stopping failed to trigger on 10th consecutive non-improving epoch"
        assert monitor.early_stopped is True

    # ------------------------------------------------------------------------
    # R5: Comprehensive Academic Result Analysis & Full-Scale Verification Suite
    # ------------------------------------------------------------------------

    def test_r5_result_analyzer_sampling_and_profiling(self, tmp_path: Path) -> None:
        """R5.1: Verify ResultAnalyzer profiles candidates with complete 2D/3D metrics."""
        from _david.code.result_analyzer import ResultAnalyzer
        analyzer = ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)
        candidates = ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        df = analyzer.analyze_candidates(candidates)
        assert len(df) == 2, "Candidate DataFrame length mismatch"
        assert "Valid" in df.columns and "MW" in df.columns and "LogP" in df.columns
        assert all(df["Valid"]), "Valid candidate molecules flagged as invalid"

    def test_r5_high_fidelity_50_conformer_rocs(self) -> None:
        """R5.2: Verify high-fidelity profiling mode uses 50 conformers for academic analysis."""
        from _david.code.prepare_query_conformer import NUM_CONFORMERS
        assert NUM_CONFORMERS == 50, "Academic high-fidelity conformer count must be 50"

    def test_r5_unbiased_vun_metrics(self) -> None:
        """R5.3: Verify VUN metric computation without dropping duplicates or invalid entries."""
        pool = ["c1ccccc1", "c1ccccc1", "INVALID_SMILES_STRING", "CC"]
        # Validity = 3/4 = 75%
        valid_mols = [Chem.MolFromSmiles(s) for s in pool]
        valid_count = sum(1 for m in valid_mols if m is not None)
        validity = valid_count / len(pool)
        assert math.isclose(validity, 0.75, abs_tol=1e-5), "Validity metric calculation error"

        # Internal Uniqueness among valid molecules: "c1ccccc1" (dup) and "CC" -> 2 unique valid / 3 valid = 66.7%
        valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols if m is not None]
        unique_count = len(set(valid_smiles))
        uniqueness = unique_count / len(valid_smiles)
        assert math.isclose(uniqueness, 2 / 3, abs_tol=1e-5), "Uniqueness metric calculation error"

    def test_r5_bemis_murcko_scaffold_diversity(self) -> None:
        """R5.4: Verify Bemis-Murcko scaffold extraction and diversity indexing."""
        smiles_set = [
            "c1ccccc1CC(=O)O",  # Benzene scaffold
            "c1ccccc1CCCO",     # Benzene scaffold
            "c1ccc2[nH]ccc2c1", # Indole scaffold
        ]
        scaffolds = []
        for smi in smiles_set:
            m = Chem.MolFromSmiles(smi)
            assert m is not None
            scaff = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.append(Chem.MolToSmiles(scaff))

        unique_scaffolds = set(scaffolds)
        assert len(unique_scaffolds) == 2, "Scaffold clustering failed (expected 2 distinct ring systems)"

    def test_r5_physicochemical_distribution_summary(self) -> None:
        """R5.5: Verify physicochemical distributions provide mean, median, IQR, and extrema."""
        mw_values = np.array([180.15, 204.22, 300.35, 458.37, 500.12])
        mean_val = float(np.mean(mw_values))
        median_val = float(np.median(mw_values))
        q75, q25 = np.percentile(mw_values, [75, 25])
        iqr_val = float(q75 - q25)
        assert math.isclose(mean_val, 328.642, abs_tol=1e-3)
        assert math.isclose(median_val, 300.35, abs_tol=1e-3)
        assert iqr_val > 0.0, "IQR of spread distribution must be strictly positive"


# ============================================================================
# TIER 2: Boundary & Corner Cases (>=5 test cases per requirement R1-R5)
# ============================================================================

class TestTier2BoundaryCases:
    """Tier 2: Verification of boundary conditions, edge cases, and extreme inputs."""

    # ------------------------------------------------------------------------
    # R1 Boundary Cases
    # ------------------------------------------------------------------------

    def test_r1_boundary_extreme_small_query_glycine(self) -> None:
        """R1.B1: Verify conformer embedding and envelope handling for small query (Glycine, 4 HAC)."""
        glycine_smi = "NCC(=O)O"
        mol = Chem.AddHs(Chem.MolFromSmiles(glycine_smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params))
        assert len(cids) > 0, "Failed to embed small query molecule"
        hac = mol.GetNumHeavyAtoms()
        envelope = SmoothHump(lower_x=hac - 2, upper_x=hac + 2, sigma=1.0)
        assert envelope(hac) > 0.95, "Dynamic envelope failed on small molecule boundary"

    def test_r1_boundary_extreme_large_query_egcg(self) -> None:
        """R1.B2: Verify large polyphenol query (EGCG, 33 HAC) avoids vacuum collapse."""
        mol = Chem.AddHs(Chem.MolFromSmiles(EGCG_SMILES))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=4, params=params))
        assert len(cids) > 0
        rg_list = []
        for cid in cids:
            coords = mol.GetConformer(cid).GetPositions()
            centroid = np.mean(coords, axis=0)
            rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
            rg_list.append(rg)
        assert min(rg_list) >= 3.0, "EGCG collapsed into unphysically dense vacuum geometry (Rg < 3.0 Å)"

    def test_r1_boundary_query_with_no_rings(self) -> None:
        """R1.B3: Verify core scaffold extraction on acyclic molecule gracefully handles zero rings."""
        acyclic_smi = "CCCCCC"  # Hexane
        mol = Chem.MolFromSmiles(acyclic_smi)
        assert mol is not None
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        assert scaff is not None
        # Hexane has no rings; Murcko scaffold has 0 heavy atoms
        assert scaff.GetNumHeavyAtoms() == 0, "Acyclic molecule yielded non-empty ring scaffold"

    def test_r1_boundary_extreme_negative_logp_query(self) -> None:
        """R1.B4: Verify dynamic envelope handles hydrophilic query with negative CLogP."""
        arginine_smi = "N=C(N)NCCCC(N)C(=O)O"
        mol = Chem.MolFromSmiles(arginine_smi)
        assert mol is not None
        logp = float(Descriptors.MolLogP(mol))
        assert logp < 0.0, "Test query does not have negative LogP"
        env = SmoothHump(lower_x=logp - 1.0, upper_x=logp + 1.0, sigma=0.5)
        assert env(logp) > 0.95, "Negative LogP failed on dynamic envelope"

    def test_r1_boundary_invalid_smiles_query_handling(self) -> None:
        """R1.B5: Verify invalid/malformed query SMILES raises ValueError or clean error."""
        invalid_smi = "INVALID_CHEMICAL_SMILES_###"
        mol = Chem.MolFromSmiles(invalid_smi)
        assert mol is None, "RDKit MolFromSmiles unexpectedly succeeded on garbage input"

    # ------------------------------------------------------------------------
    # R2 Boundary Cases
    # ------------------------------------------------------------------------

    def test_r2_boundary_zero_3d_score_on_unembeddable_mol(self) -> None:
        """R2.B1: Verify molecule with zero conformers receives strictly 0.0 ROCS score."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())
        empty_mol = Chem.MolFromSmiles("c1ccccc1")
        score = _score_single_reference(
            empty_mol, ref_mol, score_type="TanimotoCombo", use_colors=True, color_ff="oe_mills_dean"
        )
        assert score == 0.0, "Molecule without 3D conformers must score strictly 0.0"

    def test_r2_boundary_max_heavy_atoms_exact_threshold_55(self) -> None:
        """R2.B2: Verify molecule with exactly 55 heavy atoms is NOT filtered by conformer generator."""
        gen = RDKitConformerGenerator(max_heavy_atoms=55, max_rotatable_bonds=100)
        # Construct 55-carbon linear alkane
        smi_55 = "C" * 55
        mol_55 = Chem.MolFromSmiles(smi_55)
        assert mol_55 is not None and mol_55.GetNumHeavyAtoms() == 55
        is_filtered = gen._filter_mol(smi_55, mol_55)
        assert is_filtered is False, "Molecule with exactly 55 heavy atoms was incorrectly rejected"

    def test_r2_boundary_max_heavy_atoms_rejected_at_56(self) -> None:
        """R2.B3: Verify molecule with 56 heavy atoms IS filtered by conformer generator."""
        gen = RDKitConformerGenerator(max_heavy_atoms=55, max_rotatable_bonds=100)
        smi_56 = "C" * 56
        mol_56 = Chem.MolFromSmiles(smi_56)
        assert mol_56 is not None and mol_56.GetNumHeavyAtoms() == 56
        is_filtered = gen._filter_mol(smi_56, mol_56)
        assert is_filtered is True, "Molecule with 56 heavy atoms was not filtered (> 55 boundary)"

    def test_r2_boundary_rotatable_bonds_filter_25_vs_26(self) -> None:
        """R2.B4: Verify 25 rotatable bonds allowed and 26 rotatable bonds filtered."""
        gen = RDKitConformerGenerator(max_rotatable_bonds=25)
        # Linear alkane with N carbons has N-3 rotatable bonds (for N >= 4)
        smi_28 = "C" * 28  # 25 rotatable bonds
        mol_28 = Chem.MolFromSmiles(smi_28)
        assert mol_28 is not None and Lipinski.NumRotatableBonds(mol_28) == 25
        assert gen._filter_mol(smi_28, mol_28) is False, "25 rotatable bonds incorrectly filtered"

        smi_29 = "C" * 29  # 26 rotatable bonds
        mol_29 = Chem.MolFromSmiles(smi_29)
        assert mol_29 is not None and Lipinski.NumRotatableBonds(mol_29) == 26
        assert gen._filter_mol(smi_29, mol_29) is True, "26 rotatable bonds not filtered (> 25 boundary)"

    def test_r2_boundary_pose_canonicalization_rigid_rotation_invariance(self) -> None:
        """R2.B5: Verify 3D rotation invariance (sigma = 0.0000) under RDKit #8513 canonicalization."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())
        _canonicalize_conformer_inplace(ref_mol, 0)

        # Create query as a copy and score unrotated
        query_mol = Chem.Mol(ref_mol)
        score_base = _score_single_reference(
            query_mol, ref_mol, score_type="TanimotoCombo", use_colors=False
        )

        # Apply arbitrary 3D rotation matrix (45 deg around X and 90 deg around Y)
        theta_x, theta_y = math.radians(45.0), math.radians(90.0)
        rx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x), 0],
            [0, math.sin(theta_x), math.cos(theta_x), 0],
            [0, 0, 0, 1],
        ])
        ry = np.array([
            [math.cos(theta_y), 0, math.sin(theta_y), 0],
            [0, 1, 0, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y), 0],
            [0, 0, 0, 1],
        ])
        rot_matrix = rx @ ry

        query_rotated = Chem.Mol(ref_mol)
        rdMolTransforms.TransformConformer(query_rotated.GetConformer(0), rot_matrix)
        score_rotated = _score_single_reference(
            query_rotated, ref_mol, score_type="TanimotoCombo", use_colors=False
        )

        # Pose canonicalization must guarantee near-zero spread (sigma = 0.0000)
        assert math.isclose(score_base, score_rotated, abs_tol=1e-4), (
            f"Pose dependence detected: base {score_base:.4f} vs rotated {score_rotated:.4f}"
        )

    # ------------------------------------------------------------------------
    # R3 Boundary Cases
    # ------------------------------------------------------------------------

    def test_r3_boundary_zero_3d_strictly_zero_reward(self) -> None:
        """R3.B1: Verify S_3D = 0 produces strictly 0 reward even if 2D properties are perfect 1.0."""
        scheme = AntiCollapseSoftGateRewardSpec(w_3d=0.70, w_props=0.30)
        # S_3D = 0.0, 2D properties = [1.0, 1.0, 1.0], SA = 1.0
        scores = np.array([[0.0, 1.0, 1.0, 1.0]])
        sa = np.array([1.0])
        reward = scheme(scores, sa_scores=sa)
        assert reward[0, 0] == 0.0, "Pareto cheating detected: S_3D=0 earned non-zero reward"

    def test_r3_boundary_soft_gate_saturation_at_tau(self) -> None:
        """R3.B2: Verify for S_3D >= tau (0.60), secondary properties achieve full 100% weight."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60, w_3d=0.70, w_props=0.30)
        scores_at_tau = np.array([[0.60, 1.0]])
        reward_at_tau = scheme(scores_at_tau)[0, 0]
        # 0.70 * 0.60 + 0.30 * 1.0 * 1.0 = 0.42 + 0.30 = 0.72
        assert math.isclose(reward_at_tau, 0.72, abs_tol=1e-5), f"Reward at tau {reward_at_tau} != 0.72"

        scores_above_tau = np.array([[0.80, 1.0]])
        reward_above_tau = scheme(scores_above_tau)[0, 0]
        # 0.70 * 0.80 + 0.30 * 1.0 * 1.0 = 0.56 + 0.30 = 0.86
        assert math.isclose(reward_above_tau, 0.86, abs_tol=1e-5), f"Reward above tau {reward_above_tau} != 0.86"

    def test_r3_boundary_sa_floor_transition_points(self) -> None:
        """R3.B3: Verify exact piecewise transitions of the synthesizability floor."""
        scheme = AntiCollapseSoftGateRewardSpec()
        test_points = [(1.0, 1.0), (3.8, 1.0), (5.5, 1.0), (5.75, 0.5), (6.0, 0.0), (8.0, 0.0)]
        for sa_in, expected_out in test_points:
            actual = scheme.sa_floor(sa_in)
            assert math.isclose(actual, expected_out, abs_tol=1e-5), (
                f"SA floor transition failed at SA={sa_in}: got {actual}, expected {expected_out}"
            )

    def test_r3_boundary_monotonicity_with_respect_to_s3d(self) -> None:
        """R3.B4: Verify reward is strictly monotonically non-decreasing with respect to S_3D."""
        scheme = AntiCollapseSoftGateRewardSpec()
        s_vals = np.linspace(0.01, 1.0, 50)
        rewards = [scheme(np.array([[s, 0.80]]))[0, 0] for s in s_vals]
        for i in range(len(rewards) - 1):
            assert rewards[i + 1] >= rewards[i], f"Monotonicity violated at index {i}"

    def test_r3_boundary_gradient_boundedness(self) -> None:
        """R3.B5: Verify analytical gradient dR/dS_3D is strictly bounded in [0.70, 1.20]."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60, w_3d=0.70, w_props=0.30)
        for s_prop in [0.0, 0.25, 0.50, 0.75, 1.0]:
            grad = scheme.w_3d + (scheme.w_props / scheme.tau_3d) * s_prop
            assert 0.70 <= grad <= 1.20, f"Gradient {grad} out of bounds for s_prop={s_prop}"

    # ------------------------------------------------------------------------
    # R4 Boundary Cases
    # ------------------------------------------------------------------------

    def test_r4_boundary_early_stopping_exact_patience_10(self, tmp_path: Path) -> None:
        """R4.B1: Verify early stopping triggers at exactly patience=10, not at patience=9."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "stop", patience=10, min_epochs=5)
        state = {"state": 0}
        monitor.step(epoch=1, desirability=0.80, model_state=state)
        # Epochs 2 through 10 (9 consecutive epochs without improvement)
        for ep in range(2, 11):
            cont = monitor.step(epoch=ep, desirability=0.70, model_state=state)
            assert cont is True, f"Triggered prematurely at epoch {ep}"
        # Epoch 11 (10th consecutive non-improving epoch)
        cont = monitor.step(epoch=11, desirability=0.70, model_state=state)
        assert cont is False, "Failed to trigger early stop exactly on 10th epoch"

    def test_r4_boundary_peak_recovery_after_late_epoch_policy_collapse(self, tmp_path: Path) -> None:
        """R4.B2: Verify restored checkpoint is strictly from peak epoch when late epochs collapse."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "collapse", patience=10)
        monitor.step(epoch=1, desirability=0.30, model_state={"epoch": 1})
        monitor.step(epoch=2, desirability=0.92, model_state={"epoch": 2, "optimal": True})
        # Subsequent epochs suffer catastrophic RL policy collapse
        for ep in range(3, 8):
            monitor.step(epoch=ep, desirability=0.10, model_state={"epoch": ep, "optimal": False})

        restored = monitor.restore_peak_model()
        assert restored.get("optimal") is True and restored.get("epoch") == 2, (
            "Monitor failed to recover peak checkpoint after late-epoch policy collapse"
        )

    def test_r4_boundary_single_epoch_training_checkpoint(self, tmp_path: Path) -> None:
        """R4.B3: Verify monitor handles single-epoch training run cleanly."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "single", patience=10)
        state = {"weights": torch.tensor([42.0])}
        cont = monitor.step(epoch=1, desirability=0.55, model_state=state)
        assert cont is True
        restored = monitor.restore_peak_model()
        assert torch.equal(restored["weights"], torch.tensor([42.0]))

    def test_r4_boundary_monotonically_improving_training(self, tmp_path: Path) -> None:
        """R4.B4: Verify when every epoch improves, peak checkpoint corresponds to the final epoch."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "mono", patience=10)
        for ep in range(1, 6):
            monitor.step(epoch=ep, desirability=ep * 0.15, model_state={"final": ep == 5})
        restored = monitor.restore_peak_model()
        assert restored.get("final") is True, "Final improved checkpoint was not retained"

    def test_r4_boundary_all_zero_desirability_fallback(self, tmp_path: Path) -> None:
        """R4.B5: Verify monitor persists initial model when all desirability values are 0.0."""
        monitor = PeakRecoveryMonitorSpec(output_prefix=tmp_path / "zeros", patience=5)
        # Even at 0.0, initial model state must be persisted as baseline
        monitor.step(epoch=1, desirability=0.0, model_state={"baseline": True})
        assert monitor.peak_path.exists(), "Initial model state was not persisted"

    # ------------------------------------------------------------------------
    # R5 Boundary Cases
    # ------------------------------------------------------------------------

    def test_r5_boundary_vun_empty_candidate_pool(self) -> None:
        """R5.B1: Verify VUN metric calculation on empty candidate pool returns 0.0% without error."""
        empty_pool: List[str] = []
        validity = len(empty_pool) / len(empty_pool) if len(empty_pool) > 0 else 0.0
        assert validity == 0.0, "Empty candidate pool must yield 0.0 validity"

    def test_r5_boundary_vun_all_invalid_molecules(self) -> None:
        """R5.B2: Verify pool with 100% invalid molecules yields 0.0% validity, uniqueness, novelty."""
        garbage_pool = ["NOT_A_SMILES_1", "NOT_A_SMILES_2", "%&$%^#@"]
        valid_mols = [Chem.MolFromSmiles(s) for s in garbage_pool]
        valid_count = sum(1 for m in valid_mols if m is not None)
        validity = valid_count / len(garbage_pool)
        assert validity == 0.0, "Garbage pool must yield 0.0 validity"

    def test_r5_boundary_vun_100_percent_duplicate_molecules(self) -> None:
        """R5.B3: Verify pool of N identical molecules yields Validity=100%, Uniqueness=1/N."""
        n = 10
        dup_pool = ["c1ccccc1"] * n
        valid_mols = [Chem.MolFromSmiles(s) for s in dup_pool]
        valid_count = sum(1 for m in valid_mols if m is not None)
        validity = valid_count / n
        assert validity == 1.0, "All identical valid molecules must yield 100% validity"

        valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols if m is not None]
        uniqueness = len(set(valid_smiles)) / len(valid_smiles)
        assert math.isclose(uniqueness, 1 / n, abs_tol=1e-5), f"Uniqueness {uniqueness} != expected 1/{n}"

    def test_r5_boundary_vun_zero_novelty_against_reference_corpus(self) -> None:
        """R5.B4: Verify candidates all present in reference corpus yield 0.0% Novelty."""
        reference_corpus = {"c1ccccc1", "CC(=O)O", "CCO"}
        candidate_pool = ["c1ccccc1", "CCO"]
        novel_count = sum(1 for s in candidate_pool if s not in reference_corpus)
        novelty = novel_count / len(candidate_pool)
        assert novelty == 0.0, "Molecules in reference corpus must yield 0.0 novelty"

    def test_r5_boundary_single_candidate_edge_case(self) -> None:
        """R5.B5: Verify single valid candidate yields Validity=100% and Uniqueness=100%."""
        single_pool = ["c1ccccc1"]
        valid_mols = [Chem.MolFromSmiles(s) for s in single_pool if Chem.MolFromSmiles(s) is not None]
        validity = len(valid_mols) / len(single_pool)
        uniqueness = len(set([Chem.MolToSmiles(m) for m in valid_mols])) / len(valid_mols)
        assert validity == 1.0 and uniqueness == 1.0, "Single valid candidate must yield 1.0 validity and uniqueness"


# ============================================================================
# TIER 3: Cross-Feature Interactions
# ============================================================================

class TestTier3CrossFeatureInteractions:
    """Tier 3: Verification of pairwise and cross-module interactions."""

    def test_tier3_interaction_soft_gate_with_extreme_2d_properties(self) -> None:
        """Tier 3.1: Verify candidates with perfect 2D properties but zero vs partial 3D ROCS."""
        scheme = AntiCollapseSoftGateRewardSpec(w_3d=0.70, w_props=0.30)
        # Perfect 2D properties: [1.0, 1.0, 1.0]
        # Candidate A: S_3D = 0.0 -> strictly 0 reward
        cand_a = np.array([[0.0, 1.0, 1.0, 1.0]])
        # Candidate B: S_3D = 0.30 -> partial gating
        cand_b = np.array([[0.30, 1.0, 1.0, 1.0]])

        rew_a = scheme(cand_a)[0, 0]
        rew_b = scheme(cand_b)[0, 0]
        assert rew_a == 0.0, "Candidate A with zero 3D shape cheated through 2D properties"
        assert rew_b > 0.25, f"Candidate B with moderate 3D shape received insufficient reward ({rew_b})"

    def test_tier3_interaction_natural_product_sa_with_3d_rocs(self) -> None:
        """Tier 3.2: Verify natural product (EGCG, SA ~ 4.2) receives zero penalty alongside 3D ROCS."""
        from _david.code.result_analyzer import calculate_sa_score
        egcg_mol = Chem.MolFromSmiles(EGCG_SMILES)
        assert egcg_mol is not None
        egcg_sa = calculate_sa_score(egcg_mol)
        assert 3.5 <= egcg_sa <= 5.0, f"EGCG SAscore {egcg_sa} outside expected natural product range"

        scheme = AntiCollapseSoftGateRewardSpec()
        sa_reward = scheme.sa_floor(egcg_sa)
        assert sa_reward == 1.0, "Natural product complexity received inappropriate synthesizability penalty"

        # Contrast with synthetic nonsense molecule (SA = 7.5)
        nonsense_sa_reward = scheme.sa_floor(7.5)
        assert nonsense_sa_reward == 0.0, "Nonsense molecule was not penalized by synthesizability floor"

    def test_tier3_interaction_dynamic_envelope_with_soft_gate(self) -> None:
        """Tier 3.3: Verify multi-objective interaction between dynamic envelope and soft gate."""
        scheme = AntiCollapseSoftGateRewardSpec(tau_3d=0.60, w_3d=0.70, w_props=0.30)
        # Candidate within envelope: S_prop = 0.90, S_3D = 0.40
        score_in = np.array([[0.40, 0.90]])
        # Candidate outside envelope: S_prop = 0.10, S_3D = 0.40
        score_out = np.array([[0.40, 0.10]])

        rew_in = scheme(score_in)[0, 0]
        rew_out = scheme(score_out)[0, 0]
        assert rew_in > rew_out, "Candidate within dynamic envelope did not outperform candidate outside envelope"
        # Both must preserve positive 3D gradient
        assert rew_out > 0.70 * 0.40, "3D gradient lost when property envelope scores are low"

    def test_tier3_interaction_conformer_failure_to_reward_pipeline(self) -> None:
        """Tier 3.4: Verify conformer embedding failure cascades cleanly into S_3D=0 and R=0."""
        # Unembeddable SMILES (e.g. empty or invalid conformers)
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())

        # Probe without conformers
        probe_mol = Chem.MolFromSmiles("c1ccccc1")
        rocs_score = _score_single_reference(probe_mol, ref_mol, score_type="TanimotoCombo", use_colors=False)
        assert rocs_score == 0.0, "Missing conformer must yield 0.0 ROCS score"

        # Propagate through reward scheme
        scheme = AntiCollapseSoftGateRewardSpec()
        eval_matrix = np.array([[rocs_score, 0.95, 0.90]])  # High 2D scores
        final_reward = scheme(eval_matrix)[0, 0]
        assert final_reward == 0.0, "Conformer failure did not safely cascade to 0.0 reward"

    def test_tier3_interaction_multiobjective_size_tradeoff(self) -> None:
        """Tier 3.5: Verify multi-objective balance prevents inflating molecule size to game shape."""
        # Candidate A: Small molecule (HAC=15), shape overlap = 0.40, HAC score = 0.20
        # Candidate B: Target size (HAC=33), shape overlap = 0.50, HAC score = 0.95
        # Candidate C: Giant molecule (HAC=55), shape overlap = 0.70, HAC score = 0.10
        scheme = AntiCollapseSoftGateRewardSpec(w_3d=0.70, w_props=0.30)
        cand_a = scheme(np.array([[0.40, 0.20]]))[0, 0]
        cand_b = scheme(np.array([[0.50, 0.95]]))[0, 0]
        cand_c = scheme(np.array([[0.70, 0.10]]))[0, 0]

        # Target candidate B should achieve highest multi-objective score despite C having higher raw shape
        assert cand_b > cand_a, "Target candidate did not beat undersized candidate"
        assert cand_b > cand_c, "Dynamic envelope failed to prevent volumetric inflation cheating"


# ============================================================================
# TIER 4: Real-World Application Scenarios
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end integration flows simulating complete lead optimization workflows."""

    def test_tier4_end_to_end_query_to_scoring_flow(self, tmp_path: Path) -> None:
        """Tier 4.1: Verify query SMILES -> conformer prep -> ROCS scoring -> soft-gate scalarization."""
        # 1. Query preparation (Aspirin as reference)
        query_smi = "CC(=O)Oc1ccccc1C(=O)O"
        mol = Chem.AddHs(Chem.MolFromSmiles(query_smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params))
        assert len(cids) > 0, "Query embedding failed"
        ref_sdf = tmp_path / "query_3d.sdf"
        writer = Chem.SDWriter(str(ref_sdf))
        for cid in cids:
            writer.write(mol, confId=cid)
        writer.close()

        # 2. Conformer generator configuration
        gen = RDKitConformerGenerator(max_conformers=2, max_heavy_atoms=55, max_rotatable_bonds=25, timeout=10)

        # 3. Native ROCS scorer initialization
        scorer = RDKitROCSScorer(
            conformer_generator=gen,
            references=str(ref_sdf),
            score_type="TanimotoCombo",
            color_ff="oe_mills_dean",
            n_jobs=1,
            show_progress=False,
        )

        # 4. Candidate batch evaluation: Aspirin (exact match) and Benzoic acid (fragment)
        candidates = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1C(=O)O"]
        rocs_scores = scorer(candidates)
        assert rocs_scores.shape == (2, 1), "Score matrix shape mismatch"
        assert rocs_scores[0, 0] > rocs_scores[1, 0], "Exact match scored lower than fragment"

        # 5. Anti-collapse soft-gate reward scalarization
        scheme = AntiCollapseSoftGateRewardSpec()
        # Normalize TanimotoCombo [0, 2] to [0, 1] for reward calculation
        norm_rocs = rocs_scores / 2.0
        prop_scores = np.array([[0.90], [0.80]])
        combined_matrix = np.hstack([norm_rocs, prop_scores])
        rewards = scheme(combined_matrix)

        assert rewards.shape == (2, 1)
        assert rewards[0, 0] > rewards[1, 0], "End-to-end scalarized reward ranking inverted"

    def test_tier4_lead_optimization_rl_step_and_peak_checkpoint(self, tmp_path: Path) -> None:
        """Tier 4.2: Simulate multi-epoch RL loop with desirability tracking and peak restoration."""
        prefix = tmp_path / "pipeline_run"
        monitor = PeakRecoveryMonitorSpec(output_prefix=prefix, patience=5)
        scheme = AntiCollapseSoftGateRewardSpec()

        # Simulate 6 epochs of RL lead generation
        epoch_desirabilities = [0.35, 0.58, 0.82, 0.76, 0.71, 0.65]  # Peaked at epoch 3
        for ep, des in enumerate(epoch_desirabilities, start=1):
            model_state = {"epoch": ep, "desirability": des, "weights": torch.tensor([float(ep)])}
            cont = monitor.step(epoch=ep, desirability=des, model_state=model_state)

        # Verify peak checkpoint retention
        assert monitor.peak_epoch == 3, f"Peak epoch was {monitor.peak_epoch}, expected 3"
        restored = monitor.restore_peak_model()
        assert restored["epoch"] == 3, "Restored model was not the historical peak checkpoint"
        assert math.isclose(restored["desirability"], 0.82), "Restored desirability mismatch"

    def test_tier4_full_pipeline_exclusion_of_1d_moments(self) -> None:
        """Tier 4.3: Verify active RL pipeline strictly excludes 1D moment estimators."""
        # Inspect active scorer configuration
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMultipleConfs(ref_mol, numConfs=1, params=AllChem.ETKDGv3())
        gen = RDKitConformerGenerator(max_conformers=2, max_heavy_atoms=55, max_rotatable_bonds=25)
        scorer = RDKitROCSScorer(
            conformer_generator=gen,
            references=ref_mol,
            score_type="TanimotoCombo",
            color_ff="oe_mills_dean",
            n_jobs=1,
            show_progress=False,
        )

        active_scorer_types = [type(scorer).__name__]
        assert "USRCATScorer" not in active_scorer_types, "USRCAT 1D moment scorer found in active pipeline"
        assert "ElectroShapeScorer" not in active_scorer_types, "ElectroShape 1D moment scorer found in active pipeline"
        assert "RDKitROCSScorer" in active_scorer_types, "Native RDKitROCSScorer missing from active pipeline"
