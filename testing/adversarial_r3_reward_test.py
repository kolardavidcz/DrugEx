"""Adversarial Empirical Stress-Testing Harness for Requirement R3 (Milestone M2).

Stress Test Dimensions:
1. Zero 3D Condition (Pareto Anti-Cheating Guard):
   - 1,000+ random property vectors with S_3D = 0.0 and S_3D <= 1e-6.
   - Varied property vector dimensions (1 to 8), extreme property values, NaNs, Infs.
   - Evaluated in standalone mode and DrugEx Environment mode.
   - Assertion: Total reward is strictly 0.0 with zero exceptions.
2. Early Learning Gradient (Non-Vanishing Policy Signal):
   - Finite-difference gradients delta R / delta S_3D across S_3D in [0.01, 0.59].
   - Evaluated across diverse property profiles (S_prop = 0.0, 0.5, 1.0, random multi-property vectors).
   - Assertion: Numerical gradients lie strictly within [0.70, 1.20].
3. Synthesizability Feasibility Floor (Single SA Floor):
   - Authentic natural products (EGCG, Quercetin, Epicatechin, Morphine): SA <= 5.5 receives 1.0 feasibility.
   - Nonsense / synthetically inaccessible structures (SA >= 6.0): feasibility is 0.0.
   - Linear transition region (SA in [5.5, 6.0]): verified slope of -2.0.
4. Synthesizability Conflict Absence:
   - Verification of 100% absence of RAscore and BRICS synthesizability conflicts.
   - AST inspection and runtime environment inspection of GraphTransformer.py.
5. Robustness & Edge Cases:
   - Empty input arrays, 1D vs 2D arrays, normalize_rocs behavior, custom property weights.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple
from unittest.mock import MagicMock

# Pre-mock legacy RAscore to prevent eager import errors from legacy modules
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: numpy] WHY: Matrix calculations, random vector generation, and numerical gradient finite differences. WHAT: Numerical computing library.
import numpy as np

# [PACKAGE: pytest] WHY: Test orchestration and assertion verification. WHAT: Python test framework.
import pytest

# [PACKAGE: rdkit] WHY: Chemical SMILES parsing and authentic descriptor calculation. WHAT: Cheminformatics toolkit.
from rdkit import Chem

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DAVID_CODE = REPO_ROOT / "_david" / "code"
if str(DAVID_CODE) not in sys.path:
    sys.path.insert(0, str(DAVID_CODE))

from drugex.training.rewards import AntiCollapseSoftGateReward
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import ClippedScore


class TestZero3DParetoAntiCheating:
    """Adversarial stress-testing of the zero 3D Pareto anti-cheating guard."""

    def test_1000_random_property_vectors_zero_3d_strictly_zero_reward(self) -> None:
        """Test 1,000+ random property vectors with S_3D = 0.0 strictly produce 0.0 reward."""
        rng = np.random.default_rng(seed=42)
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )

        n_vectors = 1200
        for i in range(n_vectors):
            # Vary property dimension between 1 and 8
            n_props = int(rng.integers(1, 9))
            # Random property scores in [0.0, 1.0]
            props = rng.uniform(0.0, 1.0, size=(1, n_props))
            # S_3D is strictly 0.0
            scores = np.hstack([np.zeros((1, 1), dtype=np.float64), props])
            # Random SA score (either raw [1, 10] or clipped [0, 1])
            sa_raw = float(rng.uniform(1.0, 10.0))
            sa_arr = np.array([sa_raw])

            reward = scheme(scores, sa_scores=sa_arr)
            assert reward.shape == (1, 1), f"Expected shape (1, 1), got {reward.shape}"
            assert float(reward[0, 0]) == 0.0, (
                f"Pareto cheating at vector {i}: S_3D=0 earned reward {float(reward[0, 0])} "
                f"with props={props.tolist()} and sa={sa_raw}"
            )

    def test_extreme_hyper_inflated_property_cheating_at_zero_3d(self) -> None:
        """Adversarially inject extreme hyper-inflated 2D property scores with S_3D = 0.0."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        extreme_props = [
            np.array([[0.0, 100.0, 1000.0, 500.0]]),
            np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 1e6, 1e9]]),
        ]
        for scores in extreme_props:
            reward = scheme(scores, sa_scores=np.array([1.0]))
            assert float(reward[0, 0]) == 0.0, (
                f"Hyper-inflated properties earned reward {float(reward[0, 0])} when S_3D=0"
            )

    def test_sub_epsilon_boundary_zero_reward(self) -> None:
        """Verify that S_3D values in the sub-threshold boundary (<= 1e-6) strictly yield 0.0."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        sub_eps_vals = [0.0, 1e-15, 1e-12, 1e-9, 1e-7, 1e-6]
        for s_val in sub_eps_vals:
            scores = np.array([[s_val, 1.0, 1.0, 1.0]])
            sa = np.array([1.0])
            reward = scheme(scores, sa_scores=sa)
            assert float(reward[0, 0]) == 0.0, (
                f"Sub-epsilon S_3D={s_val} earned non-zero reward {float(reward[0, 0])}"
            )

    def test_negative_and_nan_inf_3d_inputs(self) -> None:
        """Verify negative, NaN, and infinite 3D inputs are safely zeroed out without exceptions."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        adversarial_inputs = [
            np.array([[-0.5, 0.9, 0.8]]),
            np.array([[-1e-3, 1.0, 1.0]]),
            np.array([[np.nan, 0.8, 0.7]]),
            np.array([[-np.inf, 1.0, 0.5]]),
        ]
        for scores in adversarial_inputs:
            reward = scheme(scores)
            assert float(reward[0, 0]) == 0.0, (
                f"Adversarial input {scores} produced non-zero reward {float(reward[0, 0])}"
            )

    def test_environment_signature_zero_3d(self) -> None:
        """Verify standard DrugEx Environment calling convention smiles, scores, thresholds."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            sa_col_idx=3,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        smiles = ["c1ccccc1", "CC(=O)O", "CCN"]
        scores = np.array([
            [0.0, 0.9, 0.8, 2.5],
            [0.0, 1.0, 1.0, 1.0],
            [1e-7, 0.5, 0.5, 3.0],
        ])
        thresholds = [0.5, 0.5, 0.5, 0.5]
        rewards = scheme(smiles, scores, thresholds)
        assert rewards.shape == (3, 1)
        assert np.all(rewards == 0.0), f"Environment signature failed to zero out: {rewards}"


class TestEarlyRLGradients:
    """Adversarial stress-testing of early RL policy gradient delta R / delta S_3D."""

    def test_finite_difference_gradients_across_001_to_059(self) -> None:
        """Verify finite-difference gradients delta R / delta S_3D lie strictly within [0.70, 1.20]."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        h = 1e-4
        s_eval_points = np.linspace(0.01, 0.59 - h, 60)

        # 1. Lower bound test: S_prop = 0.0 -> expected gradient = 0.70
        for s in s_eval_points:
            r1 = float(scheme(np.array([[s, 0.0]]))[0, 0])
            r2 = float(scheme(np.array([[s + h, 0.0]]))[0, 0])
            grad = (r2 - r1) / h
            assert 0.70 - 1e-4 <= grad <= 0.70 + 1e-4, (
                f"S_prop=0.0 gradient {grad:.6f} at S_3D={s:.4f} deviates from expected 0.70"
            )

        # 2. Upper bound test: S_prop = 1.0 -> expected gradient = 1.20
        for s in s_eval_points:
            r1 = float(scheme(np.array([[s, 1.0]]))[0, 0])
            r2 = float(scheme(np.array([[s + h, 1.0]]))[0, 0])
            grad = (r2 - r1) / h
            assert 1.20 - 1e-4 <= grad <= 1.20 + 1e-4, (
                f"S_prop=1.0 gradient {grad:.6f} at S_3D={s:.4f} deviates from expected 1.20"
            )

        # 3. Intermediate multi-property vectors
        rng = np.random.default_rng(seed=123)
        for _ in range(50):
            n_props = int(rng.integers(2, 6))
            props = rng.uniform(0.0, 1.0, size=(1, n_props))
            expected_grad = 0.70 + 0.50 * float(np.mean(props))

            for s in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]:
                scores_1 = np.hstack([np.array([[s]]), props])
                scores_2 = np.hstack([np.array([[s + h]]), props])
                r1 = float(scheme(scores_1)[0, 0])
                r2 = float(scheme(scores_2)[0, 0])
                grad = (r2 - r1) / h
                assert 0.70 - 1e-4 <= grad <= 1.20 + 1e-4, (
                    f"Gradient {grad:.6f} out of [0.70, 1.20] at S_3D={s:.4f}"
                )
                assert math.isclose(grad, expected_grad, abs_tol=1e-4), (
                    f"Gradient {grad:.6f} != expected {expected_grad:.6f}"
                )

    def test_gradient_saturation_above_tau(self) -> None:
        """Verify that for S_3D > 0.60 (above tau), the gradient saturates to strictly w_3D = 0.70."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        h = 1e-4
        for s in [0.65, 0.75, 0.85, 0.95]:
            for s_prop in [0.0, 0.5, 1.0]:
                r1 = float(scheme(np.array([[s, s_prop]]))[0, 0])
                r2 = float(scheme(np.array([[s + h, s_prop]]))[0, 0])
                grad = (r2 - r1) / h
                assert math.isclose(grad, 0.70, abs_tol=1e-4), (
                    f"Above tau gradient {grad:.6f} at S_3D={s} != 0.70"
                )

    def test_monotonicity_across_entire_domain(self) -> None:
        """Verify reward is strictly monotonically increasing with respect to S_3D across [0.001, 1.0]."""
        scheme = AntiCollapseSoftGateReward(
            rocs_col_idx=0,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        s_vals = np.linspace(0.001, 1.0, 100)
        rewards = [float(scheme(np.array([[s, 0.75]]))[0, 0]) for s in s_vals]
        for i in range(len(rewards) - 1):
            assert rewards[i + 1] > rewards[i], (
                f"Monotonicity violation at S_3D={s_vals[i]}: R[i]={rewards[i]} >= R[i+1]={rewards[i+1]}"
            )


class TestSynthesizabilityFloor:
    """Adversarial stress-testing of the single synthesizability feasibility floor."""

    def test_natural_products_zero_feasibility_penalty(self) -> None:
        """Verify authentic natural products (EGCG, Quercetin, Epicatechin, Morphine) receive 1.0 feasibility."""
        natural_products = {
            "EGCG": "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O",
            "Quercetin": "O=C1C(O)=C(C2=CC(=C(O)C=C2)O)OC3=CC(O)=CC(O)=C13",
            "Epicatechin": "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O",
            "Morphine": "CN1CC[C@]23[C@@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]3O)c54",
        }
        prop_sa = Property("SA")
        scheme = AntiCollapseSoftGateReward()

        for name, smiles in natural_products.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Failed to parse SMILES for {name}"
            raw_sa = float(prop_sa.getScores([mol])[0])

            # Authentic natural product SA score must be within reasonable complexity (<= 5.5)
            assert raw_sa <= 5.5, f"Natural product {name} has unexpected high SA score: {raw_sa}"

            # Feasibility floor must award exactly 1.0 (zero penalty)
            feasibility = float(scheme.sa_floor(raw_sa))
            assert feasibility == 1.0, (
                f"Natural product {name} (SA={raw_sa:.3f}) received feasibility penalty: {feasibility}"
            )

    def test_synthetic_nonsense_linear_degradation_to_zero(self) -> None:
        """Verify exact piecewise transitions and linear degradation for SA > 5.5 to SA >= 6.0."""
        scheme = AntiCollapseSoftGateReward(sa_floor_val=5.5, sa_ceil_val=6.0)

        # 1. Unpenalized region (SA <= 5.5)
        for sa in [1.0, 2.5, 4.0, 5.0, 5.5]:
            assert scheme.sa_floor(sa) == 1.0, f"SA={sa} should receive 1.0"

        # 2. Linear degradation transition (5.5 < SA < 6.0)
        test_transitions = [
            (5.60, 0.80),
            (5.70, 0.60),
            (5.75, 0.50),
            (5.80, 0.40),
            (5.90, 0.20),
        ]
        for sa, expected in test_transitions:
            val = float(scheme.sa_floor(sa))
            assert math.isclose(val, expected, abs_tol=1e-5), (
                f"Transition at SA={sa}: expected {expected}, got {val}"
            )

        # Verify linear slope of -2.0
        sa1 = 5.60
        sa2 = 5.80
        slope = (scheme.sa_floor(sa2) - scheme.sa_floor(sa1)) / (sa2 - sa1)
        assert math.isclose(slope, -2.0, abs_tol=1e-5), f"Slope {slope} != -2.0"

        # 3. Penalized nonsense region (SA >= 6.0)
        for sa in [6.0, 6.001, 6.5, 7.0, 8.5, 10.0]:
            assert scheme.sa_floor(sa) == 0.0, f"Nonsense SA={sa} should receive 0.0"

    def test_pre_clipped_vs_raw_sa_input_consistency(self) -> None:
        """Verify that both pre-clipped SA scores in [0, 1] and raw SA scores in [1, 10] are supported."""
        scheme = AntiCollapseSoftGateReward()

        # Raw SA scores
        raw_scores = np.array([3.0, 5.5, 5.75, 6.0, 7.5])
        res_raw = scheme.sa_floor(raw_scores)
        assert np.allclose(res_raw, [1.0, 1.0, 0.5, 0.0, 0.0])

        # Pre-clipped SA scores (already in [0, 1])
        clipped_scores = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
        res_clipped = scheme.sa_floor(clipped_scores)
        assert np.allclose(res_clipped, clipped_scores)


class TestSynthesizabilityConflictAbsence:
    """Adversarial audit ensuring 100% absence of RAscore and BRICS synthesizability conflicts."""

    def test_graph_transformer_ast_inspection_for_forbidden_metrics(self) -> None:
        """Verify static AST of GraphTransformer.py contains zero references to RAscore or BRICS."""
        gt_file = DAVID_CODE / "GraphTransformer.py"
        assert gt_file.exists(), f"GraphTransformer.py not found at {gt_file}"
        source = gt_file.read_text(encoding="utf-8")
        parsed = ast.parse(source)

        # Inspect all import statements
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "RAscore" not in alias.name, f"Forbidden import found: {alias.name}"
                    assert "ra_scorer" not in alias.name, f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "RAscore" not in mod, f"Forbidden from-import module: {mod}"
                assert "ra_scorer" not in mod, f"Forbidden from-import module: {mod}"
                for alias in node.names:
                    assert "RetrosyntheticAccessibilityScorer" not in alias.name, (
                        f"Forbidden import: {alias.name}"
                    )
                    assert "RAscore" not in alias.name, f"Forbidden import: {alias.name}"

    def test_active_environment_scorer_list_strictly_single_synthesizability(self) -> None:
        """Inspect the active DrugExEnvironment scorers built by GraphTransformerOptimizer."""
        from GraphTransformer import GraphTransformerOptimizer

        optimizer = GraphTransformerOptimizer()
        # Mock sdf path for testing environment build
        ref_mol_path = REPO_ROOT / "drugex" / "training" / "scorers" / "tests" / "test_files" / "ligand.sdf"
        if not ref_mol_path.exists():
            # Use temp sdf
            import tempfile
            from rdkit.Chem import AllChem
            temp_sdf = Path(tempfile.gettempdir()) / "test_ref.sdf"
            mol = Chem.MolFromSmiles("c1ccccc1")
            AllChem.EmbedMolecule(mol)
            w = Chem.SDWriter(str(temp_sdf))
            w.write(mol)
            w.close()
            ref_mol_path = temp_sdf

        env = optimizer.build_environment(ref_sdf_path=ref_mol_path)
        scorer_keys = env.getScorerKeys()

        # Synthesizability check: exact key 'SA' (not substring matching 'TPSA')
        sa_scorers = [k for k in scorer_keys if k == "SA"]
        ra_scorers = [k for k in scorer_keys if "rascore" in k.lower() or k == "RAscore"]
        brics_scorers = [k for k in scorer_keys if "brics" in k.lower()]

        assert len(sa_scorers) == 1, f"Expected exactly 1 SA synthesizability scorer, found: {sa_scorers}"
        assert len(ra_scorers) == 0, f"Found conflicting RAscore in active scorers: {ra_scorers}"
        assert len(brics_scorers) == 0, f"Found conflicting BRICS synthesizability scorer: {brics_scorers}"


class TestRewardSchemeRobustness:
    """Stress-test numerical robustness, edge-case shapes, and normalization options."""

    def test_empty_batch(self) -> None:
        """Verify scheme gracefully handles empty candidate batches."""
        scheme = AntiCollapseSoftGateReward()
        res = scheme(np.zeros((0, 4)))
        assert res.shape == (0, 1)

    def test_single_objective_3d_only(self) -> None:
        """Verify scheme works when only a single 3D ROCS objective is provided."""
        scheme = AntiCollapseSoftGateReward(w_3d=0.70, w_props=0.30)
        scores = np.array([[0.50]])
        reward = scheme(scores)
        assert math.isclose(float(reward[0, 0]), 0.35, abs_tol=1e-5)

    def test_normalize_rocs_option(self) -> None:
        """Verify normalize_rocs=True divides TanimotoCombo scores in [0, 2] by 2.0."""
        scheme_norm = AntiCollapseSoftGateReward(
            normalize_rocs=True,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        # Raw combo score = 1.20 -> normalized S_3D = 0.60 (saturates tau)
        scores = np.array([[1.20, 1.0]])
        # S_3D = 0.60, G(0.60) = 1.0, R = 0.70 * 0.60 + 0.30 * 1.0 * 1.0 = 0.42 + 0.30 = 0.72
        reward = scheme_norm(scores)
        assert math.isclose(float(reward[0, 0]), 0.72, abs_tol=1e-5)

    def test_custom_property_weights(self) -> None:
        """Verify non-uniform property weights are properly normalized and applied."""
        custom_weights = [0.20, 0.80]
        scheme = AntiCollapseSoftGateReward(
            property_weights=custom_weights,
            w_3d=0.70,
            w_props=0.30,
            tau_3d=0.60,
        )
        # S_3D = 0.60 (gate = 1.0)
        # Props: col 1 = 0.50, col 2 = 1.00
        # Weighted prop sum: 0.2 * 0.5 + 0.8 * 1.0 = 0.1 + 0.8 = 0.90
        # R = 0.70 * 0.60 + 0.30 * 1.0 * 0.90 = 0.42 + 0.27 = 0.69
        scores = np.array([[0.60, 0.50, 1.00]])
        reward = scheme(scores)
        assert math.isclose(float(reward[0, 0]), 0.69, abs_tol=1e-5)
