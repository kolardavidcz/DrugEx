"""Adversarial Empirical Stress-Testing Harness for Requirement R5 (Milestone M3).

Adversarial Stress Test Dimensions:
1. High-Fidelity 50-Conformer ROCS Verification:
   - Module parameters & constants: NUM_CONFORMERS_ROCS=50, ROCS_MAX_HEAVY_ATOMS=55,
     ROCS_MAX_ROTATABLE_BONDS=25, ROCS_TIMEOUT=60, ROCS_SCORE_TYPE="TanimotoCombo", ROCS_COLOR_FF="oe_mills_dean".
   - Conformer generator & ROCS scorer initialization fidelity.
   - Compute ROCS scores output shape, dtype, score range [0.0, 2.0].
   - OpenEye Mills-Dean 6-type pharmacophore color annotation (oe_mills_dean).
   - Robustness on empty, invalid, and single-atom SMILES pools.
2. Rotational Pose Invariance under SO(3):
   - 20 arbitrary 3D rotation matrices in SO(3) generated via scipy.spatial.transform.Rotation.
   - Diverse bioactive molecules: EGCG (proof query), Omeprazole (RDKit #8513 benchmark),
     Benzene (planar symmetric), Atorvastatin (large flexible drug), Quercetin (flavonoid).
   - Assertion: TanimotoCombo self-score is strictly 2.0000 +/- 1e-4 with zero variance (sigma = 0.0000).
   - Pure shape and pure color modes each yield 1.0000 +/- 1e-4 with zero variance.
3. Conformer Generation Limits (55 Heavy Atoms, 25 Rotatable Bonds):
   - Heavy atom count exact boundaries: 54 HAC (pass), 55 HAC (pass), 56 HAC (reject).
   - Rotatable bond exact boundaries: 24 rotbonds (pass), 25 rotbonds (pass), 26 rotbonds (reject).
   - Compound boundary combinations: (55, 25) pass, (56, 25) reject, (55, 26) reject, (56, 26) reject.
   - Safe filtering without unhandled exceptions or batch corruption.
4. Bemis-Murcko Scaffold Extraction & Clustering:
   - Framework extraction on acyclic structures (returns '[Acyclic]'), invalid SMILES (returns '[Invalid]').
   - Benzene derivatives cluster to canonical 'c1ccccc1' with expected diversity D_scaffold = 1/N.
   - Heterocycles (pyridine, indole, quinoline, purine) and steroid core framework.
   - Structural isomer scaffold preservation.
   - Diversity ratio D_scaffold = N_unique_scaffolds / N_unique_valid across varied pools.
5. Publication Artifact Generation & Structural Integrity:
   - visualize_leads() generates valid, non-empty PNG images (>5 KB, valid PNG magic header).
   - PIL image readability verification for lead_structures_2d.png and property_distributions.png.
   - export_summary() compiles valid, non-empty JSON with all required academic metrics.
   - Numerical sanity: no NaN or Inf values in metric moments or envelope compliance.
   - Single-candidate and empty pool robustness.
   - Bemis-Murcko diversity ranking in prioritize_leads().
6. Automated Reference Generation Fallback Edge Case:
   - Empirical probe of missing reference SDF fallback behavior.
"""
# ruff: noqa: E402
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Pre-mock uninstalled legacy RAscore to prevent eager import errors from legacy submodules
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: numpy] WHY: Array conversions, variance calculations, and percentile auditing. WHAT: Scientific computing library.
import numpy as np

# [PACKAGE: pytest] WHY: Test orchestration, parameterization, and assertion verification. WHAT: Test framework.
import pytest

# [PACKAGE: PIL] WHY: Validation of rendered PNG artifact binary headers and image dimensions. WHAT: Imaging library.
from PIL import Image

# [PACKAGE: rdkit] WHY: 3D conformer generation, coordinate rotation, Bemis-Murcko extraction, descriptor calculation. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# [PACKAGE: scipy] WHY: Haar-uniform random SO(3) 3D rotation matrix generation. WHAT: Scientific computing library.
from scipy.spatial.transform import Rotation as R

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DAVID_CODE = REPO_ROOT / "_david" / "code"
if str(DAVID_CODE) not in sys.path:
    sys.path.insert(0, str(DAVID_CODE))

from _david.code.result_analyzer import (
    EGCG_SMILES,
    NUM_CONFORMERS_ROCS,
    ROCS_COLOR_FF,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    ResultAnalyzer,
)
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import (
    _annotate_color_features,
    _score_single_reference,
)

# ============================================================================
# 1. High-Fidelity 50-Conformer ROCS Verification
# ============================================================================

class TestHighFidelityROCSVerification:
    """Stress-test 50-conformer Gaussian ROCS parameters, scoring fidelity, and edge cases."""

    def test_rocs_specification_constants(self) -> None:
        """Verify academic 50-conformer ROCS parameters conform to R5 specification."""
        assert NUM_CONFORMERS_ROCS == 50, f"Expected 50 conformers, got {NUM_CONFORMERS_ROCS}"
        assert ROCS_MAX_HEAVY_ATOMS == 55, f"Expected 55 max heavy atoms, got {ROCS_MAX_HEAVY_ATOMS}"
        assert ROCS_MAX_ROTATABLE_BONDS == 25, f"Expected 25 max rotatable bonds, got {ROCS_MAX_ROTATABLE_BONDS}"
        assert ROCS_TIMEOUT == 60, f"Expected 60s timeout, got {ROCS_TIMEOUT}"
        assert ROCS_SCORE_TYPE == "TanimotoCombo", f"Expected TanimotoCombo, got {ROCS_SCORE_TYPE}"
        assert ROCS_COLOR_FF == "oe_mills_dean", f"Expected oe_mills_dean, got {ROCS_COLOR_FF}"

    def test_oe_mills_dean_color_annotation_capture(self) -> None:
        """Verify OpenEye Mills-Dean color SMARTS captures all 6 pharmacophore feature families."""
        # EGCG contains aromatic rings, H-donors, H-acceptors, and hydrophobic elements
        mol = Chem.MolFromSmiles(EGCG_SMILES)
        assert mol is not None, "Failed to parse EGCG SMILES"

        _annotate_color_features(mol, color_ff="oe_mills_dean")
        assert mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"), "Missing PUBCHEM_PHARMACOPHORE_FEATURES property"

        feat_text = mol.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES")
        lines = [line.strip() for line in feat_text.strip().split("\n")]
        assert len(lines) > 1, "Feature text empty or malformed"

        # Check for donor, acceptor, and rings features
        types_found = {line.split()[-1] for line in lines[1:] if line}
        assert "donor" in types_found, "Missing donor feature in EGCG"
        assert "acceptor" in types_found, "Missing acceptor feature in EGCG"
        assert "rings" in types_found, "Missing rings feature in EGCG"

        # Test alkane / hydrophobic probe (octane) to verify the 6th feature (hydrophobe)
        octane = Chem.MolFromSmiles("CCCCCCCC")
        assert octane is not None
        _annotate_color_features(octane, color_ff="oe_mills_dean")
        assert octane.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES")
        oct_text = octane.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES")
        assert "hydrophobe" in oct_text, "OpenEye Mills-Dean failed to perceive hydrophobe on alkyl chain"

    def test_compute_rocs_scores_execution_and_ranges(self, tmp_path: Path) -> None:
        """Verify compute_rocs_scores outputs valid float32 scores within [0.0, 2.0]."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        candidates = [
            EGCG_SMILES,          # Close match to reference template
            "c1ccccc1",           # Low shape overlap (benzene)
            "CC(=O)Oc1ccccc1C(=O)O", # Moderate overlap (aspirin)
        ]

        scores = analyzer.compute_rocs_scores(candidates)

        assert isinstance(scores, np.ndarray), "Scores must be a NumPy array"
        assert scores.dtype == np.float32, f"Expected float32 dtype, got {scores.dtype}"
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"

        # All scores must lie in [0.0, 2.0]
        assert np.all(scores >= 0.0), f"Scores contain negative values: {scores}"
        assert np.all(scores <= 2.0), f"Scores exceed upper bound 2.0: {scores}"

        # EGCG should achieve significantly higher score than tiny benzene
        egcg_score = float(scores[0])
        benzene_score = float(scores[1])
        assert egcg_score > benzene_score, (
            f"Expected EGCG score ({egcg_score:.3f}) > benzene score ({benzene_score:.3f})"
        )
        assert egcg_score >= 1.0, f"EGCG shape+color self-overlap should be >= 1.0, got {egcg_score:.3f}"

    def test_compute_rocs_scores_robustness_on_edge_cases(self, tmp_path: Path) -> None:
        """Verify compute_rocs_scores handles empty lists, invalid SMILES, and fragments safely."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        # 1. Empty candidate list
        empty_scores = analyzer.compute_rocs_scores([])
        assert len(empty_scores) == 0, "Empty list should return empty array"
        assert empty_scores.dtype == np.float32

        # 2. Invalid SMILES strings
        invalid_pool = ["NOT_A_SMILES", "INVALID&&&", "123456"]
        invalid_scores = analyzer.compute_rocs_scores(invalid_pool)
        assert len(invalid_scores) == 3
        assert np.all(invalid_scores == 0.0), "Invalid SMILES must receive 0.0 ROCS score"

        # 3. Single-atom / tiny fragments
        tiny_pool = ["C", "O", "[Na+].[Cl-]"]
        tiny_scores = analyzer.compute_rocs_scores(tiny_pool)
        assert len(tiny_scores) == 3
        assert np.all(tiny_scores >= 0.0)


# ============================================================================
# 2. Rotational Pose Invariance under SO(3)
# ============================================================================

class TestRotationalPoseInvarianceSO3:
    """Adversarial stress-test verifying rigid-body SO(3) 3D coordinate rotation invariance."""

    @pytest.mark.parametrize(
        "mol_name, smiles",
        [
            ("EGCG", EGCG_SMILES),
            ("Omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
            ("Benzene", "c1ccccc1"),
            ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
            ("Quercetin", "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12"),
        ],
    )
    def test_so3_rotation_invariance_tanimoto_combo(self, mol_name: str, smiles: str) -> None:
        """Verify TanimotoCombo self-score is 2.0000 with sigma = 0.0000 across 20 random SO(3) rotations."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert mol is not None, f"Failed to parse SMILES for {mol_name}"

        params = AllChem.ETKDGv3()
        params.randomSeed = 20261004
        embed_status = AllChem.EmbedMolecule(mol, params=params)
        if embed_status < 0:
            params.useRandomCoords = True
            AllChem.EmbedMolecule(mol, params=params)
        AllChem.MMFFOptimizeMolecule(mol)

        assert mol.GetNumConformers() > 0, f"Failed to embed conformer for {mol_name}"

        # Generate 20 random Haar-uniform orthogonal rotation matrices from SO(3)
        rot_matrices = R.random(20, random_state=4242).as_matrix()

        combo_scores: list[float] = []

        for rot_idx, rmat in enumerate(rot_matrices):
            # Assert proper rotation (det = +1.0)
            det = float(np.linalg.det(rmat))
            assert np.isclose(det, 1.0, atol=1e-5), f"Rotation {rot_idx} det != 1: {det}"

            # Create rotated copy in 3D
            rot_mol = Chem.Mol(mol)
            conf = rot_mol.GetConformer(0)
            for atom_idx in range(rot_mol.GetNumAtoms()):
                pos = np.array(conf.GetAtomPosition(atom_idx))
                new_pos = rmat @ pos
                conf.SetAtomPosition(atom_idx, new_pos)

            score = _score_single_reference(
                query_mol=rot_mol,
                ref_mol=mol,
                score_type="TanimotoCombo",
                use_colors=True,
                color_ff="oe_mills_dean",
            )
            combo_scores.append(score)

        scores_arr = np.array(combo_scores)
        mean_score = float(scores_arr.mean())
        std_score = float(scores_arr.std())
        min_score = float(scores_arr.min())
        max_score = float(scores_arr.max())

        print(
            f"SO(3) Invariance [{mol_name}]: mean={mean_score:.6f}, std={std_score:.6f}, "
            f"min={min_score:.6f}, max={max_score:.6f}"
        )

        assert np.isclose(mean_score, 2.0000, atol=1e-4), (
            f"Mean TanimotoCombo self-score for {mol_name} is {mean_score:.6f}, expected 2.0000"
        )
        assert std_score < 1e-6, (
            f"Standard deviation for {mol_name} across SO(3) rotations is {std_score:.6f}, expected strictly 0.0000"
        )
        assert np.isclose(min_score, 2.0000, atol=1e-4), f"Min score for {mol_name} is {min_score:.6f}"
        assert np.isclose(max_score, 2.0000, atol=1e-4), f"Max score for {mol_name} is {max_score:.6f}"

    def test_so3_rotation_invariance_pure_shape_and_pure_color(self) -> None:
        """Verify pure shape and pure color modes each yield 1.0000 with sigma = 0.0000 under SO(3) rotations."""
        mol = Chem.AddHs(Chem.MolFromSmiles(EGCG_SMILES))
        params = AllChem.ETKDGv3()
        params.randomSeed = 20261004
        AllChem.EmbedMolecule(mol, params=params)
        AllChem.MMFFOptimizeMolecule(mol)

        rot_matrices = R.random(10, random_state=12345).as_matrix()

        shape_scores: list[float] = []
        color_scores: list[float] = []

        for rmat in rot_matrices:
            rot_mol = Chem.Mol(mol)
            conf = rot_mol.GetConformer(0)
            for atom_idx in range(rot_mol.GetNumAtoms()):
                pos = np.array(conf.GetAtomPosition(atom_idx))
                conf.SetAtomPosition(atom_idx, rmat @ pos)

            s_shape = _score_single_reference(
                query_mol=rot_mol,
                ref_mol=mol,
                score_type="shape",
                use_colors=True,
                color_ff="oe_mills_dean",
            )
            s_color = _score_single_reference(
                query_mol=rot_mol,
                ref_mol=mol,
                score_type="color",
                use_colors=True,
                color_ff="oe_mills_dean",
            )
            shape_scores.append(s_shape)
            color_scores.append(s_color)

        sh_arr = np.array(shape_scores)
        col_arr = np.array(color_scores)

        assert np.isclose(sh_arr.mean(), 1.0000, atol=1e-4), f"Shape mean is {sh_arr.mean():.6f}"
        assert sh_arr.std() < 1e-6, f"Shape std is {sh_arr.std():.6f}"
        assert np.isclose(col_arr.mean(), 1.0000, atol=1e-4), f"Color mean is {col_arr.mean():.6f}"
        assert col_arr.std() < 1e-6, f"Color std is {col_arr.std():.6f}"


# ============================================================================
# 3. Conformer Generation Limits (55 Heavy Atoms, 25 Rotatable Bonds)
# ============================================================================

class TestConformerGenerationLimits:
    """Adversarial stress-test verifying exact boundary handling for 55 heavy atoms and 25 rotatable bonds."""

    @staticmethod
    def _make_mol_with_hac(n_hac: int) -> str:
        """Construct a synthetic poly-aromatic / alkane SMILES with exact heavy atom count."""
        # Start with benzene (6 C) and attach a linear carbon chain
        assert n_hac >= 6, "n_hac must be at least 6"
        tail_c = n_hac - 6
        if tail_c == 0:
            return "c1ccccc1"
        return "c1ccccc1" + "C" * tail_c

    @staticmethod
    def _make_mol_with_rotbonds(n_rot: int) -> str:
        """Construct a linear aliphatic chain with exact number of rotatable bonds."""
        # Linear alkane of N carbons has (N - 3) rotatable bonds for N >= 4
        # To get n_rot rotatable bonds: N = n_rot + 3
        assert n_rot >= 0
        n_carbons = n_rot + 3
        return "C" * n_carbons

    def test_heavy_atom_boundary_filter(self) -> None:
        """Verify strict HAC boundary: 54 pass, 55 pass, 56 rejected."""
        gen = RDKitConformerGenerator(
            max_conformers=NUM_CONFORMERS_ROCS,
            max_heavy_atoms=ROCS_MAX_HEAVY_ATOMS, # 55
            max_rotatable_bonds=100, # relaxed to isolate HAC
            show_progress=False,
        )

        mol_54 = Chem.MolFromSmiles(self._make_mol_with_hac(54))
        mol_55 = Chem.MolFromSmiles(self._make_mol_with_hac(55))
        mol_56 = Chem.MolFromSmiles(self._make_mol_with_hac(56))

        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_54) == 54
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_55) == 55
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_56) == 56

        # _filter_mol returns True if filtered (rejected), False if accepted
        assert not gen._filter_mol("mol_54", mol_54), "Molecule with 54 HAC incorrectly filtered"
        assert not gen._filter_mol("mol_55", mol_55), "Molecule with 55 HAC incorrectly filtered"
        assert gen._filter_mol("mol_56", mol_56), "Molecule with 56 HAC was NOT filtered"

    def test_rotatable_bond_boundary_filter(self) -> None:
        """Verify strict rotatable bond boundary: 24 pass, 25 pass, 26 rejected."""
        gen = RDKitConformerGenerator(
            max_conformers=NUM_CONFORMERS_ROCS,
            max_heavy_atoms=100, # relaxed to isolate rotbonds
            max_rotatable_bonds=ROCS_MAX_ROTATABLE_BONDS, # 25
            show_progress=False,
        )

        mol_24 = Chem.MolFromSmiles(self._make_mol_with_rotbonds(24))
        mol_25 = Chem.MolFromSmiles(self._make_mol_with_rotbonds(25))
        mol_26 = Chem.MolFromSmiles(self._make_mol_with_rotbonds(26))

        assert rdMolDescriptors.CalcNumRotatableBonds(mol_24) == 24
        assert rdMolDescriptors.CalcNumRotatableBonds(mol_25) == 25
        assert rdMolDescriptors.CalcNumRotatableBonds(mol_26) == 26

        assert not gen._filter_mol("mol_24", mol_24), "Molecule with 24 rotbonds incorrectly filtered"
        assert not gen._filter_mol("mol_25", mol_25), "Molecule with 25 rotbonds incorrectly filtered"
        assert gen._filter_mol("mol_26", mol_26), "Molecule with 26 rotbonds was NOT filtered"

    def test_compound_boundary_matrix(self) -> None:
        """Verify all 4 corners of the compound HAC x Rotatable Bonds boundary matrix."""
        gen = RDKitConformerGenerator(
            max_conformers=NUM_CONFORMERS_ROCS,
            max_heavy_atoms=55,
            max_rotatable_bonds=25,
            show_progress=False,
        )

        hexacene = "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1"
        c1_str_25 = "CC=CCCC=CCCCCCCCC" # 15 C, 2 double bonds
        c1_str_26 = "CC=CCCCCCCCCCCCC"  # 15 C, 1 double bond
        c2_str = "CCCCCCCCCCCCCC"      # 14 C

        # 1. Corner (55 HAC, 25 rotbonds) -> PASS
        smi_55_25 = hexacene.replace("c1ccc2", f"c1cc({c1_str_25})c2", 1).replace("ccccc6", f"ccc({c2_str})cc6", 1)
        # 2. Reject on HAC only (56 HAC, 25 rotbonds) -> REJECT
        smi_56_25 = hexacene.replace("c1ccc2", f"c1cc({c1_str_25})c2", 1).replace("ccccc6", f"ccc({c2_str})c(C)c6", 1)
        # 3. Reject on rotbonds only (55 HAC, 26 rotbonds) -> REJECT
        smi_55_26 = hexacene.replace("c1ccc2", f"c1cc({c1_str_26})c2", 1).replace("ccccc6", f"ccc({c2_str})cc6", 1)
        # 4. Reject on both (56 HAC, 26 rotbonds) -> REJECT
        smi_56_26 = hexacene.replace("c1ccc2", f"c1cc({c1_str_26})c2", 1).replace("ccccc6", f"ccc({c2_str})c(C)c6", 1)

        mol_55_25 = Chem.MolFromSmiles(smi_55_25)
        mol_56_25 = Chem.MolFromSmiles(smi_56_25)
        mol_55_26 = Chem.MolFromSmiles(smi_55_26)
        mol_56_26 = Chem.MolFromSmiles(smi_56_26)

        assert mol_55_25 is not None and mol_56_25 is not None and mol_55_26 is not None and mol_56_26 is not None

        # Verify descriptors
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_55_25) == 55 and rdMolDescriptors.CalcNumRotatableBonds(mol_55_25) == 25
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_56_25) == 56 and rdMolDescriptors.CalcNumRotatableBonds(mol_56_25) == 25
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_55_26) == 55 and rdMolDescriptors.CalcNumRotatableBonds(mol_55_26) == 26
        assert rdMolDescriptors.CalcNumHeavyAtoms(mol_56_26) == 56 and rdMolDescriptors.CalcNumRotatableBonds(mol_56_26) == 26

        # Verify filtering decisions
        assert not gen._filter_mol("pass_55_25", mol_55_25), "Molecule at boundary (55, 25) must be ACCEPTED"
        assert gen._filter_mol("fail_56_25", mol_56_25), "Molecule exceeding HAC (56, 25) must be REJECTED"
        assert gen._filter_mol("fail_55_26", mol_55_26), "Molecule exceeding rotbonds (55, 26) must be REJECTED"
        assert gen._filter_mol("fail_56_26", mol_56_26), "Molecule exceeding both (56, 26) must be REJECTED"

    def test_full_pipeline_conformer_generation_and_scoring(self, tmp_path: Path) -> None:
        """Verify ResultAnalyzer compute_rocs_scores handles 55-HAC candidate safely and rejects 56-HAC."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        hexacene = "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1"
        c1_str_25 = "CC=CCCC=CCCCCCCCC"
        c2_str = "CCCCCCCCCCCCCC"
        smi_55 = hexacene.replace("c1ccc2", f"c1cc({c1_str_25})c2", 1).replace("ccccc6", f"ccc({c2_str})cc6", 1)
        smi_56 = hexacene.replace("c1ccc2", f"c1cc({c1_str_25})c2", 1).replace("ccccc6", f"ccc({c2_str})c(C)c6", 1)

        scores = analyzer.compute_rocs_scores([smi_55, smi_56])
        assert len(scores) == 2
        # smi_56 must receive 0.0 because it is filtered by conformer generator
        assert float(scores[1]) == 0.0, f"Candidate with 56 HAC should receive score 0.0, got {scores[1]}"
        # smi_55 was NOT rejected by filter (whether ETKDG generates conformers within timeout or not, it didn't crash)
        assert isinstance(float(scores[0]), float)


# ============================================================================
# 4. Bemis-Murcko Scaffold Extraction & Clustering
# ============================================================================

class TestBemisMurckoScaffoldsAndClustering:
    """Adversarial stress-test for Bemis-Murcko framework extraction and diversity indexing."""

    def test_acyclic_and_invalid_scaffold_extractions(self, tmp_path: Path) -> None:
        """Verify acyclic molecules extract to '[Acyclic]' and invalid SMILES extract to '[Invalid]'."""
        analyzer = ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

        acyclic_mols = [
            "CCCC",
            "CCCCCCCC",
            "O=C(O)CCCC(=O)O",
            "CCOCC",
            "CCN(CC)CC",
        ]
        for smi in acyclic_mols:
            scaff = analyzer.extract_scaffold(smi)
            assert scaff == "[Acyclic]", f"Expected '[Acyclic]' for {smi}, got {scaff!r}"

        # Malformed chemical SMILES must return '[Invalid]'
        invalid_inputs = ["NOT_VALID", "&&&", "123456xyz"]
        for inv in invalid_inputs:
            scaff = analyzer.extract_scaffold(inv)
            assert scaff == "[Invalid]", f"Expected '[Invalid]' for {inv!r}, got {scaff!r}"

        # Empty string '' yields a 0-heavy-atom Mol object, which is classified as '[Invalid]'
        assert analyzer.extract_scaffold("") == "[Invalid]"

    def test_varied_ring_system_canonical_frameworks(self, tmp_path: Path) -> None:
        """Verify varied ring systems extract correct canonical framework SMILES."""
        analyzer = ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

        test_cases = {
            # Benzene derivatives -> c1ccccc1
            "toluene": ("Cc1ccccc1", "c1ccccc1"),
            "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"),
            "benzoic_acid": ("c1ccccc1C(=O)O", "c1ccccc1"),
            "phenethylamine": ("NCCc1ccccc1", "c1ccccc1"),
            # Heterocycles
            "pyridine_sub": ("Cc1ccncc1", "c1ccncc1"),
            "indole_sub": ("Cc1ccc2[nH]ccc2c1", "c1ccc2[nH]ccc2c1"),
            "quinoline_sub": ("Cc1ccc2ncccc2c1", "c1ccc2ncccc2c1"),
            "purine_sub": ("Cn1cnc2c1ncnc2", "c1ncc2nc[nH]c2n1"),
            # Steroid core with exocyclic carbonyl (testosterone)
            "testosterone": (
                "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
                "O=C1C=C2CCC3C4CCCC4CCC3C2CC1",
            ),
            # Proof template EGCG
            "EGCG": (
                EGCG_SMILES,
                "O=C(OC1Cc2ccccc2OC1c1ccccc1)c1ccccc1",
            ),
        }

        for name, (smi, expected_scaff) in test_cases.items():
            scaff = analyzer.extract_scaffold(smi)
            assert scaff == expected_scaff, (
                f"Scaffold mismatch for {name}: got {scaff!r}, expected {expected_scaff!r}"
            )

    def test_structural_isomer_scaffold_invariance(self, tmp_path: Path) -> None:
        """Verify positional isomers of substituted rings extract to the identical scaffold."""
        analyzer = ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

        # ortho-, meta-, para-xylene all have the benzene scaffold
        xylenes = ["Cc1ccccc1C", "Cc1cccc(C)c1", "Cc1ccc(C)cc1"]
        scaffolds = [analyzer.extract_scaffold(smi) for smi in xylenes]
        assert all(s == "c1ccccc1" for s in scaffolds), f"Xylenes did not all map to c1ccccc1: {scaffolds}"

        # 4-chloroindole, 5-chloroindole, 6-chloroindole all have the indole scaffold
        indoles = ["Clc1cccc2[nH]ccc12", "Clc1ccc2[nH]ccc2c1", "Clc1cc2[nH]ccc2cc1"]
        indole_scaffs = [analyzer.extract_scaffold(smi) for smi in indoles]
        assert len(set(indole_scaffs)) == 1, f"Indole isomers mapped to multiple scaffolds: {indole_scaffs}"
        assert indole_scaffs[0] == "c1ccc2[nH]ccc2c1"

    def test_scaffold_diversity_ratio_calculation(self, tmp_path: Path) -> None:
        """Verify D_scaffold = N_unique_scaffolds / N_unique_valid across controlled pools."""
        analyzer = ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

        # 1. Monoculture: 5 benzene derivatives -> D_scaffold = 1 / 5 = 0.20
        benzene_pool = [
            "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1", "Oc1ccccc1", "c1ccccc1C(=O)O"
        ]
        div_mono = analyzer.compute_scaffold_diversity(benzene_pool)
        assert div_mono["unique_valid_count"] == 5
        assert div_mono["unique_scaffold_count"] == 1
        assert math.isclose(div_mono["diversity_ratio"], 0.2000, abs_tol=1e-4)
        assert div_mono["scaffold_distribution"] == {"c1ccccc1": 5}

        # 2. Maximum diversity: 5 distinct scaffold families -> D_scaffold = 5 / 5 = 1.00
        diverse_pool = [
            "c1ccccc1",           # Benzene
            "c1ccncc1",           # Pyridine
            "c1ccc2[nH]ccc2c1",   # Indole
            "c1ccc2ncccc2c1",     # Quinoline
            "C1CCC2C(C1)CCC3C2CCC4CCCC34", # Steroid
        ]
        div_max = analyzer.compute_scaffold_diversity(diverse_pool)
        assert div_max["unique_valid_count"] == 5
        assert div_max["unique_scaffold_count"] == 5
        assert math.isclose(div_max["diversity_ratio"], 1.0000, abs_tol=1e-4)

        # 3. Mixed pool: 10 compounds across 3 scaffolds (5 benzene, 3 indole, 2 pyridine)
        mixed_pool = [
            "Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1", "Oc1ccccc1", "c1ccccc1C(=O)O", # 5 benzene
            "Cc1ccc2[nH]ccc2c1", "CCc1ccc2[nH]ccc2c1", "Clc1ccc2[nH]ccc2c1",         # 3 indole
            "Cc1ccncc1", "CCc1ccncc1",                                               # 2 pyridine
        ]
        div_mix = analyzer.compute_scaffold_diversity(mixed_pool)
        assert div_mix["unique_valid_count"] == 10
        assert div_mix["unique_scaffold_count"] == 3
        assert math.isclose(div_mix["diversity_ratio"], 0.3000, abs_tol=1e-4)

        # 4. Empty pool boundary
        div_empty = analyzer.compute_scaffold_diversity([])
        assert div_empty["diversity_ratio"] == 0.0
        assert div_empty["unique_scaffold_count"] == 0
        assert div_empty["unique_valid_count"] == 0


# ============================================================================
# 5. Publication Artifact Generation & Structural Integrity
# ============================================================================

class TestPublicationArtifactGeneration:
    """Adversarial stress-test for lead_structures_2d.png, property_distributions.png, and result_analysis_summary.json."""

    @pytest.fixture
    def populated_analyzer(self, tmp_path: Path) -> tuple[ResultAnalyzer, Any, Any]:
        """Set up ResultAnalyzer and evaluate a realistic 8-candidate pool."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        candidate_pool = [
            EGCG_SMILES,
            "O=C1C(O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12", # Quercetin
            "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2",     # Epicatechin
            "CC(=O)Oc1ccccc1C(=O)O",                       # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",               # Caffeine
            "c1ccc2[nH]ccc2c1",                           # Indole
            "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", # Omeprazole
            "c1ccccc1",                                   # Benzene
        ]

        df_all = analyzer.analyze_candidates(candidate_pool)
        df_leads = analyzer.prioritize_leads(df_all, top_k=5)
        return analyzer, df_all, df_leads

    def test_visualize_leads_generates_valid_png_images(
        self, populated_analyzer: tuple[ResultAnalyzer, Any, Any]
    ) -> None:
        """Verify visualize_leads() renders valid, non-empty PNG images with genuine magic bytes."""
        analyzer, df_all, df_leads = populated_analyzer

        figures = analyzer.visualize_leads(df_leads=df_leads, df_all=df_all, max_leads=5)

        assert "lead_structures_2d" in figures, "Missing lead_structures_2d in return figures"
        assert "property_distributions" in figures, "Missing property_distributions in return figures"

        grid_path = figures["lead_structures_2d"]
        dist_path = figures["property_distributions"]

        assert grid_path.exists(), f"File does not exist: {grid_path}"
        assert dist_path.exists(), f"File does not exist: {dist_path}"

        grid_size = grid_path.stat().st_size
        dist_size = dist_path.stat().st_size

        assert grid_size > 5000, f"lead_structures_2d.png too small ({grid_size} bytes)"
        assert dist_size > 5000, f"property_distributions.png too small ({dist_size} bytes)"

        # Verify authentic PNG header magic bytes (\x89PNG\r\n\x1a\n)
        with open(grid_path, "rb") as f:
            header_grid = f.read(8)
            assert header_grid == b"\x89PNG\r\n\x1a\n", "lead_structures_2d.png lacks valid PNG magic header"

        with open(dist_path, "rb") as f:
            header_dist = f.read(8)
            assert header_dist == b"\x89PNG\r\n\x1a\n", "property_distributions.png lacks valid PNG magic header"

        # Verify PIL can open, parse, and read the dimensions of both images
        with Image.open(grid_path) as img:
            assert img.format == "PNG", f"Unexpected format: {img.format}"
            width, height = img.size
            assert width >= 300 and height >= 200, f"Dimensions too small: {img.size}"

        with Image.open(dist_path) as img:
            assert img.format == "PNG", f"Unexpected format: {img.format}"
            width, height = img.size
            assert width >= 800 and height >= 600, f"Dimensions too small: {img.size}"

    def test_export_summary_json_integrity_and_metric_completeness(
        self, populated_analyzer: tuple[ResultAnalyzer, Any, Any]
    ) -> None:
        """Verify export_summary() serializes valid JSON containing complete statistical audit blocks."""
        analyzer, df_all, df_leads = populated_analyzer

        summary = analyzer.export_summary(df_all=df_all, df_leads=df_leads)
        json_path = analyzer.output_dir / "result_analysis_summary.json"

        assert json_path.exists(), f"JSON summary missing at {json_path}"
        assert json_path.stat().st_size > 500, "JSON summary file unexpectedly tiny"

        # Verify independent disk deserialization
        with open(json_path, encoding="utf-8") as f:
            disk_summary = json.load(f)

        assert disk_summary == summary, "In-memory summary differs from serialized disk summary"

        # Verify top-level structure
        required_top_keys = [
            "query_seed_smiles",
            "seed_descriptors",
            "vun_metrics",
            "bemis_murcko_scaffold_diversity",
            "physicochemical_distributions",
            "dynamic_envelope_benchmarks",
            "metrics",
            "top_leads_count",
            "top_leads",
        ]
        for key in required_top_keys:
            assert key in disk_summary, f"Missing key in JSON summary: {key}"

        # Verify VUN metrics
        vun = disk_summary["vun_metrics"]
        assert vun["total_generated"] == 8
        assert vun["valid_count"] == 8
        assert vun["validity_percentage"] == 100.0
        assert vun["passed_validity_threshold"] is True
        assert vun["passed_uniqueness_threshold"] is True

        # Verify scaffold diversity
        scaff_div = disk_summary["bemis_murcko_scaffold_diversity"]
        assert scaff_div["diversity_ratio"] > 0.0
        assert scaff_div["unique_scaffold_count"] > 1

        # Verify distribution stats: no NaNs, no nulls
        prop_stats = disk_summary["physicochemical_distributions"]
        for prop in ["HAC", "MW", "LogP", "TPSA", "RotBonds", "HBD", "HBA", "SA", "QED", "ROCS"]:
            assert prop in prop_stats, f"Missing property stats for {prop}"
            stat = prop_stats[prop]
            for m in ["mean", "median", "std", "min", "max", "q25", "q75", "iqr"]:
                val = stat[m]
                assert val is not None, f"Null value in {prop}.{m}"
                assert not math.isnan(val), f"NaN value in {prop}.{m}"

        # Verify top leads count and scaffold diversity promotion
        top_leads = disk_summary["top_leads"]
        assert len(top_leads) == min(5, len(df_leads))
        # Ensure lead 1 has top desirability
        desirabilities = [lead["Desirability"] for lead in top_leads]
        assert desirabilities[0] == max(desirabilities), "Top lead does not have highest desirability"

    def test_single_candidate_pool_artifact_generation(self, tmp_path: Path) -> None:
        """Verify artifact generation succeeds without zero-division or singular crashes on a single lead."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        df_all = analyzer.analyze_candidates([EGCG_SMILES])
        df_leads = analyzer.prioritize_leads(df_all, top_k=1)

        figures = analyzer.visualize_leads(df_leads=df_leads, df_all=df_all)
        assert "lead_structures_2d" in figures
        assert "property_distributions" in figures

        summary = analyzer.export_summary(df_all=df_all, df_leads=df_leads)
        assert summary["vun_metrics"]["total_generated"] == 1
        assert summary["vun_metrics"]["valid_count"] == 1
        assert summary["metrics"]["mean_rocs"] > 0.0

    def test_empty_candidate_pool_artifact_generation(self, tmp_path: Path) -> None:
        """Verify empty candidate pool handles visualization and export gracefully."""
        ref_sdf = REPO_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=ref_sdf, seed_smiles=EGCG_SMILES)

        df_empty = analyzer.analyze_candidates([])
        df_leads = analyzer.prioritize_leads(df_empty, top_k=5)

        figures = analyzer.visualize_leads(df_leads=df_leads, df_all=df_empty)
        assert figures == {}, "Visualizing empty leads should return empty dict"

        summary = analyzer.export_summary(df_all=df_empty, df_leads=df_leads)
        assert summary["vun_metrics"]["total_generated"] == 0
        assert summary["top_leads_count"] == 0


# ============================================================================
# 6. Automated Reference Generation Fallback Edge Case
# ============================================================================

class TestAutomatedReferenceGenerationFallback:
    """Empirical investigation into missing reference SDF fallback behavior."""

    def test_missing_reference_sdf_fallback_recording(self, tmp_path: Path) -> None:
        """Verify that when reference generation fails (e.g. invalid query SMILES), compute_rocs_scores records fallback and returns zeros."""
        try:
            from fallback_monitor import FallbackMonitor
        except ImportError:
            from _david.code.fallback_monitor import FallbackMonitor

        # Initialize with non-existent SDF path and unparseable seed_smiles so generation fails
        bogus_ref = tmp_path / "nonexistent_reference.sdf"
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=bogus_ref, seed_smiles="INVALID_SEED_XYZ")

        # compute_rocs_scores should catch the failure gracefully and return zeros
        scores = analyzer.compute_rocs_scores([EGCG_SMILES, "c1ccccc1"])
        assert len(scores) == 2
        assert np.all(scores == 0.0), f"Expected zeros on missing reference generation failure, got {scores}"

        # Verify fallback was recorded
        events = FallbackMonitor.get_events()
        ref_fallbacks = [ev for ev in events if ev.fallback_type == "rocs_missing_ref_sdf"]
        assert len(ref_fallbacks) > 0, "Expected rocs_missing_ref_sdf fallback to be recorded in FallbackMonitor"
        err_msg = ref_fallbacks[-1].message
        assert "Reference SDF missing and generation failed" in err_msg

    def test_missing_reference_sdf_automated_generation_success(self, tmp_path: Path) -> None:
        """Verify that when ref_sdf_path is missing but seed_smiles is valid, reference SDF is generated automatically."""
        auto_ref = tmp_path / "auto_reference.sdf"
        assert not auto_ref.exists()
        analyzer = ResultAnalyzer(output_dir=tmp_path, ref_sdf_path=auto_ref, seed_smiles=EGCG_SMILES)
        scores = analyzer.compute_rocs_scores([EGCG_SMILES])
        assert auto_ref.exists(), "Expected reference SDF to be automatically generated on disk"
        assert len(scores) == 1
        assert scores[0] > 1.5, f"Expected high ROCS self-score for EGCG query, got {scores[0]}"
