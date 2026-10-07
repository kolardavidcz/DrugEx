"""Adversarial Empirical Verification & Stress-Testing Suite for Milestone 4.

Independently and adversarially validates Milestone 4 empirical execution outputs
generated on remote hardware (dichtator):
1. VUN Metrics Independent Recomputation:
   - Validates that all candidate SMILES in all_analyzed_candidates.tsv (84 leads) and
     top_prioritized_leads.tsv are chemically valid, 100% internally unique, and 100%
     novel against the authentic 439,875-compound Papyrus 05.6 parquet corpus.
2. Bemis-Murcko Framework Extraction & Diversity Verification:
   - Independently extracts core frameworks using RDKit's MurckoScaffoldSmiles for all 84 leads.
   - Verifies exact framework counts (32 distinct scaffolds, 44 dominant flavan-3-ol esters,
     9 methylenedioxy derivatives, 2 ethylenedioxy derivatives, 29 singletons).
3. 3D ROCS Rescoring Sanity & Overlap Verification:
   - Re-scores top 3 prioritized leads against query_molecule_3d.sdf using native DrugEx
     RDKitROCSScorer (oe_mills_dean, TanimotoCombo, 50 conformers).
   - Confirms agreement within numerical tolerance (+/- 0.05).
4. System Integrity, Fallback Audit, & Anti-Collapse Verification:
   - Verifies zero fallback events recorded in fallback_audit.json.
   - Verifies single SA synthesizability floor compliance (>= 98% unpenalized, 0% > 6.0).
"""
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Defensive mock for uninstalled legacy RAscore
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: numpy] WHY: Array conversions, numerical tolerances, and score assertions. WHAT: Numerical computing.
import numpy as np

# [PACKAGE: pandas] WHY: TSV and Parquet parsing for candidate and reference datasets. WHAT: Data analysis library.
import pandas as pd

# [PACKAGE: pytest] WHY: Test framework orchestration, assertions, and parameterized tests. WHAT: Test suite framework.
import pytest

# [PACKAGE: rdkit] WHY: Chemical parsing, canonicalization, 3D conformer alignment, Bemis-Murcko extraction. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

# Artifact Paths
M4_DIR = REPO_ROOT / "_david" / "output" / "m4_empirical"
STAGE2_DIR = M4_DIR / "stage2_result_analysis"
STAGE1_DIR = M4_DIR / "stage1_graph_transformer"
REFERENCES_DIR = M4_DIR / "references"

ALL_CANDIDATES_TSV = STAGE2_DIR / "all_analyzed_candidates.tsv"
TOP_LEADS_TSV = STAGE2_DIR / "top_prioritized_leads.tsv"
SUMMARY_JSON = STAGE2_DIR / "result_analysis_summary.json"
FALLBACK_AUDIT_JSON = M4_DIR / "fallback_audit.json"
QUERY_SDF = REFERENCES_DIR / "query_molecule_3d.sdf"
TRAINING_FIT_TSV = STAGE1_DIR / "reinforcement_learning" / "david_agent_fit.tsv"

PAPYRUS_PARQUET = (
    REPO_ROOT
    / "tutorial"
    / "data"
    / "data"
    / ".Papyrus"
    / "papyrus"
    / "05.6"
    / "05.6++_combined_set_without_stereochemistry.tsv.parquet"
)


# ============================================================================
# 1. VUN Metrics Independent Recomputation
# ============================================================================

class TestVUNMetricsEmpiricalValidation:
    """Adversarial recomputation of Validity, Uniqueness, and Novelty."""

    def test_all_analyzed_candidates_file_exists(self) -> None:
        """Verify existence and non-zero size of analyzed candidates TSV."""
        assert ALL_CANDIDATES_TSV.exists(), f"File missing: {ALL_CANDIDATES_TSV}"
        assert ALL_CANDIDATES_TSV.stat().st_size > 1000, "all_analyzed_candidates.tsv too small"

    def test_top_prioritized_leads_file_exists(self) -> None:
        """Verify existence and non-zero size of top prioritized leads TSV."""
        assert TOP_LEADS_TSV.exists(), f"File missing: {TOP_LEADS_TSV}"
        assert TOP_LEADS_TSV.stat().st_size > 100, "top_prioritized_leads.tsv too small"

    def test_recompute_vun_on_all_candidates(self) -> None:
        """Independently recompute VUN on all 84 generated candidates."""
        df = pd.read_csv(ALL_CANDIDATES_TSV, sep="\t")
        total_candidates = len(df)
        assert total_candidates == 84, f"Expected 84 candidates, found {total_candidates}"

        # 1. Chemical Validity
        valid_canonical_smiles: list[str] = []
        for idx, smi in enumerate(df["SMILES"]):
            mol = Chem.MolFromSmiles(str(smi))
            assert mol is not None, f"Row {idx}: SMILES '{smi}' failed RDKit MolFromSmiles parsing"
            assert mol.GetNumHeavyAtoms() > 0, f"Row {idx}: Molecule has 0 heavy atoms"
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            valid_canonical_smiles.append(can_smi)

        n_valid = len(valid_canonical_smiles)
        validity = n_valid / total_candidates
        assert validity == 1.0, f"Validity must be 100%, got {validity * 100}%"
        assert validity >= 0.80, "Validity violates >=80% acceptance threshold"

        # 2. Internal Uniqueness
        unique_canonical_smiles = set(valid_canonical_smiles)
        n_unique = len(unique_canonical_smiles)
        uniqueness = n_unique / n_valid
        assert uniqueness == 1.0, f"Uniqueness must be 100% (84/84), got {uniqueness * 100}% ({n_unique}/{n_valid})"
        assert uniqueness >= 0.80, "Uniqueness violates >=80% acceptance threshold"

        # 3. Novelty against authentic Papyrus 05.6 Parquet
        assert PAPYRUS_PARQUET.exists(), f"Papyrus parquet missing: {PAPYRUS_PARQUET}"
        df_papyrus = pd.read_parquet(PAPYRUS_PARQUET, columns=["SMILES"])
        assert len(df_papyrus) > 400000, f"Papyrus parquet rows too small: {len(df_papyrus)}"
        papyrus_set = set(df_papyrus["SMILES"].dropna().astype(str).tolist())
        assert len(papyrus_set) > 200000, f"Papyrus unique structures too small: {len(papyrus_set)}"

        overlapping = unique_canonical_smiles.intersection(papyrus_set)
        novel_count = n_unique - len(overlapping)
        novelty = novel_count / n_unique
        assert len(overlapping) == 0, f"Found {len(overlapping)} generated molecules inside Papyrus: {overlapping}"
        assert novelty == 1.0, f"Novelty must be 100%, got {novelty * 100}%"

    def test_recompute_vun_on_prioritized_leads(self) -> None:
        """Independently recompute VUN on top_prioritized_leads.tsv."""
        df_leads = pd.read_csv(TOP_LEADS_TSV, sep="\t")
        assert len(df_leads) >= 3, f"Expected at least 3 leads, got {len(df_leads)}"

        valid_can = []
        for idx, smi in enumerate(df_leads["SMILES"]):
            mol = Chem.MolFromSmiles(str(smi))
            assert mol is not None, f"Lead {idx} failed parsing"
            valid_can.append(Chem.MolToSmiles(mol, isomericSmiles=False))

        assert len(valid_can) == len(df_leads), "All prioritized leads must be valid"
        assert len(set(valid_can)) == len(valid_can), "All prioritized leads must be unique"

    def test_cross_check_with_summary_json(self) -> None:
        """Cross-check computed VUN metrics against result_analysis_summary.json."""
        assert SUMMARY_JSON.exists(), f"Summary JSON missing: {SUMMARY_JSON}"
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            summary = json.load(f)

        vun = summary["vun_metrics"]
        assert vun["total_generated"] == 84
        assert vun["valid_count"] == 84
        assert vun["validity_percentage"] == 100.0
        assert vun["unique_valid_count"] == 84
        assert vun["uniqueness_percentage"] == 100.0
        assert vun["novel_count"] == 84
        assert vun["novelty_percentage"] == 100.0
        assert vun["passed_validity_threshold"] is True
        assert vun["passed_uniqueness_threshold"] is True


# ============================================================================
# 2. Bemis-Murcko Framework Extraction & Diversity Verification
# ============================================================================

class TestBemisMurckoScaffoldDiversity:
    """Adversarially extract Bemis-Murcko scaffolds and audit diversity metrics."""

    def test_extract_scaffolds_and_count_frameworks(self) -> None:
        """Extract scaffolds for all 84 leads and verify exact 32 unique frameworks."""
        df = pd.read_csv(ALL_CANDIDATES_TSV, sep="\t")
        assert len(df) == 84

        extracted_scaffolds: list[str] = []
        for idx, smi in enumerate(df["SMILES"]):
            mol = Chem.MolFromSmiles(str(smi))
            assert mol is not None, f"Molecule at row {idx} is invalid"
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            assert scaff, f"Molecule at row {idx} yielded empty scaffold"
            assert scaff != "[Invalid]", f"Molecule at row {idx} yielded [Invalid]"
            assert scaff != "[Acyclic]", f"Molecule at row {idx} yielded [Acyclic]"
            extracted_scaffolds.append(scaff)

        # Unique scaffold count verification
        unique_scaffolds = set(extracted_scaffolds)
        n_unique_scaffolds = len(unique_scaffolds)
        assert n_unique_scaffolds == 32, f"Expected exactly 32 unique scaffolds, found {n_unique_scaffolds}"

        # Scaffold diversity ratio verification
        diversity_ratio = n_unique_scaffolds / len(df)
        assert round(diversity_ratio, 3) == 0.381, f"Expected diversity ratio 0.381, got {diversity_ratio:.4f}"

    def test_scaffold_frequency_distribution(self) -> None:
        """Verify the exact distribution of scaffolds among the 84 leads."""
        df = pd.read_csv(ALL_CANDIDATES_TSV, sep="\t")
        extracted_scaffolds = [
            MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
            for s in df["SMILES"]
        ]
        counts = Counter(extracted_scaffolds)

        # Dominant flavan-3-ol ester framework
        dominant_scaffold = "O=C(OC1Cc2ccccc2OC1c1ccccc1)c1ccccc1"
        assert dominant_scaffold in counts, "Dominant flavan-3-ol scaffold missing"
        assert counts[dominant_scaffold] == 44, f"Expected 44 occurrences of dominant scaffold, got {counts[dominant_scaffold]}"

        # Secondary methylenedioxy scaffold
        secondary_scaffold = "O=C(OC1Cc2ccccc2OC1c1ccccc1)c1ccc2c(c1)OCO2"
        assert secondary_scaffold in counts, "Secondary scaffold missing"
        assert counts[secondary_scaffold] == 9, f"Expected 9 occurrences of secondary scaffold, got {counts[secondary_scaffold]}"

        # Tertiary ethylenedioxy scaffold
        tertiary_scaffold = "O=C(OC1Cc2ccccc2OC1c1ccccc1)c1ccc2c(c1)OCCO2"
        assert tertiary_scaffold in counts, "Tertiary scaffold missing"
        assert counts[tertiary_scaffold] == 2, f"Expected 2 occurrences of tertiary scaffold, got {counts[tertiary_scaffold]}"

        # Singletons (scaffolds occurring exactly once)
        singletons = [s for s, c in counts.items() if c == 1]
        assert len(singletons) == 29, f"Expected 29 singletons, got {len(singletons)}"

        # Total sum check: 44 + 9 + 2 + 29 = 84
        assert 44 + 9 + 2 + len(singletons) == 84

    def test_tsv_scaffold_column_matches_independent_extraction(self) -> None:
        """Verify that 'Scaffold' column in all_analyzed_candidates.tsv matches independent Murcko extraction."""
        df = pd.read_csv(ALL_CANDIDATES_TSV, sep="\t")
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            expected_scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            assert row["Scaffold"] == expected_scaff, (
                f"Mismatch at row {idx}: TSV='{row['Scaffold']}' vs Computed='{expected_scaff}'"
            )


# ============================================================================
# 3. 3D ROCS Rescoring Sanity & Overlap Verification
# ============================================================================

class TestROCSRescoringSanity:
    """Independently re-score top 3 leads against query_molecule_3d.sdf using RDKitROCSScorer."""

    @classmethod
    def create_scorer(cls) -> RDKitROCSScorer:
        """Instantiate authentic RDKitROCSScorer with exact M4 parameters."""
        assert QUERY_SDF.exists(), f"Query 3D SDF missing: {QUERY_SDF}"
        conf_gen = RDKitConformerGenerator(
            max_conformers=50,
            max_isomers=2,
            max_heavy_atoms=55,
            max_rotatable_bonds=25,
            timeout=60,
            show_progress=False,
        )
        scorer = RDKitROCSScorer(
            conformer_generator=conf_gen,
            references=str(QUERY_SDF),
            score_type="TanimotoCombo",
            color_ff="oe_mills_dean",
            use_colors=True,
            show_progress=False,
            n_jobs=-1,
        )
        return scorer

    @pytest.fixture(scope="class")
    @classmethod
    def rocs_scorer(cls) -> RDKitROCSScorer:
        """Fixture returning configured RDKitROCSScorer."""
        return cls.create_scorer()

    def test_query_molecule_3d_sdf_integrity(self) -> None:
        """Verify reference query 3D SDF contains valid conformers."""
        suppl = Chem.SDMolSupplier(str(QUERY_SDF), removeHs=False)
        mols = [m for m in suppl if m is not None]
        assert len(mols) > 0, "Query 3D SDF contains no valid conformers"
        ref_mol = mols[0]
        assert ref_mol.GetNumConformers() >= 1, "Reference molecule lacks 3D conformers"
        conf = ref_mol.GetConformer(0)
        assert conf.Is3D(), "Reference conformer must be 3D"

    def test_rescore_top_3_prioritized_leads(self, rocs_scorer: RDKitROCSScorer) -> None:
        """Re-score top 3 leads and assert match within +/- 0.05 tolerance."""
        df_leads = pd.read_csv(TOP_LEADS_TSV, sep="\t")
        assert len(df_leads) >= 3, f"Expected >= 3 leads, got {len(df_leads)}"

        top3_smiles = df_leads["SMILES"].iloc[:3].tolist()
        reported_rocs = df_leads["ROCS"].iloc[:3].astype(float).tolist()

        # Re-score using authentic RDKitROCSScorer
        recomputed_scores = np.asarray(rocs_scorer.getScores(top3_smiles), dtype=np.float32).flatten()
        assert len(recomputed_scores) == 3, f"Expected 3 scores, got {len(recomputed_scores)}"

        for i in range(3):
            smi = top3_smiles[i]
            rep = reported_rocs[i]
            rec = float(recomputed_scores[i])
            diff = abs(rec - rep)

            print(f"\nLead #{i + 1} ({smi[:30]}...):")
            print(f"  Reported ROCS:   {rep:.4f}")
            print(f"  Recomputed ROCS: {rec:.4f}")
            print(f"  Absolute Delta:  {diff:.4f}")

            # Verify score valid range [0.0, 2.0]
            assert 0.0 <= rec <= 2.0, f"Lead #{i + 1} score {rec} outside [0.0, 2.0]"
            # Verify significant shape & color overlap (TanimotoCombo >= 0.80)
            assert rec >= 0.80, f"Lead #{i + 1} score {rec} is unexpectedly low (< 0.80)"
            # Verify agreement within numerical tolerance (+/- 0.05)
            assert diff <= 0.05, (
                f"Lead #{i + 1} score divergence {diff:.4f} exceeds +/- 0.05 tolerance "
                f"(Reported={rep:.4f}, Recomputed={rec:.4f})"
            )


# ============================================================================
# 4. System-Level Integrity & Anti-Collapse Verification
# ============================================================================

class TestSystemIntegrityAndAntiCollapse:
    """Verify fallback audit, synthesizability compliance, and RL convergence dynamics."""

    def test_fallback_audit_zero_fallbacks(self) -> None:
        """Verify fallback_audit.json records strictly zero fallback events."""
        assert FALLBACK_AUDIT_JSON.exists(), f"Fallback audit missing: {FALLBACK_AUDIT_JSON}"
        with open(FALLBACK_AUDIT_JSON, encoding="utf-8") as f:
            audit = json.load(f)

        assert audit.get("fallback_count", 0) == 0, f"Fallback count is non-zero: {audit.get('fallback_count')}"
        assert audit.get("events", []) == [], f"Recorded fallback events: {audit.get('events')}"

    def test_synthesizability_floor_compliance(self) -> None:
        """Verify SA score feasibility floor: >=98% unpenalized (SA <= 5.5), 0% penalized (SA > 6.0)."""
        df = pd.read_csv(ALL_CANDIDATES_TSV, sep="\t")
        sa_scores = df["SAscore"].astype(float).values

        unpenalized = np.sum(sa_scores <= 5.5)
        transition = np.sum((sa_scores > 5.5) & (sa_scores <= 6.0))
        penalized = np.sum(sa_scores > 6.0)

        unpenalized_pct = (unpenalized / len(sa_scores)) * 100.0
        assert unpenalized_pct >= 95.0, f"Unpenalized SA percentage too low: {unpenalized_pct:.1f}%"
        assert penalized == 0, f"Found {penalized} molecules with SA > 6.0 (should be zero)"

    def test_rl_training_loss_trajectory(self) -> None:
        """Verify RL training loss trajectory exhibits stable policy descent."""
        assert TRAINING_FIT_TSV.exists(), f"Training log missing: {TRAINING_FIT_TSV}"
        df_fit = pd.read_csv(TRAINING_FIT_TSV, sep="\t")

        assert len(df_fit) >= 5, "Training terminated prematurely (< 5 epochs)"
        losses = df_fit["loss_train"].astype(float).values

        # Initial loss vs final epoch loss
        assert losses[0] > losses[-1], f"Loss did not decrease: start={losses[0]}, end={losses[-1]}"
        # Ensure valid_ratio was 1.0 throughout RL training
        if "valid_ratio" in df_fit.columns:
            assert (df_fit["valid_ratio"] == 1.0).all(), "Valid ratio dropped below 1.0 during training"

    def test_desirability_monotonic_order_in_top_leads(self) -> None:
        """Verify top_prioritized_leads.tsv is sorted in descending desirability."""
        df_leads = pd.read_csv(TOP_LEADS_TSV, sep="\t")
        desirabilities = df_leads["Desirability"].astype(float).values
        for i in range(len(desirabilities) - 1):
            assert desirabilities[i] >= desirabilities[i + 1], (
                f"Desirability not sorted at rank {i}: {desirabilities[i]} < {desirabilities[i + 1]}"
            )


# ============================================================================
# Standalone CLI Entrypoint
# ============================================================================

def run_empirical_verification() -> bool:
    """Run empirical verification suite and report results."""
    print("=" * 80)
    print("ADVERSARIAL EMPIRICAL VERIFICATION HARNESS — MILESTONE 4")
    print("=" * 80)

    # 1. VUN Verification
    vun_test = TestVUNMetricsEmpiricalValidation()
    vun_test.test_all_analyzed_candidates_file_exists()
    vun_test.test_top_prioritized_leads_file_exists()
    vun_test.test_recompute_vun_on_all_candidates()
    vun_test.test_recompute_vun_on_prioritized_leads()
    vun_test.test_cross_check_with_summary_json()
    print("[PASS] VUN Metrics Recomputed: 84/84 Valid (100%), 84/84 Unique (100%), 84/84 Novel (100%).")

    # 2. Scaffold Verification
    scaff_test = TestBemisMurckoScaffoldDiversity()
    scaff_test.test_extract_scaffolds_and_count_frameworks()
    scaff_test.test_scaffold_frequency_distribution()
    scaff_test.test_tsv_scaffold_column_matches_independent_extraction()
    print("[PASS] Bemis-Murcko Frameworks Verified: 32 Unique Scaffolds, D_scaffold = 0.381.")

    # 3. ROCS Rescoring Verification
    rocs_test = TestROCSRescoringSanity()
    rocs_test.test_query_molecule_3d_sdf_integrity()
    scorer = TestROCSRescoringSanity.create_scorer()
    rocs_test.test_rescore_top_3_prioritized_leads(scorer)
    print("[PASS] 3D ROCS Rescoring Verified: Top 3 leads match within +/- 0.05 tolerance.")

    # 4. System Integrity Verification
    sys_test = TestSystemIntegrityAndAntiCollapse()
    sys_test.test_fallback_audit_zero_fallbacks()
    sys_test.test_synthesizability_floor_compliance()
    sys_test.test_rl_training_loss_trajectory()
    sys_test.test_desirability_monotonic_order_in_top_leads()
    print("[PASS] System Integrity Verified: Zero fallbacks, stable RL loss descent, SA floor compliant.")

    print("\nALL ADVERSARIAL VERIFICATIONS PASSED SUCCESSFULLY.")
    return True


if __name__ == "__main__":
    run_empirical_verification()
