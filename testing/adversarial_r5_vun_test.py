"""Adversarial Empirical Stress-Testing Harness for Requirement R5 (Milestone M3).

Focus: ResultAnalyzer, Academic VUN Metric Calculation, Boundary Conditions,
and Unbiased Lead Sampling (drop_invalid=False, drop_duplicates=False).

Adversarial Stress Test Dimensions:
1. Empty Candidate Pool:
   - compute_vun_metrics([]) returns 0.0% validity, uniqueness, novelty without raising exceptions.
   - Also stress-tested with empty tuple, empty DataFrame, and DataFrame lacking SMILES column.
2. 100% Invalid Molecules Pool:
   - Pool with completely unparseable garbage strings, whitespace, None, 0 heavy atoms,
     and chemically impossible valences returns 0.0% validity, uniqueness, novelty without crashing.
   - Guaranteed zero-division protection on all counts and percentages.
3. 100% Duplicate Molecules Pool:
   - Pool of N identical molecules yields 100% validity, uniqueness = 1/N.
   - Multi-syntax representation of same molecule (Kekule vs aromatic, bracketed atoms)
     collapsing to single canonical SMILES.
   - Stereoisomer pool collapsing to single non-isomeric canonical SMILES (isomericSmiles=False).
4. Zero Novelty Against Reference Corpus:
   - Candidates 100% present in reference corpus yield strictly 0.0% Novelty.
   - Verified across set, list, text file, and parquet reference corpus inputs.
5. 100% Novelty Against Reference Corpus:
   - Candidates completely absent from reference corpus yield strictly 100.0% Novelty.
6. Single Candidate Edge Case:
   - Single valid molecule yields 100% validity and 100% uniqueness.
   - Single invalid molecule yields 0% validity, 0% uniqueness, 0% novelty.
7. Unbiased Sampling Verification:
   - sample_candidates() forces drop_invalid=False and drop_duplicates=False regardless
     of kwargs passed to preserve the raw generative distribution (N_total).
   - Preserves both invalid SMILES and duplicate structures in the generated DataFrame.
   - Rejects non-generator objects with TypeError.
8. Scaffold Diversity & Massive Pool Stress:
   - Stress-testing compute_scaffold_diversity with acyclic, invalid, and empty SMILES.
   - Large-scale candidate pool (1,000 compounds) with complex mixed composition.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Defensive mock for uninstalled legacy RAscore
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: pandas] WHY: Tabular dataframe manipulation for candidate sets and parquet datasets. WHAT: Data manipulation library.
import pandas as pd

# [PACKAGE: pytest] WHY: Test assertions, parameterized test matrices, and test execution. WHAT: Python test framework.
import pytest

# Ensure repo root and _david/code are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DAVID_CODE = REPO_ROOT / "_david" / "code"
if str(DAVID_CODE) not in sys.path:
    sys.path.insert(0, str(DAVID_CODE))

from _david.code.result_analyzer import EGCG_SMILES, ResultAnalyzer  # noqa: E402

# ============================================================================
# Dimension 1: Empty Candidate Pool Stress Tests
# ============================================================================

class TestEmptyCandidatePool:
    """Adversarial stress-testing of VUN computation on empty candidate pools."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        """Create ResultAnalyzer instance backed by temporary directory."""
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_empty_list_returns_zero_metrics_without_exception(self, analyzer: ResultAnalyzer) -> None:
        """Verify compute_vun_metrics([]) returns 0.0% validity, uniqueness, novelty without error."""
        res = analyzer.compute_vun_metrics([])
        assert res["total_generated"] == 0, "total_generated must be 0 for empty list"
        assert res["valid_count"] == 0, "valid_count must be 0"
        assert res["unique_valid_count"] == 0, "unique_valid_count must be 0"
        assert res["novel_count"] == 0, "novel_count must be 0"
        assert res["validity"] == 0.0, "validity must be 0.0"
        assert res["validity_percentage"] == 0.0, "validity_percentage must be 0.0"
        assert res["uniqueness"] == 0.0, "uniqueness must be 0.0"
        assert res["uniqueness_percentage"] == 0.0, "uniqueness_percentage must be 0.0"
        assert res["novelty"] == 0.0, "novelty must be 0.0"
        assert res["novelty_percentage"] == 0.0, "novelty_percentage must be 0.0"
        assert res["valid_canonical_smiles"] == []
        assert res["unique_canonical_smiles"] == []
        assert res["passed_validity"] is False
        assert res["passed_uniqueness"] is False

    def test_empty_tuple_and_empty_generator(self, analyzer: ResultAnalyzer) -> None:
        """Verify empty tuple and generator expressions are handled gracefully."""
        res_tuple = analyzer.compute_vun_metrics(())
        assert res_tuple["total_generated"] == 0
        assert res_tuple["validity"] == 0.0
        assert res_tuple["uniqueness"] == 0.0
        assert res_tuple["novelty"] == 0.0

        res_gen = analyzer.compute_vun_metrics(x for x in [])
        assert res_gen["total_generated"] == 0
        assert res_gen["validity"] == 0.0

    def test_empty_dataframe_variations(self, analyzer: ResultAnalyzer) -> None:
        """Verify empty DataFrames with various column schemas return 0.0 without crash."""
        # 1. Empty DataFrame with no columns
        df_no_cols = pd.DataFrame()
        res1 = analyzer.compute_vun_metrics(df_no_cols)
        assert res1["total_generated"] == 0
        assert res1["validity"] == 0.0

        # 2. Empty DataFrame with standard SMILES column
        df_smiles = pd.DataFrame({"SMILES": []})
        res2 = analyzer.compute_vun_metrics(df_smiles)
        assert res2["total_generated"] == 0
        assert res2["validity"] == 0.0

        # 3. Empty DataFrame with lowercase smiles column
        df_lower = pd.DataFrame({"smiles": []})
        res3 = analyzer.compute_vun_metrics(df_lower)
        assert res3["total_generated"] == 0
        assert res3["validity"] == 0.0

        # 4. Empty DataFrame with non-SMILES column
        df_unrelated = pd.DataFrame({"Structure": []})
        res4 = analyzer.compute_vun_metrics(df_unrelated)
        assert res4["total_generated"] == 0
        assert res4["validity"] == 0.0


# ============================================================================
# Dimension 2: 100% Invalid Molecules Pool Stress Tests
# ============================================================================

class TestAllInvalidMoleculesPool:
    """Adversarial stress-testing of VUN computation on completely invalid pools."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_garbage_strings_pool(self, analyzer: ResultAnalyzer) -> None:
        """Verify pool of random unparseable strings yields 0.0% for V, U, and N."""
        garbage_pool = [
            "NOT_A_SMILES_1",
            "!!!INVALID$$$",
            "123456789",
            "C12345",
            "C(=O)(=O)(=O)(=O)",  # Chemically impossible valence
            "%%%%%",
            "[[[[]]]]",
        ]
        res = analyzer.compute_vun_metrics(garbage_pool)
        assert res["total_generated"] == len(garbage_pool)
        assert res["valid_count"] == 0
        assert res["unique_valid_count"] == 0
        assert res["novel_count"] == 0
        assert res["validity"] == 0.0
        assert res["validity_percentage"] == 0.0
        assert res["uniqueness"] == 0.0
        assert res["uniqueness_percentage"] == 0.0
        assert res["novelty"] == 0.0
        assert res["novelty_percentage"] == 0.0
        assert res["passed_validity"] is False
        assert res["passed_uniqueness"] is False

    def test_empty_strings_whitespace_and_none(self, analyzer: ResultAnalyzer) -> None:
        """Verify pool containing empty strings, whitespace, and None values."""
        none_pool = ["", "   ", "\t\n", None, "  ", None]
        res = analyzer.compute_vun_metrics(none_pool)
        assert res["total_generated"] == len(none_pool)
        assert res["valid_count"] == 0
        assert res["validity"] == 0.0
        assert res["uniqueness"] == 0.0
        assert res["novelty"] == 0.0

    def test_zero_heavy_atom_molecules(self, analyzer: ResultAnalyzer) -> None:
        """Verify molecules with 0 heavy atoms (e.g. [H][H]) are not counted as valid drug leads."""
        zero_ha_pool = ["[H][H]", "[H+]", "[H-]"]
        res = analyzer.compute_vun_metrics(zero_ha_pool)
        assert res["total_generated"] == len(zero_ha_pool)
        assert res["valid_count"] == 0
        assert res["validity"] == 0.0
        assert res["uniqueness"] == 0.0
        assert res["novelty"] == 0.0


# ============================================================================
# Dimension 3: 100% Duplicate Molecules Pool Stress Tests
# ============================================================================

class TestDuplicateMoleculesPool:
    """Adversarial stress-testing of 100% duplicate molecule pools."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 20, 100])
    def test_n_identical_valid_molecules(self, analyzer: ResultAnalyzer, n: int) -> None:
        """Verify pool of N identical valid molecules yields Validity=100%, Uniqueness=1/N."""
        benzene_pool = ["c1ccccc1"] * n
        res = analyzer.compute_vun_metrics(benzene_pool)

        assert res["total_generated"] == n
        assert res["valid_count"] == n
        assert res["unique_valid_count"] == 1
        assert res["validity"] == 1.0
        assert res["validity_percentage"] == 100.0

        expected_uniqueness = 1.0 / n
        assert math.isclose(res["uniqueness"], expected_uniqueness, abs_tol=1e-5), (
            f"Uniqueness {res['uniqueness']} != expected {expected_uniqueness} for N={n}"
        )
        expected_pct = round(expected_uniqueness * 100.0, 2)
        assert res["uniqueness_percentage"] == expected_pct

    def test_multi_syntax_identical_structures(self, analyzer: ResultAnalyzer) -> None:
        """Verify syntactically distinct representations of the same molecule collapse to 1 unique valid."""
        # Benzene in aromatic, Kekule, explicit bonds, and atom-ordering permutations
        synonym_pool = [
            "c1ccccc1",
            "C1=CC=CC=C1",
            "c1c(cccc1)",
            "C=1C=CC=CC1",
            "c1ccc(cc1)",
        ]
        res = analyzer.compute_vun_metrics(synonym_pool)
        assert res["total_generated"] == 5
        assert res["valid_count"] == 5
        assert res["unique_valid_count"] == 1
        assert res["validity"] == 1.0
        assert math.isclose(res["uniqueness"], 1.0 / 5.0, abs_tol=1e-5)
        assert res["unique_canonical_smiles"] == ["c1ccccc1"]

    def test_stereoisomers_non_isomeric_canonicalization(self, analyzer: ResultAnalyzer) -> None:
        """Verify stereoisomers collapse under non-isomeric canonical SMILES (isomericSmiles=False)."""
        # (R)-butan-2-ol, (S)-butan-2-ol, and non-chiral butan-2-ol
        stereo_pool = [
            "C[C@H](O)CC",
            "C[C@@H](O)CC",
            "CCC(C)O",
        ]
        res = analyzer.compute_vun_metrics(stereo_pool)
        assert res["total_generated"] == 3
        assert res["valid_count"] == 3
        assert res["unique_valid_count"] == 1
        assert res["validity"] == 1.0
        assert math.isclose(res["uniqueness"], 1.0 / 3.0, abs_tol=1e-5)
        assert res["unique_canonical_smiles"] == ["CCC(C)O"]


# ============================================================================
# Dimension 4: Zero Novelty Against Reference Corpus
# ============================================================================

class TestZeroNoveltyAgainstReferenceCorpus:
    """Adversarial stress-testing of 0.0% Novelty against reference corpus."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_all_candidates_present_in_set_reference(self, analyzer: ResultAnalyzer) -> None:
        """Verify candidates all present in reference corpus set yield strictly 0.0% Novelty."""
        ref_corpus = {"c1ccccc1", "CCO", "CC(=O)O", "CCN", "CCC"}
        candidates = ["c1ccccc1", "CCO", "CC(=O)O"]

        res = analyzer.compute_vun_metrics(candidates, reference_corpus=ref_corpus)
        assert res["total_generated"] == 3
        assert res["valid_count"] == 3
        assert res["unique_valid_count"] == 3
        assert res["novel_count"] == 0, f"novel_count must be 0, got {res['novel_count']}"
        assert res["novelty"] == 0.0, f"novelty must be 0.0, got {res['novelty']}"
        assert res["novelty_percentage"] == 0.0

    def test_zero_novelty_with_duplicates(self, analyzer: ResultAnalyzer) -> None:
        """Verify duplicate candidates present in reference corpus yield 0.0% Novelty."""
        ref_corpus = {"c1ccccc1"}
        candidates = ["c1ccccc1"] * 10

        res = analyzer.compute_vun_metrics(candidates, reference_corpus=ref_corpus)
        assert res["unique_valid_count"] == 1
        assert res["novel_count"] == 0
        assert res["novelty"] == 0.0

    def test_zero_novelty_via_text_file_reference(self, analyzer: ResultAnalyzer, tmp_path: Path) -> None:
        """Verify reference corpus loaded from a text file enforces 0.0% Novelty."""
        ref_file = tmp_path / "ref_corpus.txt"
        ref_file.write_text("c1ccccc1\nCCO\nCC(=O)O\n", encoding="utf-8")

        candidates = ["c1ccccc1", "CCO"]
        res = analyzer.compute_vun_metrics(candidates, reference_corpus=ref_file)
        assert res["novel_count"] == 0
        assert res["novelty"] == 0.0

    def test_zero_novelty_via_parquet_reference(self, analyzer: ResultAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify reference corpus loaded from a parquet file enforces 0.0% Novelty when engine is available."""
        ref_parquet = tmp_path / "ref_corpus.parquet"
        ref_parquet.touch()

        # Mock pd.read_parquet since pyarrow/fastparquet may not be installed locally
        mock_df = pd.DataFrame({"SMILES": ["c1ccccc1", "CCO", "CC(=O)O"]})
        monkeypatch.setattr(pd, "read_parquet", lambda path, **kwargs: mock_df)

        candidates = ["c1ccccc1", "CC(=O)O"]
        res = analyzer.compute_vun_metrics(candidates, reference_corpus=ref_parquet)
        assert res["novel_count"] == 0
        assert res["novelty"] == 0.0

    def test_parquet_missing_engine_exception_handling(self, analyzer: ResultAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Adversarial stress test: when pd.read_parquet fails due to missing engine, compute_vun_metrics raises ImportError."""
        ref_parquet = tmp_path / "ref_corpus.parquet"
        ref_parquet.touch()

        def mock_read_fail(path: Any, **kwargs: Any) -> Any:
            raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'")

        monkeypatch.setattr(pd, "read_parquet", mock_read_fail)

        # In compute_vun_metrics, read_parquet is not wrapped in try/except (unlike get_reference_corpus)
        with pytest.raises(ImportError, match="Unable to find a usable engine"):
            analyzer.compute_vun_metrics(["c1ccccc1"], reference_corpus=ref_parquet)


# ============================================================================
# Dimension 5: 100% Novelty Against Reference Corpus
# ============================================================================

class TestHundredPercentNoveltyAgainstReferenceCorpus:
    """Adversarial stress-testing of 100.0% Novelty against reference corpus."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_all_candidates_absent_yields_100_percent_novelty(self, analyzer: ResultAnalyzer) -> None:
        """Verify candidates completely absent from reference corpus yield strictly 100.0% Novelty."""
        ref_corpus = {"c1ccccc1", "CCO"}
        # Compounds completely absent from reference: aspirin, paracetamol, caffeine
        novel_candidates = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Nc1ccc(O)cc1",
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        ]
        res = analyzer.compute_vun_metrics(novel_candidates, reference_corpus=ref_corpus)
        assert res["total_generated"] == 3
        assert res["valid_count"] == 3
        assert res["unique_valid_count"] == 3
        assert res["novel_count"] == 3, f"novel_count must be 3, got {res['novel_count']}"
        assert res["novelty"] == 1.0, f"novelty must be 1.0, got {res['novelty']}"
        assert res["novelty_percentage"] == 100.0

    def test_100_percent_novelty_with_empty_reference_corpus(self, analyzer: ResultAnalyzer) -> None:
        """Verify that when reference corpus is empty, all unique valid molecules are 100% novel."""
        empty_ref: set[str] = set()
        candidates = ["CCO", "CCC", "CCCC"]
        res = analyzer.compute_vun_metrics(candidates, reference_corpus=empty_ref)
        assert res["unique_valid_count"] == 3
        assert res["novel_count"] == 3
        assert res["novelty"] == 1.0
        assert res["novelty_percentage"] == 100.0


# ============================================================================
# Dimension 6: Single Candidate Edge Cases
# ============================================================================

class TestSingleCandidateEdgeCase:
    """Adversarial stress-testing of single candidate edge cases."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_single_valid_candidate(self, analyzer: ResultAnalyzer) -> None:
        """Verify single valid candidate yields 100% validity and 100% uniqueness."""
        res = analyzer.compute_vun_metrics(["c1ccccc1"])
        assert res["total_generated"] == 1
        assert res["valid_count"] == 1
        assert res["unique_valid_count"] == 1
        assert res["validity"] == 1.0
        assert res["validity_percentage"] == 100.0
        assert res["uniqueness"] == 1.0
        assert res["uniqueness_percentage"] == 100.0
        assert res["passed_validity"] is True
        assert res["passed_uniqueness"] is True

    def test_single_valid_candidate_novelty_bifurcation(self, analyzer: ResultAnalyzer) -> None:
        """Verify single candidate yields strictly 0.0% if present in ref, 100.0% if absent."""
        ref = {"c1ccccc1"}
        # Case A: Present in reference
        res_present = analyzer.compute_vun_metrics(["c1ccccc1"], reference_corpus=ref)
        assert res_present["novel_count"] == 0
        assert res_present["novelty"] == 0.0

        # Case B: Absent from reference
        res_absent = analyzer.compute_vun_metrics(["CCO"], reference_corpus=ref)
        assert res_absent["novel_count"] == 1
        assert res_absent["novelty"] == 1.0

    def test_single_invalid_candidate(self, analyzer: ResultAnalyzer) -> None:
        """Verify single invalid candidate yields 0% validity, 0% uniqueness, 0% novelty."""
        res = analyzer.compute_vun_metrics(["INVALID_STRUCTURE_STRING"])
        assert res["total_generated"] == 1
        assert res["valid_count"] == 0
        assert res["unique_valid_count"] == 0
        assert res["novel_count"] == 0
        assert res["validity"] == 0.0
        assert res["uniqueness"] == 0.0
        assert res["novelty"] == 0.0
        assert res["passed_validity"] is False
        assert res["passed_uniqueness"] is False


# ============================================================================
# Dimension 7: Unbiased Sampling Verification (drop_invalid=False, drop_duplicates=False)
# ============================================================================

class TestUnbiasedSamplingVerification:
    """Adversarial stress-testing of sample_candidates preserving invalid and duplicate molecules."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_sample_candidates_forces_drop_invalid_and_drop_duplicates_false(
        self,
        analyzer: ResultAnalyzer,
    ) -> None:
        """Verify sample_candidates forces drop_invalid=False and drop_duplicates=False even when overridden."""
        recorded_kwargs: dict[str, Any] = {}

        class MockGenerator:
            def generate(self, **kwargs: Any) -> pd.DataFrame:
                nonlocal recorded_kwargs
                recorded_kwargs = dict(kwargs)
                return pd.DataFrame({"SMILES": ["c1ccccc1", "INVALID", "c1ccccc1"]})

        gen = MockGenerator()

        # Intentionally attempt to pass drop_invalid=True, drop_duplicates=True
        df = analyzer.sample_candidates(
            generator=gen,
            num_samples=3,
            drop_invalid=True,  # Adversarial attempt to override
            drop_duplicates=True,  # Adversarial attempt to override
        )

        assert recorded_kwargs["drop_invalid"] is False, (
            "sample_candidates must strictly enforce drop_invalid=False"
        )
        assert recorded_kwargs["drop_duplicates"] is False, (
            "sample_candidates must strictly enforce drop_duplicates=False"
        )
        assert len(df) == 3, "DataFrame must preserve all 3 raw output rows"
        assert list(df["SMILES"]) == ["c1ccccc1", "INVALID", "c1ccccc1"]

    def test_sample_candidates_preserves_raw_distribution_metrics(
        self,
        analyzer: ResultAnalyzer,
    ) -> None:
        """Verify that running compute_vun_metrics on sampled DataFrame reflects unfiltered distribution."""
        raw_outputs = [
            "c1ccccc1",      # Valid 1
            "c1ccccc1",      # Duplicate of 1
            "NOT_A_SMILES",  # Invalid 1
            "CCO",           # Valid 2
            "CCO",           # Duplicate of 2
            "CCO",           # Duplicate of 2
            "INVALID_2",     # Invalid 2
            "CC(=O)O",       # Valid 3
        ]

        class RawDistributionGenerator:
            def generate(self, **kwargs: Any) -> pd.DataFrame:
                assert kwargs.get("drop_invalid") is False
                assert kwargs.get("drop_duplicates") is False
                return pd.DataFrame({"SMILES": raw_outputs, "Frags": ["c1ccccc1"] * len(raw_outputs)})

        gen = RawDistributionGenerator()
        df_sampled = analyzer.sample_candidates(generator=gen, num_samples=len(raw_outputs))

        # Check raw distribution preservation
        assert len(df_sampled) == 8

        # Run VUN calculation on sampled DataFrame directly
        vun = analyzer.compute_vun_metrics(df_sampled, reference_corpus={"c1ccccc1"})

        # Analysis:
        # Total = 8
        # Valid = 6 ("c1ccccc1" x 2, "CCO" x 3, "CC(=O)O" x 1)
        # Invalid = 2 ("NOT_A_SMILES", "INVALID_2")
        # Validity = 6 / 8 = 75%
        assert vun["total_generated"] == 8
        assert vun["valid_count"] == 6
        assert math.isclose(vun["validity"], 6.0 / 8.0, abs_tol=1e-5)
        assert vun["validity_percentage"] == 75.0

        # Unique Valid = 3 ("c1ccccc1", "CCO", "CC(=O)O")
        # Uniqueness = 3 / 6 = 50%
        assert vun["unique_valid_count"] == 3
        assert math.isclose(vun["uniqueness"], 3.0 / 6.0, abs_tol=1e-5)
        assert vun["uniqueness_percentage"] == 50.0

        # Novelty against {"c1ccccc1"}:
        # Unique valid not in ref: "CCO", "CC(=O)O" -> 2 / 3 = 66.67%
        assert vun["novel_count"] == 2
        assert math.isclose(vun["novelty"], 2.0 / 3.0, abs_tol=1e-5)
        assert vun["novelty_percentage"] == 66.67

    def test_sample_candidates_type_error_on_invalid_generator(self, analyzer: ResultAnalyzer) -> None:
        """Verify sample_candidates raises TypeError when object lacks generate() method."""
        with pytest.raises(TypeError, match="does not implement generate"):
            analyzer.sample_candidates(generator=object())


# ============================================================================
# Dimension 8: Scaffold Diversity & Large-Scale Mixture Stress Tests
# ============================================================================

class TestScaffoldDiversityAndLargeScaleMixture:
    """Adversarial stress-testing of scaffold diversity and large mixed candidate pools."""

    @pytest.fixture
    def analyzer(self, tmp_path: Path) -> ResultAnalyzer:
        return ResultAnalyzer(output_dir=tmp_path, seed_smiles=EGCG_SMILES)

    def test_scaffold_diversity_edge_cases(self, analyzer: ResultAnalyzer) -> None:
        """Verify extract_scaffold and compute_scaffold_diversity on acyclic, invalid, and empty sets."""
        # 1. Empty list
        div_empty = analyzer.compute_scaffold_diversity([])
        assert div_empty["diversity_ratio"] == 0.0
        assert div_empty["unique_scaffold_count"] == 0

        # 2. Acyclic molecules (ethanol, propane, butane)
        acyclic_mols = ["CCO", "CCC", "CCCC"]
        div_acyclic = analyzer.compute_scaffold_diversity(acyclic_mols)
        # All 3 have Murcko scaffold "[Acyclic]"
        assert div_acyclic["unique_scaffold_count"] == 1
        assert "[Acyclic]" in div_acyclic["scaffold_distribution"]
        # ResultAnalyzer rounds diversity_ratio to 4 decimal places (0.3333)
        assert math.isclose(div_acyclic["diversity_ratio"], 1.0 / 3.0, abs_tol=1e-4)

        # 3. Invalid SMILES strings
        assert analyzer.extract_scaffold("INVALID_MOL") == "[Invalid]"

    def test_large_scale_1000_candidate_mixture(self, analyzer: ResultAnalyzer) -> None:
        """Verify high-volume 1,000 candidate pool with exact mixture of valid, duplicate, and invalid."""
        # Construct deterministic mixture:
        # 250 invalid strings
        # 250 benzene duplicates ("c1ccccc1")
        # 250 toluene duplicates ("Cc1ccccc1" -> canonical Murcko scaffold c1ccccc1, unique smiles Cc1ccccc1)
        # 250 completely distinct valid molecules
        invalids = [f"INVALID_STRING_{i}" for i in range(250)]
        benzenes = ["c1ccccc1"] * 250
        toluenes = ["Cc1ccccc1"] * 250

        # Generate 250 distinct aliphatic alcohols: C(n)H(2n+1)OH
        distinct_valids = [f"{'C' * (i + 1)}O" for i in range(250)]

        pool = invalids + benzenes + toluenes + distinct_valids
        assert len(pool) == 1000

        # Reference corpus contains benzene and toluene, but none of the aliphatic alcohols
        ref_corpus = {"c1ccccc1", "Cc1ccccc1"}

        res = analyzer.compute_vun_metrics(pool, reference_corpus=ref_corpus)

        assert res["total_generated"] == 1000
        # Valid = 250 (benzene) + 250 (toluene) + 250 (alcohols) = 750
        assert res["valid_count"] == 750
        assert math.isclose(res["validity"], 750.0 / 1000.0, abs_tol=1e-6)
        assert res["validity_percentage"] == 75.0

        # Unique valid = 1 (benzene) + 1 (toluene) + 250 (alcohols) = 252
        assert res["unique_valid_count"] == 252
        expected_uniqueness = 252.0 / 750.0
        assert math.isclose(res["uniqueness"], expected_uniqueness, abs_tol=1e-6)
        assert res["uniqueness_percentage"] == round(expected_uniqueness * 100.0, 2)

        # Novel count = 252 - 2 (benzene, toluene in ref) = 250
        assert res["novel_count"] == 250
        expected_novelty = 250.0 / 252.0
        assert math.isclose(res["novelty"], expected_novelty, abs_tol=1e-6)
        assert res["novelty_percentage"] == round(expected_novelty * 100.0, 2)
