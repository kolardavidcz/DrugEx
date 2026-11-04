"""Tests for the conformer plumbing layer: contracts, cache, datasets, paths."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.chemistry import Optimization
from _david.code.conformer.core import datasets, paths, storage
from _david.code.conformer.core.cache import ArtifactCache, CacheStats
from _david.code.conformer.core.contracts import (
    ALGORITHM_VERSION,
    ArmSpec,
    EmbedSpec,
    MinimizeSpec,
    MoleculeSpec,
    ReferenceSpec,
    ScoreSpec,
    ScoreVariant,
    SeedPlan,
    SelectSpec,
    SelectionMode,
    StudySpec,
    arm,
    default_candidate_spec,
    default_reference_spec,
    fingerprint,
    jsonable,
)
from _david.code.conformer.core.errors import StudyError

from _david.code.tests._conformer_data import requires_canonical_data

class TestContracts(unittest.TestCase):
    def test_embed_spec_rejects_non_positive_attempts(self) -> None:
        with self.assertRaises(ValueError):
            EmbedSpec(attempts=0)
        with self.assertRaises(TypeError):
            EmbedSpec(attempts=True)  # type: ignore[arg-type]

    def test_embed_spec_defaults_to_a_single_identity(self) -> None:
        self.assertEqual(EmbedSpec().isomer_cap, 1)

    def test_default_reference_and_candidate_seeds_are_disjoint(self) -> None:
        """Reference and candidate RNG streams must never collide (42/43 bug)."""
        from _david.code.chemistry import verify_disjoint_conformer_seeds

        reference = ReferenceSpec().embed
        candidate = default_candidate_spec().embed
        verify_disjoint_conformer_seeds(
            reference.seed, reference.attempts, candidate.seed, candidate.attempts
        )

    def test_select_spec_cap_zero_means_keep_all(self) -> None:
        self.assertEqual(SelectSpec(cap=0).cap, 0)
        with self.assertRaises(ValueError):
            SelectSpec(cap=-1)
        with self.assertRaises(ValueError):
            SelectSpec(cap=2.5)  # type: ignore[arg-type]

    def test_score_spec_requires_positive_denominators(self) -> None:
        with self.assertRaises(ValueError):
            ScoreSpec(alpha=0.0, beta=0.0)
        with self.assertRaises(ValueError):
            ScoreSpec(shape_weight=0.0, color_weight=0.0)
        with self.assertRaises(ValueError):
            # TanimotoCombo is symmetric; an asymmetric request must not be
            # silently downgraded to the default metric.
            ScoreSpec(alpha=0.6, beta=1.4)
        spec = ScoreSpec(
            variant=ScoreVariant.TVERSKY,
            alpha=0.6,
            beta=1.4,
            shape_weight=0.3,
            color_weight=0.7,
        )
        self.assertIn("tversky", spec.label)

    def test_minimize_spec_label_distinguishes_policies(self) -> None:
        spec = MinimizeSpec(profile=Optimization("MMFF94s", 78.4), energy_window=8.0)
        raw = MinimizeSpec(profile=None, energy_window=None)
        self.assertTrue(spec.enabled)
        self.assertFalse(raw.enabled)
        self.assertEqual(raw.label, "raw")
        self.assertIn("mmff94s", spec.label)
        self.assertNotEqual(spec.label, MinimizeSpec(profile=Optimization("UFF", 1.0)).label)

    def test_study_spec_requires_unique_arm_names(self) -> None:
        molecule = MoleculeSpec(name="m", smiles="CCO")
        duplicate = ArmSpec(
            name="same",
            reference=default_reference_spec(),
            candidate=default_candidate_spec(),
        )
        with self.assertRaises(ValueError):
            StudySpec(
                name="s",
                molecules=(molecule,),
                arms=(duplicate, duplicate),
                output_dir=Path("tmp"),
            )

    def test_study_spec_rejects_empty_matrix(self) -> None:
        with self.assertRaises(ValueError):
            StudySpec(name="s", molecules=(), arms=(), output_dir=Path("tmp"))

    def test_seed_plan_requires_repeats(self) -> None:
        with self.assertRaises(ValueError):
            SeedPlan(repeats=())

    def test_arm_constructor_round_trips(self) -> None:
        spec = arm("demo", reference_cap=6, candidate_attempts=8)
        self.assertEqual(spec.reference.select.cap, 6)
        self.assertEqual(spec.candidate.embed.attempts, 8)
        self.assertEqual(spec.reference.select.mode, SelectionMode.DIVERSE)


class TestConformerSeeds(unittest.TestCase):
    def test_the_historical_multiplier_collision_is_detected(self) -> None:
        from _david.code.chemistry import verify_disjoint_conformer_seeds

        # RDKit effectively seeds conformer i with base * (i + 1), so 42 * 43 and
        # 43 * 42 are the same draw: the 1806 collision produced coordinate clones.
        with self.assertRaises(ValueError) as raised:
            verify_disjoint_conformer_seeds(42, 50, 43, 50)
        self.assertIn("overlap", str(raised.exception))

    def test_disjoint_budgets_pass(self) -> None:
        from _david.code.chemistry import verify_disjoint_conformer_seeds

        verify_disjoint_conformer_seeds(104729, 100, 42, 20)

    def test_zero_or_negative_budgets_are_rejected(self) -> None:
        from _david.code.chemistry import verify_disjoint_conformer_seeds

        for arguments in ((0, 10, 42, 10), (42, 10, -1, 10), (42, 0, 42, 10)):
            with self.assertRaises(ValueError, msg=str(arguments)):
                verify_disjoint_conformer_seeds(*arguments)

    def test_seed_budget_beyond_the_32bit_limit_is_rejected(self) -> None:
        from _david.code.chemistry import verify_disjoint_conformer_seeds

        with self.assertRaises(ValueError):
            verify_disjoint_conformer_seeds(2**30, 4, 42, 1)


class TestFingerprints(unittest.TestCase):
    def test_fingerprint_is_stable_and_order_independent_for_mappings(self) -> None:
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(fingerprint(left), fingerprint(right))
        self.assertNotEqual(fingerprint(left), fingerprint({"a": 1, "b": 3}))

    def test_jsonable_handles_enums_paths_and_dataclasses(self) -> None:
        payload = jsonable(
            {
                "mode": SelectionMode.BUTINA,
                "path": Path("a") / "b",
                "spec": SelectSpec(cap=2),
                "items": (1, 2),
            }
        )
        self.assertEqual(payload["mode"], "butina")
        self.assertEqual(payload["path"], "a/b")
        self.assertEqual(payload["items"], [1, 2])
        self.assertIsInstance(payload["spec"], dict)

    def test_contract_payloads_differ_when_a_field_changes(self) -> None:
        self.assertNotEqual(
            fingerprint(EmbedSpec(seed=1)), fingerprint(EmbedSpec(seed=2))
        )


class TestArtifactCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = ArtifactCache(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_json_round_trip_and_stats(self) -> None:
        key = self.cache.key("demo", {"a": 1})
        self.assertIsNone(self.cache.get_json("demo", key))
        self.cache.put_json("demo", key, {"value": 42}, metadata={"note": "x"})
        self.assertEqual(self.cache.get_json("demo", key), {"value": 42})
        self.assertEqual(self.cache.stats.hits, 1)
        self.assertEqual(self.cache.stats.misses, 1)
        self.assertEqual(self.cache.read_meta("demo", key)["metadata"], {"note": "x"})

    def test_keys_are_namespaced_and_versioned(self) -> None:
        payload = {"a": 1}
        self.assertEqual(
            self.cache.key("one", payload),
            ArtifactCache(self.root).key("one", payload),
        )
        self.assertNotEqual(self.cache.key("one", payload), self.cache.key("two", payload))
        other = ArtifactCache(self.root, algorithm_version="different")
        self.assertNotEqual(self.cache.key("one", payload), other.key("one", payload))
        self.assertEqual(other.algorithm_version, "different")
        self.assertEqual(self.cache.algorithm_version, ALGORITHM_VERSION)

    def test_same_key_serves_multiple_representations(self) -> None:
        key = self.cache.key("multi", {"molecule": "ethanol"})
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.cache.put_npz("multi", key, {"x": np.arange(3)})
        self.cache.put_sdf("multi", key, [mol])
        npz = self.cache.get_npz("multi", key)
        self.assertIsNotNone(npz)
        assert npz is not None
        np.testing.assert_array_equal(npz["x"], np.arange(3))
        sdf = self.cache.get_sdf("multi", key)
        self.assertIsNotNone(sdf)
        assert sdf is not None
        self.assertEqual(len(sdf), 1)

    def test_disabled_cache_never_writes(self) -> None:
        disabled = ArtifactCache(self.root / "disabled", enabled=False)
        key = disabled.key("demo", {"a": 1})
        self.assertIsNone(disabled.get_json("demo", key))
        written = disabled.put_json("demo", key, {"value": 1})
        self.assertFalse(written.exists())
        self.assertEqual(disabled.stats.disabled, 1)
        self.assertEqual(disabled.stats.hits, 0)
        self.assertFalse(disabled.has("demo", key, "json"))

    def test_cache_stats_hit_rate(self) -> None:
        stats = CacheStats(hits=3, misses=1, disabled=2)
        self.assertEqual(stats.lookups, 6)
        self.assertAlmostEqual(stats.hit_rate, 0.75)
        self.assertEqual(CacheStats().hit_rate, 0.0)

    def test_summary_records_root_and_version(self) -> None:
        summary = self.cache.summary()
        self.assertEqual(summary["algorithm"], ALGORITHM_VERSION)
        self.assertEqual(summary["root"], self.root.as_posix())

    def test_empty_payload_counts_as_a_miss(self) -> None:
        key = self.cache.key("demo", {"a": 2})
        path = self.cache.entry_path("demo", key, "json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self.assertIsNone(self.cache.get_json("demo", key))


class TestDatasets(unittest.TestCase):
    @requires_canonical_data
    def test_cell_table_keeps_scores_numeric(self) -> None:
        frame = datasets.load_table("drug_panel_cells")
        self.assertTrue(pd.api.types.is_float_dtype(frame["score"]))
        self.assertTrue(pd.api.types.is_integer_dtype(frame["reference_cap"]))

    @requires_canonical_data
    def test_declared_string_columns_stay_strings(self) -> None:
        frame = datasets.load_table("egcg_omnibus_grid")
        self.assertEqual(str(frame["energy_window"].dtype), "string")
        self.assertIn("inf", set(frame["energy_window"]))

    @requires_canonical_data
    def test_panel_loads_into_specs_with_split_tag(self) -> None:
        specs = datasets.load_panel("drug_panel_development")
        self.assertTrue(specs)
        self.assertEqual(len(specs), len({spec.name for spec in specs}))
        self.assertTrue(all(spec.smiles for spec in specs))
        self.assertTrue(any("split:development" in spec.tags for spec in specs))

    def test_unknown_table_and_panel_raise(self) -> None:
        with self.assertRaises(StudyError):
            datasets.load_table("no_such_table")
        with self.assertRaises(StudyError):
            datasets.load_panel("no_such_panel")

    def test_panel_round_trip(self) -> None:
        specs = (MoleculeSpec(name="ethanol", smiles="CCO", tags=("group:x",)),)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "panels").mkdir()
            written = datasets.write_panel("egcg", specs, root=Path(tmp, "panels"))
            self.assertTrue(written.is_file())
            loaded = datasets.load_panel("egcg", root=Path(tmp, "panels"))
        self.assertEqual(loaded[0].name, "ethanol")
        self.assertIn("group:x", loaded[0].tags)

    def test_deduplicate_specs_keeps_identical_duplicates(self) -> None:
        specs = (
            MoleculeSpec(name="glibenclamide", smiles="CCO"),
            MoleculeSpec(name="glibenclamide", smiles="CCO", tags=("split:confirmation",)),
            MoleculeSpec(name="egcg", smiles="C1CCCCC1"),
        )
        unique = datasets.deduplicate_specs(specs)
        self.assertEqual([spec.name for spec in unique], ["glibenclamide", "egcg"])

    def test_deduplicate_specs_rejects_a_structure_collision(self) -> None:
        specs = (
            MoleculeSpec(name="same-name", smiles="CCO"),
            MoleculeSpec(name="same-name", smiles="CCN"),
        )
        with self.assertRaises(StudyError):
            datasets.deduplicate_specs(specs)

    def test_legacy_fallback_is_marked_non_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp, "legacy.csv")
            storage.write_csv(legacy, pd.DataFrame({"value": [1, 2]}))
            source = datasets.resolve_table(
                "reference_counts",
                legacy={"reference_counts": legacy},
                root=tmp,
            )
            self.assertFalse(source.canonical)
            self.assertEqual(source.path, legacy)


class TestPaths(unittest.TestCase):
    def test_component_directories_are_distinct(self) -> None:
        values = list(paths.COMPONENT_DIRS.values())
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(values))

    def test_component_dir_resolves_under_output_root(self) -> None:
        target = paths.component_dir("clustering")
        self.assertEqual(target.parent, paths.OUTPUT_ROOT)
        self.assertEqual(target.name, paths.COMPONENT_DIRS["clustering"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
