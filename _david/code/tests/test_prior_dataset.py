"""Behaviour tests for the frozen prior dataset: the raw/scaled contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _david.code import prior_dataset as prior
from _david.code.config import PipelineConfig
from _david.code.schema import std


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            std.SMILES: ["CCO", "c1ccccc1", "not-a-smiles"],
            std.CANONICAL_SMILES: ["CCO", "c1ccccc1", None],
            prior.RAW_ROCS: [0.8, 1.6, 0.0],
            prior.RAW_SA: [4.0, 2.0, 10.0],
            std.QED: [0.5, 1.2, np.nan],
            std.VALID: [1, 1, 0],
        }
    )


class TestRawScaledContract(unittest.TestCase):
    def test_raw_rocs_passes_through_without_modifier(self) -> None:
        scaled = prior.raw_to_scaled(_frame(), PipelineConfig(rocs_use_modifier=False))
        np.testing.assert_allclose(scaled[std.ROCS].iloc[:2], [0.8, 1.6])

    def test_modifier_divides_by_the_clip_upper(self) -> None:
        config = PipelineConfig(rocs_use_modifier=True, rocs_clip_upper=2.0)
        scaled = prior.raw_to_scaled(_frame(), config)
        np.testing.assert_allclose(scaled[std.ROCS].iloc[:2], [0.4, 0.8])

    def test_sa_is_the_descending_clipped_window(self) -> None:
        scaled = prior.raw_to_scaled(_frame(), PipelineConfig())
        np.testing.assert_allclose(scaled[std.SA].iloc[:2], [0.5, 1.0])
        self.assertEqual(scaled[std.SA].iloc[2], 0.0)

    def test_qed_is_clipped_to_one(self) -> None:
        scaled = prior.raw_to_scaled(_frame(), PipelineConfig())
        self.assertEqual(scaled[std.QED].iloc[1], 1.0)

    def test_a_different_window_rescales_the_same_raw_data(self) -> None:
        config = PipelineConfig(sa_score_lower=4.0, sa_score_upper=2.0)
        scaled = prior.raw_to_scaled(_frame(), config)
        np.testing.assert_allclose(scaled[std.SA].iloc[:2], [0.0, 1.0])


class TestDescriptors(unittest.TestCase):
    def test_adds_canonical_valid_descriptors_and_scaffold(self) -> None:
        described = prior.add_descriptors(_frame()[[std.SMILES]])
        self.assertEqual(described[std.CANONICAL_SMILES].iloc[0], "CCO")
        self.assertEqual(described[std.VALID].tolist(), [1, 1, 0])
        self.assertAlmostEqual(float(described[std.MW].iloc[0]), 46.069, places=2)
        self.assertIsInstance(described[std.SCAFFOLD].iloc[1], str)


class TestDatasetIO(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        target = root / "toy_prior_3_s1"
        target.mkdir(parents=True)
        frame = _frame().drop(columns=[std.QED])
        frame[std.QED] = [0.5, 1.2, np.nan]
        frame.to_csv(target / prior.CSV_NAME, index=False, compression="gzip")
        manifest = {
            "schema_version": prior.SCHEMA_VERSION,
            "name": target.name,
            "seed": 1,
            "files": {
                prior.CSV_NAME: prior.sha256_file(target / prior.CSV_NAME),
            },
        }
        (target / prior.MANIFEST_NAME).write_text(json.dumps(manifest))
        return target

    def test_load_arm_rescales_for_the_loading_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write_dataset(Path(tmp))
            frame, manifest = prior.load_arm(
                target, PipelineConfig(rocs_use_modifier=True, sa_score_lower=4.0)
            )
            self.assertEqual(manifest["seed"], 1)
            np.testing.assert_allclose(frame[std.ROCS].iloc[:2], [0.4, 0.8])
            np.testing.assert_allclose(frame[std.SA].iloc[:2], [0.0, 1.0])

    def test_load_arm_refuses_a_short_dataset_instead_of_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write_dataset(Path(tmp))
            with self.assertRaises(ValueError):
                prior.load_arm(target, PipelineConfig(), n=10)

    def test_load_arm_rejects_a_missing_raw_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write_dataset(Path(tmp))
            frame = pd.read_csv(target / prior.CSV_NAME)
            frame.drop(columns=[prior.RAW_SA]).to_csv(
                target / prior.CSV_NAME, index=False, compression="gzip"
            )
            with self.assertRaises(ValueError):
                prior.load_arm(target, PipelineConfig())

    def test_verify_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write_dataset(Path(tmp))
            self.assertEqual(prior.verify(target), [])
            (target / prior.CSV_NAME).write_bytes(b"tampered")
            self.assertTrue(prior.verify(target))


class TestNaming(unittest.TestCase):
    def test_default_name_is_content_addressable(self) -> None:
        name = prior.default_name("CCO", 500, 7)
        self.assertTrue(name.startswith("mol_"))
        self.assertIn("_prior_500_s7", name)

    def test_egcg_default_gets_a_readable_tag(self) -> None:
        from _david.code.config import DEFAULT_MOLECULE

        self.assertEqual(prior.molecule_tag(DEFAULT_MOLECULE), "egcg")


if __name__ == "__main__":
    unittest.main()
