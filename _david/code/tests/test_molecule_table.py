"""Behaviour tests for the per-run molecule dataset cache."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from _david.code.analysis import visualization as viz
from _david.code.schema import std


def _monitor(epochs: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    rows = []
    smiles = ["CCO", "c1ccccc1", "not-a-smiles"]
    for epoch in epochs:
        for index, value in enumerate(smiles):
            rows.append(
                {
                    "Epoch": epoch,
                    std.SMILES: value,
                    std.ROCS: 0.4 + 0.1 * index + 0.01 * epoch,
                    std.DESIRED: index,
                }
            )
    return pd.DataFrame(rows)


class TestMoleculeTableCache(unittest.TestCase):
    def test_build_persists_table_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = viz.cached_molecule_table(tmp, _monitor(), "CCO")

            csv_path = viz.molecule_table_path(tmp)
            meta_path = csv_path.with_name(viz.MOLECULE_TABLE_META_NAME)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(meta_path.is_file())

            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta["schema_version"], viz.MOLECULE_TABLE_SCHEMA_VERSION)
            self.assertEqual(meta["rows"], len(table))
            self.assertIn("molecules_sha256", meta)
            self.assertIn("generated_at", meta)

            for column in ("MW", "HAC", "LogP", "TPSA", "Fsp3", "RotBonds"):
                self.assertIn(column, table.columns)
            self.assertEqual(int(table["ValidMol"].sum()), 4)
            self.assertTrue(table.loc[table["ValidMol"] == 0, "MW"].isna().all())

    def test_second_call_reuses_the_cached_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _monitor()
            first = viz.cached_molecule_table(tmp, monitor, "CCO")
            with mock.patch.object(
                viz, "epoch_molecule_table", side_effect=AssertionError("rebuilt")
            ):
                second = viz.cached_molecule_table(tmp, monitor, "CCO")

            pd.testing.assert_frame_equal(first, second, check_dtype=False)

    def test_changed_rocs_scale_invalidates_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _monitor()
            first = viz.cached_molecule_table(tmp, monitor, "CCO", rocs_scale=1.0)
            scaled = viz.cached_molecule_table(tmp, monitor, "CCO", rocs_scale=1.8)

            self.assertFalse(first["ROCS"].equals(scaled["ROCS"]))
            self.assertAlmostEqual(
                float(scaled["ROCS"].iloc[0]),
                float(first["ROCS"].iloc[0]) * 1.8,
            )

    def test_changed_molecules_invalidate_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = viz.cached_molecule_table(tmp, _monitor(), "CCO")
            changed = viz.cached_molecule_table(tmp, _monitor(epochs=(1, 2, 3)), "CCO")

            self.assertGreater(len(changed), len(first))
            meta = json.loads(
                viz.molecule_table_path(tmp)
                .with_name(viz.MOLECULE_TABLE_META_NAME)
                .read_text()
            )
            self.assertEqual(meta["rows"], len(changed))

    def test_refresh_rebuilds_even_when_provenance_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = _monitor()
            viz.cached_molecule_table(tmp, monitor, "CCO")
            with mock.patch.object(
                viz, "epoch_molecule_table", wraps=viz.epoch_molecule_table
            ) as rebuild:
                viz.cached_molecule_table(tmp, monitor, "CCO", refresh=True)
            rebuild.assert_called_once()

    def test_empty_monitor_returns_no_table_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = viz.cached_molecule_table(tmp, pd.DataFrame(), "CCO")
            self.assertTrue(table.empty)
            self.assertFalse(
                viz.molecule_table_path(tmp)
                .with_name(viz.MOLECULE_TABLE_META_NAME)
                .exists()
            )


if __name__ == "__main__":
    unittest.main()
