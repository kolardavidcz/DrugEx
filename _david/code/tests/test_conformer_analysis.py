"""Tests for the analysis layer: taxonomy, table loading, and figure provenance."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from _david.code.conformer.analysis import common, loaders
from _david.code.conformer.analysis.optimization_methods import (
    figure_reference_counts_vs_energy_window,
)
from _david.code.conformer.analysis.style import apply_style, save_figure
from _david.code.conformer.core.taxonomy import ISOMER_BIN_LABELS, bin_counts, isomer_bin


class TestIsomerTaxonomy(unittest.TestCase):
    def test_bin_boundaries_are_right_closed(self) -> None:
        cases = {
            1: "1–2",
            2: "1–2",
            3: "3–4",
            4: "3–4",
            5: "5–8",
            8: "5–8",
            9: "9–15",
            15: "9–15",
            16: "16–32",
            32: "16–32",
            33: "33+",
            512: "33+",
        }
        labels = isomer_bin(pd.Series(list(cases))).astype(str).tolist()
        self.assertEqual(labels, list(cases.values()))

    def test_zero_and_missing_values_are_not_silently_binned(self) -> None:
        bins = isomer_bin(pd.Series([0, None]))
        self.assertTrue(bins.isna().all())

    def test_bin_counts_keeps_empty_bins(self) -> None:
        counts = bin_counts(pd.Series([1, 1, 6, 128]))
        self.assertEqual(list(counts.index), list(ISOMER_BIN_LABELS))
        self.assertEqual(int(counts["1–2"]), 2)
        self.assertEqual(int(counts["9–15"]), 0)
        self.assertEqual(int(counts["33+"]), 1)

    def test_analysis_reexports_the_canonical_taxonomy(self) -> None:
        self.assertEqual(common.ISOMER_BIN_LABELS, ISOMER_BIN_LABELS)


class TestTableLoading(unittest.TestCase):
    def test_required_tables_resolve_to_canonical_sources(self) -> None:
        table = loaders.load("tversky_grid")
        self.assertTrue(table.canonical)
        self.assertTrue(table.path.is_file())
        self.assertEqual(len(table.digest or ""), 64)

    def test_missing_columns_fail_before_plotting(self) -> None:
        table = loaders.load("tversky_grid")
        with self.assertRaises(ValueError):
            loaders.require_columns(table, ("self_mean", "no_such_column"))

    def test_input_paths_map_names_to_paths(self) -> None:
        table = loaders.load("synthesis_grid")
        self.assertEqual(loaders.input_paths(table), {"synthesis_grid": table.path})


class TestFigureProvenance(unittest.TestCase):
    def test_save_figure_writes_a_manifest_with_digests(self) -> None:
        apply_style()
        figure, axis = plt.subplots(1, 1, figsize=(3, 2))
        axis.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            source.write_text("a,b\n1,2\n")
            target = save_figure(
                figure,
                Path(tmp) / "figure.png",
                inputs={"input": source},
                notes="unit test",
            )
            self.assertTrue(target.is_file())
            manifest = json.loads(
                target.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["figure"], "figure.png")
        self.assertEqual(manifest["notes"], "unit test")
        self.assertEqual(manifest["inputs"]["input"]["bytes"], 8)
        self.assertEqual(len(manifest["inputs"]["input"]["sha256"]), 64)
        self.assertFalse(plt.get_fignums())

    def test_energy_window_figure_renders_from_canonical_data(self) -> None:
        table = loaders.load("egcg_omnibus_grid")
        with tempfile.TemporaryDirectory() as tmp:
            path = figure_reference_counts_vs_energy_window(table, output_dir=tmp)
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_suffix(".manifest.json").is_file())

    def test_default_scorer_neighbourhood_covers_the_axes(self) -> None:
        from _david.code.conformer.analysis.color_shape import (
            NEIGHBOURHOOD,
            default_scorer_summary,
        )

        summary = default_scorer_summary(loaders.load("tversky_grid"))
        self.assertEqual(len(summary), len(NEIGHBOURHOOD))
        self.assertEqual(list(summary["role"]), [entry[3] for entry in NEIGHBOURHOOD])
        self.assertEqual(int((summary["role"] == "default").sum()), 1)
        # The neighbourhood must be one step away from the default, not a
        # shuffled sample: each row moves either the (alpha, beta) pair or the
        # shape weight, never both.
        default = summary[summary["role"] == "default"].iloc[0]
        for _, row in summary.iterrows():
            tversky_moved = any(
                abs(float(row[column]) - float(default[column])) > 1e-9
                for column in ("alpha", "beta")
            )
            weight_moved = abs(float(row["w_shape"]) - float(default["w_shape"])) > 1e-9
            self.assertFalse(tversky_moved and weight_moved, row["role"])
        for column in ("self_mean", "worst_decoy_p95", "fp_rate_worst_ge_035"):
            self.assertTrue(summary[column].between(0.0, 1.0).all(), column)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
