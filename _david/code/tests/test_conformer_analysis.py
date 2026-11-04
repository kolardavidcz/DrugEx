"""Tests for the analysis layer: taxonomy, table loading, and figure provenance."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from _david.code.conformer.analysis import loaders
from _david.code.conformer.analysis.style import apply_style, save_figure
from _david.code.conformer.core.taxonomy import ISOMER_BIN_LABELS, bin_counts, isomer_bin
from _david.code.tests._conformer_data import requires_canonical_data


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


class TestTableLoading(unittest.TestCase):
    @requires_canonical_data
    def test_required_tables_resolve_to_canonical_sources(self) -> None:
        table = loaders.load("tversky_grid")
        self.assertTrue(table.canonical)
        self.assertTrue(table.path.is_file())
        self.assertEqual(len(table.digest or ""), 64)

    @requires_canonical_data
    def test_missing_columns_fail_before_plotting(self) -> None:
        table = loaders.load("tversky_grid")
        with self.assertRaises(ValueError):
            loaders.require_columns(table, ("self_mean", "no_such_column"))

    @requires_canonical_data
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


class TestDefaultScorerSummary(unittest.TestCase):
    def test_summary_selects_one_row_per_requested_setting(self) -> None:
        from _david.code.conformer.analysis.color_shape import (
            NEIGHBOURHOOD,
            default_scorer_summary,
        )

        def row(alpha: float, beta: float, w_shape: float) -> dict[str, object]:
            return {
                "tversky_label": f"a{alpha:g}_b{beta:g}",
                "weight_label": f"{w_shape:g}S",
                "alpha": alpha,
                "beta": beta,
                "w_shape": w_shape,
            }

        requested = [
            row(alpha, beta, w_shape) for alpha, beta, w_shape, _ in NEIGHBOURHOOD
        ]
        grid = loaders.Table(
            name="synthetic",
            frame=pd.DataFrame([*requested, row(9.0, 9.0, 9.0)]),
            path=Path("synthetic.csv"),
            canonical=False,
        )
        summary = default_scorer_summary(grid)
        self.assertEqual(list(summary["role"]), [entry[3] for entry in NEIGHBOURHOOD])

        # A setting the grid does not contain is skipped, never invented.
        partial = default_scorer_summary(
            loaders.Table(
                name="synthetic",
                frame=pd.DataFrame(requested[1:]),
                path=Path("synthetic.csv"),
                canonical=False,
            )
        )
        self.assertEqual(
            list(partial["role"]), [entry[3] for entry in NEIGHBOURHOOD[1:]]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
