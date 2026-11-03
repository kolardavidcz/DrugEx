from __future__ import annotations

import contextlib
import io
import unittest
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum

from _david.code.config import PipelineConfig
from _david.code.environment import HarringtonDesirabilityReward, resolve_reward_scheme
from _david.code.schema import std


class TestSchemaStandards(unittest.TestCase):
    def test_schema_single_standard_naming(self) -> None:
        """Verify that every conceptual category in std contains only unique, canonical standard names."""
        reward_constants = [
            getattr(std, k) for k in dir(std) if k.startswith("REWARD_")
        ]
        self.assertEqual(
            len(reward_constants),
            len(set(reward_constants)),
            f"Duplicate reward constants detected in std: {reward_constants}",
        )

        column_constants = [
            getattr(std, k)
            for k in dir(std)
            if not k.startswith("_")
            and not k.startswith("REWARD_")
            and not k.startswith("VARIANT_")
            and not k.startswith("ROCS_")
            and isinstance(getattr(std, k), str)
        ]
        self.assertEqual(
            len(column_constants),
            len(set(column_constants)),
            f"Duplicate column constants detected in std: {column_constants}",
        )

    def test_reward_scheme_strict_canonical_names(self) -> None:
        """Verify that resolve_reward_scheme strictly accepts only canonical std constants."""
        self.assertIsInstance(resolve_reward_scheme(std.REWARD_PARETO), ParetoCrowdingDistance)
        self.assertIsInstance(resolve_reward_scheme(std.REWARD_WEIGHTED_SUM), WeightedSum)
        self.assertIsInstance(resolve_reward_scheme(std.REWARD_HARRINGTON), HarringtonDesirabilityReward)

        # Informal synonyms must be rejected (no loose string aliases)
        invalid_synonyms = [
            "pareto",
            "pareto_crowding_distance",
            "ws",
            "weighted",
            "desirability",
            "harrington_desirability",
        ]
        for syn in invalid_synonyms:
            with self.assertRaises(ValueError):
                resolve_reward_scheme(syn)

    def test_pipeline_config_canonical_validation(self) -> None:
        """Verify that PipelineConfig enforces schema standards and default 0.99 CPU ratio."""
        cfg = PipelineConfig()
        self.assertEqual(cfg.cpu_ratio, 0.99)
        self.assertEqual(cfg.reward_scheme, std.REWARD_PARETO)

        with self.assertRaises(ValueError):
            PipelineConfig(reward_scheme="pareto")

        with self.assertRaises(ValueError):
            PipelineConfig(reward_scheme="ws")

    def test_architectural_boundaries_no_kwargs_getters(self) -> None:
        """Assert PipelineConfig has no legacy kwargs getters."""
        cfg = PipelineConfig()
        for getter in ("preparer_kwargs", "rocs_scorer_kwargs", "explorer_kwargs", "fit_kwargs"):
            self.assertFalse(hasattr(cfg, getter), f"Legacy getter {getter} still exists on PipelineConfig")

    def test_production_scoring_is_canonical_tanimoto_combo_only(self) -> None:
        """Assert training cannot be pointed at an experimental similarity variant.

        The 162-cell asymmetric Tversky grid is a calibration result, not a runtime
        knob. Historical submission specs pass ``--score-variant tanimoto_combo`` and
        must keep replaying verbatim, but every other value has to fail loudly
        instead of being recorded in the manifest and then ignored by the scorer.
        """
        from _david.code.pipeline import parse_args
        from _david.code.rocs import create_rocs_scorer

        # Historical specs stay replayable.
        self.assertEqual(parse_args(["--score-variant", std.VARIANT_TANIMOTO_COMBO]).score_variant, std.VARIANT_TANIMOTO_COMBO)

        # Removed multi-metric knobs are gone from the config and the CLI.
        for legacy_field in ("rocs_tversky_alpha", "rocs_tversky_beta", "rocs_shape_weight", "rocs_color_weight"):
            self.assertNotIn(legacy_field, PipelineConfig.__dataclass_fields__)
        for flag in ("--tversky-alpha", "--tversky-beta", "--shape-weight", "--color-weight"):
            with self.assertRaises(SystemExit, msg=f"{flag} is still accepted by the CLI"), contextlib.redirect_stderr(io.StringIO()):
                parse_args([flag, "1.0"])

        # Any non-canonical variant is rejected by the CLI, the config and the factory.
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--score-variant", "tversky_scaffold_hop"])
        with self.assertRaises(ValueError):
            PipelineConfig(score_variant="tversky_scaffold_hop")
        with self.assertRaises(ValueError):
            create_rocs_scorer("CCO", n_jobs=1, score_variant="tversky_scaffold_hop")

    def test_architectural_boundaries_no_to_do_imports(self) -> None:
        """Assert production code never imports from _david.code.to_do."""
        from pathlib import Path
        code_dir = Path(__file__).resolve().parent.parent
        for py_file in code_dir.rglob("*.py"):
            if "to_do" in py_file.parts or "archive" in py_file.parts or py_file == Path(__file__):
                continue
            content = py_file.read_text(encoding="utf-8")
            self.assertNotIn(
                "import _david.code.to_do",
                content,
                f"Production file {py_file} violates boundary by importing from to_do",
            )
            self.assertNotIn(
                "from _david.code.to_do",
                content,
                f"Production file {py_file} violates boundary by importing from to_do",
            )


if __name__ == "__main__":
    unittest.main()
