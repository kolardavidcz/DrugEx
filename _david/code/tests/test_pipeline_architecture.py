from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import tempfile
import unittest
from rdkit import Chem

from _david.code.config import (
    DEFAULT_MOLECULE,
    PipelineConfig,
    RunPaths,
    free_run_directory,
    run_directory_name,
)
from _david.code.training.environment import HarringtonDesirabilityReward, create_environment
from _david.code.training.pipeline import config_from_args, parse_args
from _david.code.reference import PrepareMolecule
from _david.code.rocs import create_rocs_scorer, has_color, score_pair
from _david.code.schema import std


class TestRunDirectoryNaming(unittest.TestCase):
    """Runs must be findable by time, and two runs must never share a directory."""

    def test_default_name_is_minute_stamp_with_optional_tag(self) -> None:
        moment = datetime(2026, 11, 4, 15, 30, 12)
        self.assertEqual(run_directory_name(now=moment), "20261104_1530")
        self.assertEqual(
            run_directory_name("egcg gold/k=4", now=moment),
            "20261104_1530_egcg_gold_k_4",
        )

    def test_free_run_directory_never_reuses_a_minute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = free_run_directory(root, "20261104_1530_egcg")
            first.mkdir()
            second = free_run_directory(root, "20261104_1530_egcg")
            self.assertEqual(second.name, "20261104_1530_egcg_2")
            second.mkdir()
            self.assertEqual(
                free_run_directory(root, "20261104_1530_egcg").name,
                "20261104_1530_egcg_3",
            )

    def test_resolve_creates_the_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = RunPaths.resolve(
                output_root=tmp, tag="egcg", now=datetime(2026, 11, 4, 15, 30)
            )
            self.assertEqual(paths.root.name, "20261104_1530_egcg")
            for directory in (
                paths.models_dir,
                paths.raw_data_dir,
                paths.logs_dir,
                paths.figures_dir,
            ):
                self.assertTrue(directory.is_dir())
            again = RunPaths.resolve(
                output_root=tmp, tag="egcg", now=datetime(2026, 11, 4, 15, 30)
            )
            self.assertEqual(again.root.name, "20261104_1530_egcg_2")

    def test_explicit_base_dir_is_used_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom_run"
            paths = RunPaths.resolve(base_dir=target)
            self.assertEqual(paths.root, target)
            self.assertTrue(target.is_dir())


class TestPipelineArchitecture(unittest.TestCase):
    def test_build_explorer_constructs_both_branches(self) -> None:
        """Both explorer branches must construct; Stage 2 dies otherwise.

        ``build_explorer`` previously passed ``config`` to ``SequenceExplorer``
        and ``batch_size``/``epsilon``/``beta``/``n_samples`` to
        ``DecayingSequenceExplorer``, so every run raised ``TypeError`` at the
        start of RL exploration while the rest of the suite stayed green.
        """
        import torch
        from drugex.training.explorers import SequenceExplorer

        from _david.code.training.models import DecayingSequenceExplorer, SequenceRNNOptimizer

        def dummy_torch_model() -> torch.nn.Linear:
            return torch.nn.Linear(4, 4)

        plain_cfg = PipelineConfig(gpu="cpu", use_epsilon_decay=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = SequenceRNNOptimizer(models_dir=Path(tmpdir), config=plain_cfg)
            plain = optimizer.build_explorer(
                agent=dummy_torch_model(),
                env=object(),
                mutate=dummy_torch_model(),
                config=plain_cfg,
            )
            self.assertIs(type(plain), SequenceExplorer)
            self.assertEqual(plain.batchSize, plain_cfg.batch_size)
            self.assertEqual(plain.epsilon, plain_cfg.epsilon)
            self.assertEqual(plain.beta, plain_cfg.beta)
            self.assertEqual(plain.nSamples, plain_cfg.n_samples)

            decay_cfg = PipelineConfig(gpu="cpu", use_epsilon_decay=True)
            decayed = optimizer.build_explorer(
                agent=dummy_torch_model(),
                env=object(),
                mutate=dummy_torch_model(),
                config=decay_cfg,
            )
            self.assertIsInstance(decayed, DecayingSequenceExplorer)
            self.assertEqual(decayed.epsilon_start, decay_cfg.epsilon)
            self.assertEqual(decayed.epsilon_end, decay_cfg.epsilon_end)

    def test_has_color_and_score_pair(self) -> None:
        """Verify pharmacophore feature perception and 3D score_pair baseline."""
        egcg = Chem.MolFromSmiles(DEFAULT_MOLECULE)
        self.assertIsNotNone(egcg)
        self.assertTrue(has_color(egcg))

        ref_sdf = Path(__file__).parent / "data" / "EGCG_reference.sdf"
        if not ref_sdf.exists():
            self.skipTest("EGCG reference SDF not found")
        suppl = Chem.SDMolSupplier(str(ref_sdf), removeHs=False)
        mol = next(suppl)
        combo, shape, color = score_pair(mol, mol)
        self.assertGreaterEqual(combo, 1.95)
        self.assertGreaterEqual(shape, 0.95)
        self.assertGreaterEqual(color, 0.95)

    def test_prepare_molecule_with_config(self) -> None:
        """Verify PrepareMolecule directly consumes PipelineConfig."""
        cfg = PipelineConfig(
            reference_attempts=5,
            max_reference_conformers=2,
            energy_window_kcal=5.0,
        )
        preparer = PrepareMolecule(config=cfg)
        self.assertEqual(preparer.num_conformers, 5)
        self.assertEqual(preparer.max_reference_conformers, 2)
        self.assertEqual(preparer.energy_window_kcal, 5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "ref.sdf"
            out = preparer.generate_reference_conformers("c1ccccc1O", output_path=target)
            self.assertTrue(out.is_file())

    def test_create_rocs_scorer_and_environment(self) -> None:
        """Verify create_rocs_scorer and create_environment directly consume PipelineConfig."""
        ref_sdf = Path(__file__).parent / "data" / "EGCG_reference.sdf"
        if not ref_sdf.exists():
            self.skipTest("EGCG reference SDF not found")

        base = PipelineConfig(max_conformers=5, max_isomers=2)
        scoring = PipelineConfig(
            max_conformers=5,
            max_isomers=2,
            include_qed=True,
            reward_scheme=std.REWARD_PARETO,
        )
        scorer = create_rocs_scorer(ref_sdf, cfg=scoring)
        self.assertEqual(scorer.getKey(), [std.ROCS])
        self.assertEqual(scorer.score_type, "TanimotoCombo")

        without_qed = create_environment(ref_sdf, config=base)
        with_qed = create_environment(ref_sdf, config=scoring)
        self.assertEqual(len(with_qed.scorers), len(without_qed.scorers) + 1)
        self.assertEqual(
            list(with_qed.thresholds)[:-1], list(without_qed.thresholds)
        )
        self.assertEqual(len(with_qed.thresholds), len(with_qed.scorers))


class TestRewardWeightOverrides(unittest.TestCase):
    """``--weights`` must reach ``PipelineConfig`` and dominate the reward it names."""

    def test_cli_weights_override_the_auto_calibrated_pair(self) -> None:
        args = parse_args(["--weights", "0.95", "0.05"])
        self.assertEqual(tuple(args.weights), (0.95, 0.05))
        self.assertEqual(config_from_args(args).weights, (0.95, 0.05))

    def test_absent_weights_keep_the_calibrated_defaults(self) -> None:
        cfg = config_from_args(parse_args([]))
        self.assertEqual(cfg.weights, PipelineConfig().weights)

    def test_lopsided_weights_make_rocs_dominate_the_reward(self) -> None:
        scores = [[0.40, 1.0], [0.60, 1.0]]
        balanced = HarringtonDesirabilityReward(weights=(0.75, 0.25))(scores).ravel()
        rocs_only = HarringtonDesirabilityReward(weights=(0.95, 0.05))(scores).ravel()
        # Both rank the higher-ROCS molecule first; the lopsided pair is steeper.
        self.assertLess(balanced[0], balanced[1])
        self.assertLess(rocs_only[0], rocs_only[1])
        self.assertGreater(
            rocs_only[1] / rocs_only[0], balanced[1] / balanced[0]
        )


class TestBestEpochRestore(unittest.TestCase):
    """The evaluation arm must sample the checkpoint the run saved as best."""

    class _Agent:
        def __init__(self) -> None:
            self.loaded: str | None = None
            self.evaluated = False

        def loadStatesFromFile(self, path: str) -> None:  # noqa: N802 - core API name
            self.loaded = path

        def eval(self) -> None:  # noqa: A003 - core API name
            self.evaluated = True

    def test_the_saved_best_package_is_loaded_and_hashed(self) -> None:
        from _david.code.training.pipeline import restore_best_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            (models / "SequenceRNN_reinforced.pkg").write_bytes(b"weights")
            agent = self._Agent()
            provenance = restore_best_checkpoint(agent, models, 21)

        self.assertEqual(agent.loaded, str(models / "SequenceRNN_reinforced.pkg"))
        self.assertTrue(agent.evaluated)
        self.assertEqual(provenance["evaluation_source"], "best_epoch")
        self.assertEqual(provenance["best_epoch"], 21)
        self.assertEqual(
            provenance["best_checkpoint_sha256"],
            hashlib.sha256(b"weights").hexdigest(),
        )

    def test_no_package_means_the_last_epoch_is_evaluated(self) -> None:
        from _david.code.training.pipeline import restore_best_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            agent = self._Agent()
            provenance = restore_best_checkpoint(agent, Path(tmp), 21)

        self.assertIsNone(agent.loaded)
        self.assertEqual(provenance["evaluation_source"], "last_epoch")
        self.assertIsNone(provenance["best_epoch"])
        self.assertIsNone(provenance["best_checkpoint_sha256"])

    def test_a_package_without_a_recorded_epoch_is_not_claimed_as_best(self) -> None:
        from _david.code.training.pipeline import restore_best_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            (models / "SequenceRNN_reinforced.pkg").write_bytes(b"weights")
            agent = self._Agent()
            for epoch in (None, 0, -1):
                provenance = restore_best_checkpoint(agent, models, epoch)

        self.assertIsNone(agent.loaded)
        self.assertEqual(provenance["evaluation_source"], "last_epoch")
        self.assertIsNone(provenance["best_epoch"])


class TestRocsColourTyping(unittest.TestCase):
    """The ROCS colour channel must use the configured pharmacophore typing."""

    def test_invalid_color_ff_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(color_ff="basefeatures")

    def test_color_ff_override_reaches_the_scorer(self) -> None:
        ref_sdf = Path(__file__).parent / "data" / "EGCG_reference.sdf"
        if not ref_sdf.exists():
            self.skipTest("EGCG reference SDF not found")
        scorer = create_rocs_scorer(
            ref_sdf, cfg=PipelineConfig(max_conformers=2), color_ff="default"
        )
        self.assertEqual(scorer.color_ff, "default")


if __name__ == "__main__":
    unittest.main()
