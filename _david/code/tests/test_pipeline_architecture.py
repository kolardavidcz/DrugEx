from __future__ import annotations

from datetime import datetime
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
from _david.code.environment import create_environment
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
    def test_pipeline_config_direct_access(self) -> None:
        """Verify PipelineConfig attributes are directly accessible without kwargs getters."""
        cfg = PipelineConfig(
            epochs=100,
            n_samples=250,
            epsilon=0.10,
            beta=0.05,
            max_conformers=15,
            max_isomers=2,
        )
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.n_samples, 250)
        self.assertEqual(cfg.epsilon, 0.10)
        self.assertEqual(cfg.beta, 0.05)
        self.assertEqual(cfg.max_conformers, 15)
        self.assertEqual(cfg.max_isomers, 2)
        self.assertEqual(cfg.cpu_ratio, 0.99)
        self.assertFalse(hasattr(cfg, "preparer_kwargs"))
        self.assertFalse(hasattr(cfg, "rocs_scorer_kwargs"))
        self.assertFalse(hasattr(cfg, "explorer_kwargs"))
        self.assertFalse(hasattr(cfg, "fit_kwargs"))

    def test_build_explorer_constructs_both_branches(self) -> None:
        """Both explorer branches must construct; Stage 2 dies otherwise.

        ``build_explorer`` previously passed ``config`` to ``SequenceExplorer``
        and ``batch_size``/``epsilon``/``beta``/``n_samples`` to
        ``DecayingSequenceExplorer``, so every run raised ``TypeError`` at the
        start of RL exploration while the rest of the suite stayed green.
        """
        import torch
        from drugex.training.explorers import SequenceExplorer

        from _david.code.models import DecayingSequenceExplorer, SequenceRNNOptimizer

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
        if ref_sdf.exists():
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

        cfg = PipelineConfig(
            max_conformers=5,
            max_isomers=2,
            include_qed=True,
            reward_scheme=std.REWARD_PARETO,
        )
        scorer = create_rocs_scorer(ref_sdf, cfg=cfg)
        self.assertEqual(scorer.getKey(), [std.ROCS])
        # Production colour typing is the OpenEye ImplicitMillsDean port, not fdef.
        self.assertEqual(cfg.color_ff, "oe_mills_dean")
        self.assertEqual(scorer.color_ff, "oe_mills_dean")
        self.assertEqual(scorer.score_type, "TanimotoCombo")
        # TanimotoCombo lives on [0, 2]; a 1.8 clip silently truncated the top decile.
        self.assertEqual(cfg.rocs_clip_upper, 2.0)

        env = create_environment(ref_sdf, config=cfg)
        self.assertEqual(len(env.scorers), 3)  # ROCS, SA, QED
        self.assertEqual(len(env.thresholds), 3)


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
