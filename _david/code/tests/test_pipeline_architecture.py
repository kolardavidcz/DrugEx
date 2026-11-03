from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from rdkit import Chem

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig
from _david.code.environment import create_environment
from _david.code.reference import PrepareMolecule
from _david.code.rocs import create_rocs_scorer, has_color, score_pair
from _david.code.schema import std


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

        env = create_environment(ref_sdf, config=cfg)
        self.assertEqual(len(env.scorers), 3)  # ROCS, SA, QED
        self.assertEqual(len(env.thresholds), 3)


if __name__ == "__main__":
    unittest.main()
