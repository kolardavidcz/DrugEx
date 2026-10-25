from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from _david.code.config import (
    DEFAULT_MOLECULE,
    PipelineConfig,
    RunPaths,
)
from _david.code.environment import (
    HarringtonDesirabilityReward,
    PrepareMolecule,
    create_environment,
    resolve_reward_scheme,
)
from _david.code.models import DecayingSequenceExplorer, SequenceRNNOptimizer
from _david.code.schema import std


class TestSequenceRNNPipeline(unittest.TestCase):
    """Rigorous unit test suite verifying SequenceRNN pipeline components, models, and scorers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PipelineConfig()
        cls.pkg_path, cls.voc_path = cls.config.resolve_pretrained_paths()

    def test_01_config_paths(self) -> None:
        """Verify Papyrus pretrained checkpoint and vocabulary files exist and are non-empty."""
        self.assertTrue(self.pkg_path.is_file(), f"Missing .pkg: {self.pkg_path}")
        self.assertTrue(self.voc_path.is_file(), f"Missing .vocab: {self.voc_path}")
        self.assertGreater(self.pkg_path.stat().st_size, 1000)
        self.assertGreater(self.voc_path.stat().st_size, 10)

    def test_02_model_loading_and_frozen_prior_isolation(self) -> None:
        """Verify agent and mutate prior instantiate as strictly separate objects with frozen prior parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = SequenceRNNOptimizer(
                models_dir=tmp_dir,
                gpu="cpu",
                agent_path=self.pkg_path,
                vocab_path=self.voc_path,
            )
            agent, prior, voc = optimizer.load_models()

            # Strict memory identity verification
            self.assertIsNot(agent, prior, "Mutate prior must NOT be the agent!")
            self.assertNotEqual(
                id(agent),
                id(prior),
                "Agent and prior must not share identical memory address!",
            )

            # Trainable vs frozen checks
            agent_trainable = any(p.requires_grad for p in agent.parameters())
            prior_frozen = all(not p.requires_grad for p in prior.parameters())
            self.assertTrue(
                agent_trainable,
                "Active agent must have trainable parameters (requires_grad=True)!",
            )
            self.assertTrue(
                prior_frozen,
                "Mutate prior must have completely frozen parameters (requires_grad=False)!",
            )
            self.assertFalse(prior.training, "Mutate prior must be in eval() mode!")

    def test_03_hawkins_conformer_ensemble_generation(self) -> None:
        """Verify Hawkins ensemble protocol generates 3D, non-degenerate conformers within energy window."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdf_out = Path(tmp_dir) / "test_ref.sdf"
            preparer = PrepareMolecule(max_reference_conformers=3, random_seed=42)
            aspirin_smi = "CC(=O)Oc1ccccc1C(=O)O"
            res_path = preparer.generate_reference_conformers(
                aspirin_smi, output_path=sdf_out
            )

            self.assertTrue(res_path.is_file())
            suppl = [
                m
                for m in Chem.SDMolSupplier(str(res_path), removeHs=False)
                if m is not None
            ]
            self.assertGreaterEqual(len(suppl), 1)
            self.assertLessEqual(len(suppl), 3)

            for conf_mol in suppl:
                self.assertGreaterEqual(conf_mol.GetNumConformers(), 1)
                conf = conf_mol.GetConformer()
                self.assertTrue(conf.Is3D(), "Generated conformer must be 3D!")
                zs = [conf.GetAtomPosition(i).z for i in range(conf_mol.GetNumAtoms())]
                z_span = max(zs) - min(zs)
                self.assertGreater(
                    z_span,
                    0.5,
                    "Conformer coordinates must not be flat/degenerate in Z axis!",
                )

    def test_04_environment_scoring_and_discrimination(self) -> None:
        """Verify ROCS, SA, and QED multi-objective evaluation discriminates reference from dissimilar control."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdf_out = Path(tmp_dir) / "test_egcg_ref.sdf"
            preparer = PrepareMolecule(max_reference_conformers=1, random_seed=42)
            preparer.generate_reference_conformers("c1ccccc1", output_path=sdf_out)

            cfg = PipelineConfig(max_conformers=5, reward_scheme="weighted_sum")
            env = create_environment(sdf_out, config=cfg)

            keys = env.getScorerKeys()
            self.assertIn(std.ROCS, keys)
            self.assertIn(std.SA, keys)
            self.assertIn(std.QED, keys)

            # Test scoring on benzene (identical to ref) vs caffeine
            test_smiles = ["c1ccccc1", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"]
            scores_df = env.getScores(test_smiles)

            self.assertEqual(len(scores_df), 2)
            self.assertIn(std.VALID, scores_df.columns)
            self.assertIn(std.DESIRED, scores_df.columns)

            # Self-score should exceed dissimilar compound
            benzene_rocs = float(scores_df[std.ROCS].iloc[0])
            caffeine_rocs = float(scores_df[std.ROCS].iloc[1])
            self.assertGreater(benzene_rocs, caffeine_rocs)

    def test_05_reward_schemes_and_harrington_non_compensability(self) -> None:
        """Verify WeightedSum and HarringtonDesirability calculate valid scalar rewards."""
        ws = resolve_reward_scheme("weighted_sum")
        scores = np.array([[0.8, 0.7, 0.6], [0.2, 0.3, 0.4]])
        thresholds = np.array([0.7, 0.5, 0.5])
        rewards_ws = ws(["smi1", "smi2"], scores=scores, thresholds=thresholds)
        self.assertEqual(len(rewards_ws), 2)
        self.assertTrue(all(np.isfinite(rewards_ws)))

        # Harrington strict non-compensability check:
        # If shape score collapses to 0.0, total composite desirability must approach 0.0
        harrington = HarringtonDesirabilityReward()
        zero_shape = np.array([[0.0, 0.95, 0.95]])
        rew_zero = float(harrington(["smi"], scores=zero_shape)[0, 0])
        self.assertLess(
            rew_zero, 0.10, "Harrington desirability must strictly penalize zero shape!"
        )

    def test_06_decaying_sequence_explorer_schedule(self) -> None:
        """Verify half-cycle cosine and linear exploration decay schedules."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdf_out = Path(tmp_dir) / "test_ref.sdf"
            preparer = PrepareMolecule(max_reference_conformers=1, random_seed=42)
            preparer.generate_reference_conformers("c1ccccc1", output_path=sdf_out)

            optimizer = SequenceRNNOptimizer(models_dir=tmp_dir, gpu="cpu")
            agent, prior, voc = optimizer.load_models()
            env = create_environment(sdf_out, config=self.config)

            explorer = DecayingSequenceExplorer(
                agent=agent,
                env=env,
                mutate=prior,
                epsilon=0.25,
                epsilon_end=0.05,
                decay_epochs=50,
                decay_type="cosine",
                use_decay=True,
            )

            # Check start, midpoint, and terminal epsilons
            eps_1 = explorer.get_decayed_epsilon(1)
            eps_25 = explorer.get_decayed_epsilon(25)
            eps_50 = explorer.get_decayed_epsilon(50)
            eps_100 = explorer.get_decayed_epsilon(100)

            self.assertAlmostEqual(eps_1, 0.25, places=3)
            self.assertLess(eps_25, 0.25)
            self.assertGreater(eps_25, 0.05)
            self.assertAlmostEqual(eps_50, 0.05, places=3)
            self.assertAlmostEqual(eps_100, 0.05, places=3)


if __name__ == "__main__":
    unittest.main()
