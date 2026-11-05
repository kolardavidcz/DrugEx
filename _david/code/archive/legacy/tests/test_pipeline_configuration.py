"""Check production chemistry wiring and scheduler resource boundaries."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from _david.code.config import PROJECT_ROOT, PipelineConfig
from _david.code.environment import PrepareMolecule, create_environment
from _david.code.pipeline import MoleculePipeline, config_from_args, parse_args
from _david.code.reference import PrepareMolecule as ReferencePreparer
from _david.code.metacentrum.grid import (
    GridJobSpec,
    MetaCentrumGridManager,
    generate_4hypothesis_grid,
    generate_8arm_grid,
)
from _david.code.utils import HardwareManager
from _david.code.conformer.sampling import plan_sampling
from rdkit import Chem


class TestPipelineConfiguration(unittest.TestCase):
    def test_sampling_reports_caps_and_uses_the_same_stereo_policy(self):
        mol = Chem.MolFromSmiles("FC(Cl)C(O)C")
        plan = plan_sampling(mol, isomer_limit=2, conformer_limit=5)
        self.assertEqual(len(plan.reference_isomers), 2)
        self.assertEqual(plan.total_conformer_attempts, 10)
        self.assertTrue(plan.truncated)
        self.assertTrue(plan.conformer_budget_capped)

    def test_explicit_profile_reaches_reference_and_rl(self):
        cfg = config_from_args(
            parse_args(
                [
                    "--optimization",
                    "MMFF94s",
                    "--dielectric",
                    "78.4",
                    "--minimize-candidates",
                    "--conformers",
                    "20",
                    "--reference-conformers",
                    "3",
                    "--reference-attempts",
                    "50",
                    "--reference-rmsd",
                    "0.5",
                    "--reference-seed",
                    "43",
                    "--conformer-threads",
                    "1",
                    "--epochs",
                    "200",
                    "--min-epochs",
                    "200",
                    "--patience",
                    "201",
                ]
            )
        )
        self.assertIs(PrepareMolecule, ReferencePreparer)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            pipeline = MoleculePipeline(cfg, output_dir=directory)
            self.assertEqual(pipeline.preparer.optimization, cfg.optimization)
            self.assertEqual(pipeline.preparer.random_seed, 43)
            self.assertEqual(pipeline.preparer.rmsd_cutoff, 0.5)
            self.assertEqual(pipeline.preparer.num_conformers, 50)
        with patch("_david.code.environment.create_rocs_scorer") as factory:
            create_environment("unused.sdf", config=cfg)
        self.assertEqual(factory.call_args.kwargs["optimization"], cfg.optimization)
        self.assertTrue(factory.call_args.kwargs["minimize_candidates"])
        self.assertEqual(factory.call_args.kwargs["max_conformers"], 20)
        self.assertEqual(cfg.optimization.method, "MMFF94s")
        self.assertEqual(cfg.optimization.dielectric, 78.4)
        self.assertEqual(cfg.min_epochs, 200)

    def test_defaults_reflect_calibrated_production_chemistry(self):
        cfg = config_from_args(parse_args([]))
        self.assertTrue(cfg.minimize_candidates)
        self.assertIsNotNone(cfg.optimization)
        self.assertEqual(cfg.optimization.method, "MMFF94s")
        self.assertEqual(cfg.optimization.dielectric, 78.4)
        self.assertEqual(cfg.max_conformers, 20)
        self.assertEqual(cfg.reference_attempts, 100)
        self.assertEqual(cfg.reference_rmsd_cutoff, 0.35)
        self.assertEqual(cfg.max_reference_conformers, 4)
        self.assertEqual(cfg.reference_selection_mode, "diverse")
        self.assertEqual(cfg.score_variant, "tanimoto_combo")
        self.assertEqual(cfg.epochs, 200)
        self.assertEqual(cfg.min_epochs, 50)
        self.assertEqual(cfg.patience, 50)
        self.assertEqual(cfg.epsilon, 0.05)
        self.assertFalse(cfg.use_epsilon_decay)
        self.assertFalse(cfg.include_qed)
        self.assertFalse(cfg.rocs_use_modifier)
        self.assertEqual(cfg.thresholds, (0.35, 0.50))
        self.assertEqual(cfg.weights, (0.75, 0.25))
        self.assertEqual(cfg.sa_score_lower, 5.0)
        self.assertEqual(cfg.sa_score_upper, 3.0)
        with self.assertRaises(ValueError):
            config_from_args(parse_args(["--dielectric", "78.4"]))

    def test_pbs_allocation_bounds_workers(self):
        with tempfile.NamedTemporaryFile(mode="w") as nodefile:
            nodefile.write("compute-node\n" * 8)
            nodefile.flush()
            with (
                patch.dict(
                    os.environ, {"PBS_JOBID": "123", "PBS_NODEFILE": nodefile.name}
                ),
                patch("os.sched_getaffinity", return_value=set(range(48))),
            ):
                self.assertEqual(HardwareManager.get_safe_parallel_jobs(ratio=1), 8)
                self.assertEqual(HardwareManager.get_safe_parallel_jobs(ratio=0.8), 6)

    def test_auto_gpu_prefers_second_gpu_when_all_gpus_are_free(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(
                HardwareManager,
                "get_device_vram",
                side_effect=lambda i: (0.0, 20000.0, 20000.0),
            ),
        ):
            self.assertEqual(HardwareManager.get_safe_gpus("auto"), (1,))
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch.object(
                HardwareManager,
                "get_device_vram",
                side_effect=lambda i: (0.0, 20000.0 if i == 0 else 2000.0, 20000.0),
            ),
        ):
            self.assertEqual(HardwareManager.get_safe_gpus("auto"), (0,))

    def test_grid_preserves_arguments_and_uses_actual_script_path(self):
        spec = GridJobSpec("egcg_test", "--molecule 'CC(O)C' --epochs 200")
        manager = MetaCentrumGridManager()
        with patch(
            "subprocess.run", return_value=SimpleNamespace(stdout="123.pbs-m1\n")
        ) as run:
            self.assertEqual(manager.submit_job(spec, "a" * 40), "123.pbs-m1")
        written = json.loads(run.call_args_list[0].kwargs["input"])
        self.assertEqual(written["args"][:2], ["--molecule", "CC(O)C"])
        command = run.call_args_list[1].args[0][-1]
        self.assertIn("/_david/code/metacentrum/job_template.sh", command)
        self.assertIn("spec=10.0", command)
        self.assertIn("ngpus=1", command)
        self.assertNotIn("gpu_cap=", command)
        self.assertNotIn("EXP_ARGS", command)
        self.assertEqual(
            PROJECT_ROOT / "_david/code/metacentrum/job_template.sh",
            Path(__file__).resolve().parents[1] / "code/metacentrum/job_template.sh",
        )

        # Also verify explicit gpu_cap is passed when provided
        spec_with_cap = GridJobSpec(
            "egcg_cap", "--epochs 10", gpu_cap="sm_80"
        )
        with patch(
            "subprocess.run", return_value=SimpleNamespace(stdout="124.pbs-m1\n")
        ) as run_cap:
            manager.submit_job(spec_with_cap, "a" * 40)
        command_cap = run_cap.call_args_list[1].args[0][-1]
        self.assertIn("gpu_cap=sm_80", command_cap)

    def test_grid_cli_submit_action(self):
        from _david.code.metacentrum.grid import main as grid_main

        with patch.object(
            MetaCentrumGridManager, "submit_job", return_value="999.pbs-m1"
        ) as mock_submit, patch(
            "sys.argv",
            [
                "grid.py",
                "submit",
                "--name",
                "cli_test_exp",
                "--no-sync",
                "--revision",
                "b" * 40,
                "--ncpus",
                "32",
                "--spec",
                "11.0",
            ],
        ):
            grid_main()
            mock_submit.assert_called_once()
            spec, revision = mock_submit.call_args[0]
            self.assertEqual(spec.experiment_name, "cli_test_exp")
            self.assertEqual(spec.n_cpus, 32)
            self.assertEqual(spec.spec, 11.0)
            self.assertEqual(revision, "b" * 40)

    def test_sa_clipped_score_and_rocs_modifier(self):
        from rdkit.Chem import AllChem
        from drugex.training.scorers.modifiers import ClippedScore

        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)

        # Default: rocs_use_modifier=False, SA modifier is ClippedScore(lower_x=5, upper_x=3)
        cfg_default = PipelineConfig()
        env = create_environment(mol, config=cfg_default)
        rocs_scorer, sa_scorer = env.scorers[0], env.scorers[1]
        self.assertIsNone(rocs_scorer.modifier)

        # Test SA ClippedScore(lower_x=5.0, upper_x=3.0) bounds
        self.assertAlmostEqual(float(sa_scorer.modifier(2.0)), 1.0)
        self.assertAlmostEqual(float(sa_scorer.modifier(3.0)), 1.0)
        self.assertAlmostEqual(float(sa_scorer.modifier(4.0)), 0.5)
        self.assertAlmostEqual(float(sa_scorer.modifier(5.0)), 0.0)
        self.assertAlmostEqual(float(sa_scorer.modifier(6.0)), 0.0)

        # When rocs_use_modifier=True
        cfg_mod = PipelineConfig(rocs_use_modifier=True)
        env_mod = create_environment(mol, config=cfg_mod)
        self.assertIsNotNone(env_mod.scorers[0].modifier)
        self.assertIsInstance(env_mod.scorers[0].modifier, ClippedScore)

    def test_qed_inclusion_and_objective_scaling(self):
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)

        # 2-objective: without QED
        cfg_noqed = PipelineConfig(include_qed=False)
        env_noqed = create_environment(mol, config=cfg_noqed)
        self.assertEqual(len(env_noqed.scorers), 2)
        self.assertEqual(len(env_noqed.thresholds), 2)
        self.assertEqual(env_noqed.thresholds, [0.35, 0.50])
        self.assertEqual(cfg_noqed.weights, (0.75, 0.25))

        # 3-objective: with QED
        cfg_qed = PipelineConfig(include_qed=True)
        env_qed = create_environment(mol, config=cfg_qed)
        self.assertEqual(len(env_qed.scorers), 3)
        self.assertEqual(len(env_qed.thresholds), 3)
        self.assertEqual(env_qed.thresholds, [0.35, 0.50, 0.50])
        self.assertEqual(cfg_qed.weights, (0.50, 0.25, 0.25))

    def test_generate_8arm_grid(self):
        specs = generate_8arm_grid(prefix="suite", molecule="CCO")
        self.assertEqual(len(specs), 8)
        names = {s.experiment_name for s in specs}
        self.assertEqual(len(names), 8)

        # Verify shared hyperparameters
        for s in specs:
            self.assertIn("--epochs 200", s.args)
            self.assertIn("--min-epochs 50", s.args)
            self.assertIn("--patience 50", s.args)
            self.assertIn("--epsilon 0.05", s.args)
            self.assertIn("--score-variant tanimoto_combo", s.args)
            self.assertIn("--molecule CCO", s.args)

        # Verify factor levels
        no_qed_specs = [s for s in specs if "--no-include-qed" in s.args]
        qed_specs = [s for s in specs if "--include-qed" in s.args]
        self.assertEqual(len(no_qed_specs), 4)
        self.assertEqual(len(qed_specs), 4)

        pareto_specs = [s for s in specs if "--reward-scheme pareto_crowding" in s.args]
        harrington_specs = [s for s in specs if "--reward-scheme harrington" in s.args]
        self.assertEqual(len(pareto_specs), 4)
        self.assertEqual(len(harrington_specs), 4)

        unmin_specs = [
            s
            for s in specs
            if "--no-minimize-candidates" in s.args and "--conformers 30" in s.args
        ]
        min_specs = [
            s
            for s in specs
            if "--minimize-candidates" in s.args and "--conformers 20" in s.args
        ]
        self.assertEqual(len(unmin_specs), 4)
        self.assertEqual(len(min_specs), 4)

    def test_4hypothesis_sprint_grid(self):
        specs = generate_4hypothesis_grid(prefix="sprint", molecule="CCO")
        self.assertEqual(len(specs), 4)
        names = [s.experiment_name for s in specs]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(
            names,
            [
                "sprint_h1_without_qed_pareto",
                "sprint_h2_without_qed_harrington",
                "sprint_h3_with_qed_pareto",
                "sprint_h4_with_qed_harrington",
            ],
        )

        # Verify shared Regime 2 hyperparameters
        for s in specs:
            self.assertIn("--epochs 200", s.args)
            self.assertIn("--min-epochs 50", s.args)
            self.assertIn("--patience 50", s.args)
            self.assertIn("--epsilon 0.05", s.args)
            self.assertIn("--score-variant tanimoto_combo", s.args)
            self.assertIn("--no-minimize-candidates", s.args)
            self.assertIn("--conformers 30", s.args)
            self.assertIn("--reference-conformers 1", s.args)
            self.assertIn("--molecule CCO", s.args)
            self.assertEqual(s.n_cpus, 16)
            self.assertEqual(s.mem_gb, 32)
            self.assertEqual(s.scratch_gb, 15)
            self.assertEqual(s.spec, 10.0)
            self.assertEqual(s.n_gpus, 1)
            self.assertEqual(s.walltime, "24:00:00")

        # Verify specific arms
        self.assertIn("--no-include-qed", specs[0].args)
        self.assertIn("--reward-scheme pareto_crowding", specs[0].args)

        self.assertIn("--no-include-qed", specs[1].args)
        self.assertIn("--reward-scheme harrington", specs[1].args)

        self.assertIn("--include-qed", specs[2].args)
        self.assertIn("--reward-scheme pareto_crowding", specs[2].args)

        self.assertIn("--include-qed", specs[3].args)
        self.assertIn("--reward-scheme harrington", specs[3].args)

        # Long prefix truncation
        long_specs = generate_4hypothesis_grid(prefix="x" * 60)
        self.assertEqual(len(long_specs), 4)
        for s in long_specs:
            self.assertLessEqual(len(s.experiment_name), 64)

    def test_grid_cli_actions(self):
        from _david.code.metacentrum.grid import main as grid_main

        with patch("sys.argv", ["grid.py", "grid", "--name", "test_run"]):
            grid_main()

        with patch("sys.argv", ["grid.py", "sprint", "--name", "test_sprint"]):
            grid_main()

        with (
            patch.object(
                MetaCentrumGridManager,
                "submit_matrix",
                return_value=["1.pbs", "2.pbs", "3.pbs", "4.pbs", "5.pbs", "6.pbs", "7.pbs", "8.pbs"],
            ) as mock_matrix,
            patch(
                "sys.argv",
                [
                    "grid.py",
                    "submit-grid",
                    "--name",
                    "matrix_test",
                    "--no-sync",
                    "--revision",
                    "c" * 40,
                ],
            ),
        ):
            grid_main()
            mock_matrix.assert_called_once()
            specs_submitted, rev = mock_matrix.call_args[0]
            self.assertEqual(len(specs_submitted), 8)
            self.assertEqual(rev, "c" * 40)

        with (
            patch.object(
                MetaCentrumGridManager,
                "submit_matrix",
                return_value=["h1.pbs", "h2.pbs", "h3.pbs", "h4.pbs"],
            ) as mock_sprint_matrix,
            patch(
                "sys.argv",
                [
                    "grid.py",
                    "submit-sprint",
                    "--name",
                    "sprint_test",
                    "--no-sync",
                    "--revision",
                    "d" * 40,
                ],
            ),
        ):
            grid_main()
            mock_sprint_matrix.assert_called_once()
            specs_submitted, rev = mock_sprint_matrix.call_args[0]
            self.assertEqual(len(specs_submitted), 4)
            self.assertEqual(rev, "d" * 40)

        with patch("sys.argv", ["grid.py", "grid"]):
            grid_main()

        with patch("sys.argv", ["grid.py", "sprint"]):
            grid_main()

        long_specs = generate_8arm_grid(prefix="a" * 50)
        self.assertEqual(len(long_specs), 8)
        for s in long_specs:
            self.assertLessEqual(len(s.experiment_name), 64)
            self.assertTrue(s.experiment_name.isalnum() or "_" in s.experiment_name)

    def test_harrington_desirability_reward_properties(self):
        import numpy as np
        from _david.code.environment import HarringtonDesirabilityReward

        # 2-objective evaluation
        h2 = HarringtonDesirabilityReward(weights=(0.75, 0.25), rocs_raw=True)
        scores2 = np.array([[1.2, 0.8], [0.4, 0.2], [0.0, 0.0], [2.0, 1.0]])
        smiles = ["CCO", "CCN", "CCC", "c1ccccc1"]
        r2 = h2(smiles, scores2)
        self.assertEqual(r2.shape, (4, 1))
        self.assertAlmostEqual(float(r2[1, 0]), 0.05, places=3)
        self.assertGreater(float(r2[3, 0]), 0.95)
        self.assertLess(float(r2[2, 0]), 0.01)

        # 3-objective evaluation
        h3 = HarringtonDesirabilityReward(weights=(0.50, 0.25, 0.25), rocs_raw=True)
        scores3 = np.array([[1.2, 0.8, 0.7], [0.4, 0.2, 0.3]])
        r3 = h3(smiles[:2], scores3)
        self.assertEqual(r3.shape, (2, 1))
        self.assertAlmostEqual(float(r3[1, 0]), 0.05, places=3)

        # Defensive weight padding when weights has fewer elements than objectives
        h_pad = HarringtonDesirabilityReward(weights=(0.80, 0.20), rocs_raw=True)
        r_pad = h_pad(["CCO"], np.array([[1.2, 0.8, 0.7]]))
        self.assertEqual(r_pad.shape, (1, 1))
        self.assertFalse(np.isnan(r_pad).any())

        # Empty input handling
        r_empty = h2([], np.zeros((0, 2)))
        self.assertEqual(r_empty.shape, (0, 1))

        # NaN / Inf robustness
        scores_nan = np.array([[np.nan, np.nan], [np.inf, -np.inf]])
        r_nan = h2(["CCO", "CCN"], scores_nan)
        self.assertFalse(np.isnan(r_nan).any())
        self.assertFalse(np.isinf(r_nan).any())

    def test_create_environment_defensive_dimensions(self):
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)

        # Pass 4 thresholds for 2-objective env: should defensively slice to 2
        env = create_environment(
            mol,
            include_qed=False,
            thresholds=(0.35, 0.50, 0.60, 0.70),
        )
        self.assertEqual(len(env.thresholds), 2)
        self.assertEqual(env.thresholds, [0.35, 0.50])

        # Pass 1 threshold for 2-objective env: should defensively pad to 2
        env_pad = create_environment(
            mol,
            include_qed=False,
            thresholds=(0.35,),
        )
        self.assertEqual(len(env_pad.thresholds), 2)
        self.assertEqual(env_pad.thresholds, [0.35, 0.50])

    def test_grid_kerberos_error_handling(self):
        mgr = MetaCentrumGridManager(remote_host="meta", remote_user="kolardavidcz")
        mock_res = SimpleNamespace(
            args=["ssh", "meta", "qstat"],
            returncode=255,
            stdout="",
            stderr="qstat: cannot connect to server pbs-m1.metacentrum.cz (errno=15010)",
        )
        with patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(RuntimeError) as ctx:
                mgr.get_queue_status()
            self.assertIn("15010", str(ctx.exception))
            self.assertIn("kinit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
