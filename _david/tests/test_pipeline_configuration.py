"""Check production chemistry wiring and scheduler resource boundaries."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from _david.code.config import PROJECT_ROOT
from _david.code.environment import PrepareMolecule, create_environment
from _david.code.pipeline import MoleculePipeline, config_from_args, parse_args
from _david.code.reference import PrepareMolecule as ReferencePreparer
from _david.code.metacentrum.grid import GridJobSpec, MetaCentrumGridManager
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

    def test_defaults_keep_previous_chemistry(self):
        cfg = config_from_args(parse_args([]))
        self.assertFalse(cfg.minimize_candidates)
        self.assertIsNone(cfg.optimization)
        self.assertEqual(cfg.max_conformers, 30)
        self.assertEqual(cfg.reference_attempts, 100)
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
        self.assertIn("gpu_cap=compute_80", command)
        self.assertNotIn("EXP_ARGS", command)
        self.assertEqual(
            PROJECT_ROOT / "_david/code/metacentrum/job_template.sh",
            Path(__file__).resolve().parents[1] / "code/metacentrum/job_template.sh",
        )


if __name__ == "__main__":
    unittest.main()
