"""Behavioural tests: each pins a decision that was argued about, not a line of code."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import tempfile
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import FAILURE_VALUES, Property

from _david.code import run as runner
from _david.code.analysis import _helpers as analysis_helpers, figures, report as analysis
from _david.code.analysis import atlas_v2, collapse, diversity
from _david.code.config import PROJECT_ROOT, SCHEMES, Config
from _david.code.metacentrum import campaign, monitor, queues, submit as submit_module
from _david.code.metacentrum import job as job_module
from _david.code.metacentrum import memreport
from _david.code.metacentrum.submit import Job
from _david.code.reconstruct import provenance
from _david.code.scorers import conformers, objectives as scorers, rocs
from _david.code.scorers import utils as scorer_utils
from _david.code.training import StopCheckpointExplorer
from _david.code.run import (
    arm_is_complete,
    config_from_args,
    incomplete_arms,
    recorded_arm_size,
    write_arm,
)


def small(**changes) -> Config:
    cheap = {
        "max_conformers": 2,
        "max_isomers": 1,
        "reference_attempts": 4,
        "reference_conformers": 2,
        "n_jobs": 1,
    }
    return replace(Config(), **{**cheap, **changes})


class FailureValues(unittest.TestCase):

    def test_sa_failure_is_the_worst_value_not_the_best(self):
        mols = [Chem.MolFromSmiles("CCO"), None]
        raw = Property("SA").getScores(mols)
        self.assertAlmostEqual(raw[1], 10.0, msg="an unscorable SA must sit at the worst value")
        reward = Property("SA", modifier=ClippedScore(lower_x=5.0, upper_x=3.0))(mols)
        self.assertAlmostEqual(reward[1], 0.0, msg="the descending modifier must map it to 0")
        self.assertGreater(reward[0], reward[1], "ethanol must beat an unscorable molecule")

    def test_upward_properties_keep_zero_on_failure(self):
        for prop in ("QED", "MW", "logP"):
            self.assertEqual(FAILURE_VALUES.get(prop, 0.0), 0.0, prop)
            self.assertAlmostEqual(Property(prop).getScores([None])[0], 0.0, msg=prop)

    def test_unparameterized_energy_sorts_last_not_first(self):
        with patch.object(conformers, "_force_field", return_value=None):
            self.assertEqual(
                conformers.single_point_energy(Chem.MolFromSmiles("CCO"), small()), math.inf
            )

    def test_non_finite_energy_sorts_last(self):
        class NaNField:
            def CalcEnergy(self):
                return float("nan")

        with patch.object(conformers, "_force_field", return_value=NaNField()):
            self.assertEqual(
                conformers.single_point_energy(Chem.MolFromSmiles("CCO"), small()), math.inf
            )


class Thresholds(unittest.TestCase):
    def test_pareto_reward_ignores_thresholds(self):
        scheme = ParetoCrowdingDistance()
        values = np.array([[0.9, 0.2], [0.2, 0.9], [0.5, 0.5]])
        loose = scheme(["C", "CC", "CCC"], values, [0.0, 0.0])
        tight = scheme(["C", "CC", "CCC"], values, [9.9, 9.9])
        np.testing.assert_allclose(loose, tight)

    def test_invalid_rows_are_never_desired(self):
        frame = pd.DataFrame({"Valid": [1, 0], "ROCS": [0.5, 0.0], "SA": [0.5, 0.0]})
        marked = scorers.mark_desired(frame, ["ROCS", "SA"], [0.0, 0.0])
        self.assertEqual(marked["Desired"].tolist(), [1, 0])

    @staticmethod
    def _calibrate(cfg, frame, smiles):
        with (
            patch.object(DrugExEnvironment, "getScores", return_value=frame),
            patch.object(DrugExEnvironment, "getScorerKeys", return_value=["A", "B"]),
        ):
            return scorers.calibrate_thresholds(cfg, [None, None], smiles)

    def test_fixed_thresholds_bypass_the_percentile(self):
        frame = pd.DataFrame({"Valid": [1, 1], "A": [0.1, 2.0], "B": [0.1, 0.9]})
        thresholds, _ = self._calibrate(small(fixed_thresholds=(1.5, 0.25)), frame, ["C", "CC"])
        self.assertEqual(thresholds, [1.5, 0.25])

    def test_fixed_thresholds_must_match_the_objective_count(self):
        frame = pd.DataFrame({"Valid": [1], "A": [0.1], "B": [0.1]})
        with self.assertRaises(ValueError):
            self._calibrate(small(fixed_thresholds=(1.0,)), frame, ["C"])


class Metrics(unittest.TestCase):
    def test_desired_ratio_counts_a_molecule_once(self):
        frame = pd.DataFrame(
            {
                "SMILES": ["A", "A", "A", "B"],
                "Valid": [1] * 4,
                "Desired": [1, 1, 1, 0],
                "ROCS": [1.0] * 4,
            }
        )
        self.assertAlmostEqual(analysis._describe(frame, ["ROCS"])["desired"], 0.25)
        with tempfile.TemporaryDirectory() as directory:
            ratios = write_arm(
                Path(directory) / "arm.tsv",
                frame.drop(columns="SMILES"),
                ["A", "A", "A", "B"],
            )
        self.assertAlmostEqual(ratios["desired_ratio"], 0.25)
        self.assertAlmostEqual(ratios["unique_ratio"], 0.5)

    def test_empty_arm_reports_rather_than_raising(self):
        frame = pd.DataFrame({"SMILES": ["x"], "Valid": [0], "Desired": [0], "ROCS": [0.0]})
        self.assertEqual(analysis._table([], "arm"), "_none_")
        self.assertAlmostEqual(analysis._describe(frame, ["ROCS"])["desired"], 0.0)


class _FakeAgent:

    def __init__(self, ratios):
        self.ratios = list(ratios)
        self.epoch = 0

    def sample(self, n):
        return ["CCO"] * n

    def evaluate(self, smiles, evaluator=None, no_multifrag_smiles=True):
        index = min(self.epoch - 1, len(self.ratios) - 1)  # forward() already incremented
        desired = self.ratios[index]
        n = len(smiles)
        hits = int(round(desired * n))
        return pd.DataFrame(
            {
                "Valid": [1] * n,
                "Desired": [1] * hits + [0] * (n - hits),
                "ROCS": [0.5] * n,
            }
        )


class _FakeExplorer(StopCheckpointExplorer):

    def __init__(self, ratios, **kwargs):
        self.agent = _FakeAgent(ratios)
        self.env = None
        self.no_multifrag_smiles = True
        self.checkpoint_path = kwargs["checkpoint_path"]
        self.training_samples_path = None
        self.n_eval = 100  # a small arm would quantise hits/n_eval
        self.monitor = None

    def forward(self):
        self.agent.epoch += 1
        return ["CCO"], None

    def policy_gradient(self, smiles, seqs, epoch=None, epochs=None):
        return 1.0

    def getNovelMoleculeMetrics(self, scores):
        return {
            "desired_ratio": float(scores["Desired"].mean()),
            "avg_gmean": 0.5,
            "valid_ratio": 1.0,
        }

    def getModel(self):
        return {"epoch": self.agent.epoch}

    def logPerformanceAndCompounds(self, epoch, metrics, scores):
        pass


class Checkpoints(unittest.TestCase):

    def _run(self, ratios, directory, patience=1, **kwargs):
        path = Path(directory) / "at_stop.pkg"
        explorer = _FakeExplorer(ratios, checkpoint_path=path)
        summary = explorer.fit(epochs=len(ratios), patience=patience, min_epochs=1, **kwargs)
        return summary, path

    def test_the_checkpoint_is_the_policy_at_stop_not_the_best_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, path = self._run([1.0, 0.0, 0.0, 0.0], directory)
            self.assertEqual(summary["best_desired_ratio"], 1.0)
            self.assertEqual(summary["best_desired_ratio_epoch"], 1)
            self.assertTrue(summary["desired_ratio_frozen"])
            self.assertGreater(summary["desired_ratio_freeze_epoch"], 1)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(
                saved["epoch"],
                summary["desired_ratio_freeze_epoch"],
                "the shipped checkpoint must be the epoch training stopped on",
            )
            self.assertNotEqual(saved["epoch"], 1, "there is no best-epoch checkpoint any more")

    def test_a_single_checkpoint_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path = self._run([0.25, 0.5, 0.75, 1.0], directory)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), [path.name])

    def test_the_epoch_cap_keeps_the_live_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, path = self._run([0.25, 0.5, 0.75, 1.0], directory)
            self.assertFalse(summary["desired_ratio_frozen"])
            self.assertEqual(summary["desired_ratio_freeze_epoch"], 4)
            self.assertEqual(summary["epochs_run"], 4)
            self.assertEqual(summary["desired_ratio_at_stop"], 1.0)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 4)

    def test_a_plateau_ends_the_run_and_reports_the_live_value(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = self._run(
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
                directory,
                patience=2,
            )
        self.assertTrue(summary["desired_ratio_frozen"])
        self.assertEqual(summary["best_desired_ratio"], 0.9)
        self.assertEqual(summary["best_desired_ratio_epoch"], 1)
        self.assertEqual(summary["desired_ratio_at_stop"], 0.1)
        self.assertEqual(summary["epochs_run"], 4, "stop after patience + one epoch")

    def test_min_epochs_holds_the_run_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "at_stop.pkg"
            explorer = _FakeExplorer([0.0] * 6, checkpoint_path=path)
            summary = explorer.fit(epochs=6, patience=1, min_epochs=4)
        self.assertEqual(summary["epochs_run"], 4)
        self.assertTrue(summary["desired_ratio_frozen"])

    def test_the_summary_has_no_best_epoch_checkpoint_key(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = self._run([0.5, 0.5, 0.5], directory, patience=5)
        for gone in ("best_avg_gmean", "best_avg_gmean_epoch", "best_desired_ratio_before_freeze"):
            self.assertNotIn(gone, summary)
        self.assertEqual(set(summary["checkpoint_population"]), {"desired_ratio_at_stop"})


class Relaxation(unittest.TestCase):
    @staticmethod
    def _both_arms(cfg, directory):
        return (
            conformers.build_reference(replace(cfg, minimize=True), Path(directory) / "a.sdf"),
            conformers.build_reference(replace(cfg, minimize=False), Path(directory) / "b.sdf"),
        )

    def test_minimization_lowers_the_energy_of_the_reference_bank(self):
        cfg = small()
        with tempfile.TemporaryDirectory() as directory:
            relaxed, plain = self._both_arms(cfg, directory)
        best_relaxed = min(conformers.single_point_energy(mol, cfg) for mol in relaxed)
        best_plain = min(conformers.single_point_energy(mol, cfg) for mol in plain)
        self.assertLess(best_relaxed, best_plain, "relaxation must reduce MMFF energy")

    def test_bank_size_does_not_depend_on_the_arm(self):
        cfg = replace(small(), reference_attempts=8, reference_conformers=3)
        with tempfile.TemporaryDirectory() as directory:
            relaxed, plain = self._both_arms(cfg, directory)
        self.assertEqual(
            len(relaxed), len(plain), "the two arms must offer the same number of poses"
        )

    def test_parallel_embedding_is_identical_to_serial(self):
        smiles = [
            "CCO",
            "CCN",
            "CCC",
            "c1ccccc1O",
            "CC(=O)Nc1ccccc1",
            "CCOC(=O)C",
            "CN1CCCC1",
            "CC(C)Cc1ccccc1",
            "CCCCO",
        ]
        cfg = replace(small(), max_conformers=2, max_isomers=1, minimize=False)

        def embed(n_jobs):
            with tempfile.TemporaryDirectory() as directory:
                generator = conformers.RelaxingConformerGenerator(replace(cfg, n_jobs=n_jobs))
                path = Path(generator.genConformers(smiles, directory))
                out = {}
                for mol in scorer_utils.read_sdf(path):
                    out.setdefault(mol.GetProp("_Name"), []).append(
                        mol.GetConformer().GetPositions()
                    )
                return out

        serial, parallel = embed(1), embed(4)
        self.assertEqual(set(serial), set(parallel), "the molecule indices must match")
        self.assertTrue(serial, "the serial path produced nothing to compare")
        for name, poses in serial.items():
            np.testing.assert_allclose(
                np.array(poses), np.array(parallel[name]), atol=1e-12, err_msg=name
            )

    def test_force_field_is_mmff94s_and_never_falls_back(self):
        self.assertEqual(conformers.FORCE_FIELD, "MMFF94s")
        with patch.object(AllChem, "MMFFHasAllMoleculeParams", return_value=False):
            self.assertIsNone(conformers._force_field(Chem.MolFromSmiles("CCO"), small()))

    def test_unconverged_conformers_are_kept_not_dropped(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)

        class Unconverged:
            def Minimize(self, maxIts=0):
                return 1

            def CalcEnergy(self):
                return 12.5

        with patch.object(conformers, "_force_field", return_value=Unconverged()):
            out, reason = conformers._relax((mol, small(minimize=True), False))
        self.assertIsNotNone(out, "an unconverged conformer must survive")
        self.assertEqual(reason, "unconverged")

    def test_a_conformer_fault_never_ends_the_run(self):
        class Exploding:
            def Minimize(self, maxIts=0):
                raise ValueError("bad direction")

            def CalcEnergy(self):
                return 0.0

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        with patch.object(conformers, "_force_field", return_value=Exploding()):
            out, reason = conformers._relax((mol, small(), False))
        self.assertIsNotNone(out, "the molecule must survive a per-conformer fault")
        self.assertEqual(reason, "relax_raised")

    def test_unkekulizable_molecule_is_kept_not_filtered(self):
        self.assertIsNone(conformers._stereo_key(object()))

    def test_protonation_is_off_and_stated(self):
        self.assertFalse(conformers.PROTONATE)

    @staticmethod
    def _conformer():
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)CC"))
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        return mol

    def test_a_reference_conformer_that_changed_stereo_is_dropped(self):
        class Relaxed:
            def Minimize(self, maxIts=0):
                return 0

            def CalcEnergy(self):
                return 1.0

        with (
            patch.object(conformers, "_force_field", return_value=Relaxed()),
            patch.object(conformers, "_stereo_key", side_effect=["before", "after"]),
        ):
            kept, reason = conformers._relax((self._conformer(), small(), True))
        self.assertIsNone(kept)
        self.assertEqual(reason, "stereo_changed")

    def test_a_candidate_conformer_that_changed_stereo_is_kept(self):
        class Relaxed:
            def Minimize(self, maxIts=0):
                return 0

            def CalcEnergy(self):
                return 1.0

        with (
            patch.object(conformers, "_force_field", return_value=Relaxed()),
            patch.object(conformers, "_stereo_key", side_effect=["before", "after"]),
        ):
            kept, reason = conformers._relax((self._conformer(), small(), False))
        self.assertIsNotNone(kept)
        self.assertEqual(reason, "stereo_changed")

    def test_uncheckable_stereo_is_fatal_only_for_a_reference(self):
        class Relaxed:
            def Minimize(self, maxIts=0):
                return 0

            def CalcEnergy(self):
                return 1.0

        with (
            patch.object(conformers, "_force_field", return_value=Relaxed()),
            patch.object(conformers, "_stereo_key", return_value=None),
        ):
            self.assertEqual(
                conformers._relax((self._conformer(), small(), True))[1], "stereo_uncheckable"
            )
            kept, reason = conformers._relax((self._conformer(), small(), False))
        self.assertIsNotNone(kept)
        self.assertEqual(reason, "stereo_uncheckable")

    def test_a_reference_bank_holds_exactly_the_supplied_stereoisomer(self):
        cfg = replace(
            small(),
            reference_smiles="C[C@H](O)CC",
            reference_attempts=4,
            reference_conformers=2,
            minimize=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            bank = conformers.build_reference(cfg, Path(directory) / "ref.sdf")
        expected = conformers._smiles_stereo_key(cfg.reference_smiles)
        self.assertTrue(bank)
        self.assertEqual({conformers._stereo_key(mol) for mol in bank}, {expected})

    def test_a_reference_bank_refuses_to_ship_another_stereoisomer(self):
        cfg = replace(
            small(),
            reference_smiles="C[C@H](O)CC",
            reference_attempts=2,
            reference_conformers=1,
            minimize=False,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(conformers, "_smiles_stereo_key", return_value="expected"),
            patch.object(conformers, "_stereo_key", return_value="unexpected"),
            self.assertRaises(RuntimeError),
        ):
            conformers.build_reference(cfg, Path(directory) / "ref.sdf")


class Rocs(unittest.TestCase):

    def _bank(self, directory):
        cfg = replace(
            small(),
            reference_smiles="CCO",
            reference_attempts=3,
            reference_conformers=2,
            minimize=False,
        )
        path = Path(directory) / "ref.sdf"
        conformers.build_reference(cfg, path)
        return cfg, path

    def _scorer(self, cfg, path, **changes):
        return rocs.ComboROCSScorer(
            conformer_generator=conformers.RelaxingConformerGenerator(cfg),
            references=str(path),
            score_type="TanimotoCombo",
            color_ff=cfg.color_ff,
            n_jobs=1,
            show_progress=False,
            alignment_shape_weight=0.5,
            **changes,
        )

    @staticmethod
    def _two_objectives(scorer):
        return scorers.build_environment(
            [scorer, Property("SA")], [0.0, 0.0], scorers.reward_scheme("PRCD")
        )

    def test_matches_native_bit_for_bit_at_equal_weight(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

        with tempfile.TemporaryDirectory() as directory:
            cfg, path = self._bank(directory)
            mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1O")] + [None]
            native = RDKitROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg),
                references=str(path),
                score_type="TanimotoCombo",
                color_ff=cfg.color_ff,
                n_jobs=1,
                show_progress=False,
            )
            ours = self._scorer(cfg, path)
            np.testing.assert_allclose(ours.getScores(mols), native.getScores(mols), atol=0)
            self.assertEqual(ours.getKey(), native.getKey())

    def test_components_sum_to_the_reported_combo(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self._scorer(*self._bank(directory))
            smiles = ["CCO", "CCN"]
            scores = scorer.getScores([Chem.MolFromSmiles(s) for s in smiles]).ravel()
            for value, name in zip(scores, (Chem.CanonSmiles(s) for s in smiles)):
                shape, colour = scorer.components[name]
                self.assertAlmostEqual(shape + colour, float(value), places=9)

    def test_components_never_become_objectives(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self._scorer(*self._bank(directory))
            self.assertEqual(len(scorer.getKey()), 1)
            keys = self._two_objectives(scorer).getScorerKeys()
            self.assertEqual(len(keys), 2, f"the reward must see two objectives, saw {keys}")
            self.assertNotIn("ShapeTanimoto", keys)
            self.assertNotIn("ColorTanimoto", keys)

    def test_components_survive_the_environment_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self._scorer(*self._bank(directory))
            raw = ["C(C)O", "C(C)N"]  # deliberately non-canonical
            frame = self._two_objectives(scorer).getScores(raw)
            split = scorer.component_frame(raw)
            total = np.asarray(split["ShapeTanimoto"]) + np.asarray(split["ColorTanimoto"])
            combo = frame[scorer.getKey()[0]].to_numpy(dtype=float)
            np.testing.assert_allclose(total, combo, atol=1e-9)

    def test_alignment_weight_is_validated(self):
        with self.assertRaises(ValueError):
            replace(Config(), alignment_shape_weight=1.5)

    def test_malformed_align_result_costs_one_pose_not_the_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path = self._bank(directory)
            reference = next(m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m)
            with patch.object(rocs.rdShapeAlign, "AlignMol", return_value=None):
                self.assertEqual(rocs.pair_components(reference, reference, opt_param=0.5), (0.0, 0.0))


class Provenance(unittest.TestCase):
    def test_kernel_hashes_cover_every_module_a_score_depends_on(self):
        digests = provenance.kernel_fingerprint(PROJECT_ROOT)
        self.assertEqual(set(digests), set(provenance.KERNEL_FILES))
        for name, value in digests.items():
            self.assertNotEqual(value, "missing", name)
            self.assertEqual(len(value), 64, name)

    def test_source_digests_cover_this_package(self):
        digests = provenance.source_digests(provenance.PACKAGE_ROOT)
        for name in (
            "run.py",
            "config.py",
            "training.py",
            "reconstruct/provenance.py",
            "reconstruct/score_arms.py",
            "scorers/objectives.py",
            "scorers/conformers.py",
            "scorers/utils.py",
            "scorers/rocs.py",
            "archive/explorer.py",
            "analysis/report.py",
            "analysis/figures.py",
            "metacentrum/job.py",
            "metacentrum/submit.py",
            "metacentrum/monitor.py",
            "metacentrum/queues.py",
        ):
            self.assertIn(name, digests)

    def test_records_where_drugex_was_imported_from(self):
        described = provenance.describe(PROJECT_ROOT)
        self.assertEqual(
            Path(described["drugex_imported_from"]).resolve(),
            (Path(described["kernel_hashed_from"]) / "drugex").resolve(),
            "the hashed kernel is not the one Python imported",
        )

    def test_manifest_states_are_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = provenance.Manifest(path, run="x")
            self.assertEqual(json.loads(path.read_text())["state"], "running")
            manifest.update(state="failed", error="boom")
            self.assertEqual(json.loads(path.read_text())["state"], "failed")
            self.assertFalse(list(Path(directory).glob("*.tmp")), "no temporary left behind")


class Configuration(unittest.TestCase):
    def test_reversed_sa_clip_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(Config(), sa_reward_zero=3.0, sa_reward_one=5.0)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(SystemExit):
            config_from_args(["not_a_field=1"])

    def test_override_round_trips_through_the_declared_type(self):
        cfg = config_from_args(["epochs=3", "min_epochs=1", "minimize=False"])
        self.assertEqual(cfg.epochs, 3)
        self.assertIs(cfg.minimize, False)

    def test_tuple_fields_are_settable_from_the_command_line(self):
        self.assertEqual(
            config_from_args(["fixed_thresholds=0.8,1.0"]).fixed_thresholds, (0.8, 1.0)
        )
        self.assertEqual(
            config_from_args(["fixed_thresholds=(0.8, 1.0)"]).fixed_thresholds, (0.8, 1.0)
        )
        self.assertEqual(config_from_args(["fixed_thresholds="]).fixed_thresholds, ())

    def test_min_epochs_cannot_exceed_epochs(self):
        with self.assertRaises(ValueError):
            replace(Config(), epochs=3, min_epochs=50)


class RewardSchemes(unittest.TestCase):

    def test_every_scheme_name_is_one_of_drugexs_own(self):
        self.assertIsInstance(scorers.reward_scheme("PRCD"), ParetoCrowdingDistance)
        self.assertIsInstance(scorers.reward_scheme("WS"), WeightedSum)
        self.assertEqual(sorted(SCHEMES), ["PRCD", "PRTD", "WS"])

    def test_an_unknown_scheme_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as caught:
            scorers.reward_scheme("pareto")
        self.assertIn("PRCD", str(caught.exception))

    def test_weighted_sum_does_read_the_thresholds(self):
        scheme = scorers.reward_scheme("WS")
        values = np.array([[0.9, 0.2], [0.2, 0.9], [0.5, 0.5]])
        loose = scheme(["C", "CC", "CCC"], values, np.array([0.0, 0.0]))
        tight = scheme(["C", "CC", "CCC"], values, np.array([0.8, 0.1]))
        self.assertFalse(np.allclose(loose, tight), "WS must respond to its thresholds")

    def test_every_objective_reaches_the_reward_on_the_same_scale(self):
        for scheme in ("WS", "PRCD", "PRTD"):
            cfg = small(scheme=scheme)
            rocs = scorers.build_scorers(cfg, [Chem.MolFromSmiles("c1ccccc1")])[0]
            self.assertIsNotNone(rocs.getModifier(), scheme)
        modifier = scorers.build_scorers(
            small(scheme="WS"), [Chem.MolFromSmiles("c1ccccc1")]
        )[0].getModifier()
        np.testing.assert_allclose(
            modifier(np.array([0.0, 1.0, 2.0])), np.array([0.0, 0.5, 1.0]), atol=1e-9
        )

    def test_every_table_reports_rocs_on_zero_to_two_whatever_it_stored(self):
        raw = pd.DataFrame(
            {
                "SMILES": ["c1ccccc1", "CCO"],
                "Valid": [1, 1],
                "RDKit_Aggregate_1refs_TanimotoCombo": [1.4, 0.8],
                "ShapeTanimoto": [0.9, 0.5],
                "ColorTanimoto": [0.5, 0.3],
            }
        )
        modified = raw.assign(RDKit_Aggregate_1refs_TanimotoCombo=[0.7, 0.4])
        col = "RDKit_Aggregate_1refs_TanimotoCombo"
        self.assertEqual(analysis_helpers.rocs_scale({}), 1.0)
        self.assertEqual(analysis_helpers.rocs_scale({"rocs_reward_scale": 2.0}), 2.0)
        self.assertEqual(analysis_helpers.rocs_scale(None), 1.0)
        for frame, config in ((raw, {}), (modified, {"rocs_reward_scale": 2.0})):
            enriched = analysis_helpers.enrich_descriptors(frame, col, "", config)
            self.assertEqual(list(enriched["ROCS"]), [1.4, 0.8])
        # A per-epoch table has no components and must still come back on [0, 2].
        epochs = modified[["SMILES", "Valid", col]]
        enriched = analysis_helpers.enrich_descriptors(epochs, col, "", {"rocs_reward_scale": 2.0})
        self.assertEqual(list(enriched["ROCS"]), [1.4, 0.8])

    def test_the_environment_uses_the_scheme_it_was_given(self):
        env = scorers.build_environment([Property("SA")], [0.0], scorers.reward_scheme("WS"))
        self.assertIsInstance(env.rewardScheme, WeightedSum)

    def test_the_report_never_claims_ws_keeps_thresholds_out_of_the_reward(self):
        config = {"threshold_percentile": 85.0, "n_prior": 10, "threshold_from_reference": []}
        pareto = analysis._threshold_prose({**config, "scheme": "PRCD"}, {}, ["ROCS"], [0.5])
        weighted = analysis._threshold_prose({**config, "scheme": "WS"}, {}, ["ROCS"], [0.5])
        self.assertIn("stop condition only", pareto)
        self.assertIn("part of the reward", weighted)
        self.assertNotIn("stop condition only", weighted)

    def test_a_saturated_prior_does_not_turn_ws_report_prose_into_pareto(self):
        config = {
            "scheme": "WS",
            "threshold_percentile": 85.0,
            "n_prior": 2,
            "threshold_from_reference": [],
            "sa_reward_one": 3.0,
        }
        prior = pd.DataFrame({"SA_raw": [2.5, 2.6], "Valid": [1, 1]})
        prose = analysis._threshold_prose(config, {"prior": prior}, ["SA"], [0.5])
        self.assertIn("WeightedSum", prose)
        self.assertNotIn("Pareto reward is effectively the ROCS", prose)
        self.assertNotIn("stop condition only", prose)


class ScoringBatches(unittest.TestCase):

    def _policy_gradient(self, molecules: int, batch_size: int) -> tuple[list[int], int]:
        from drugex.training.explorers.sequence_explorer import SequenceExplorer

        scored, steps = [], []

        class _Env:
            def getRewards(self, smiles, frags=None):
                scored.append(len(smiles))
                return np.ones((len(smiles), 1))

        class _Agent:
            def __init__(self):
                self.w = torch.nn.Parameter(torch.zeros(1))

            def likelihood(self, seq):  # native returns (batch, sequence length)
                return self.w * torch.ones(seq.shape, dtype=torch.float32)

        agent = _Agent()
        stub = types.SimpleNamespace(
            env=_Env(),
            agent=agent,
            batchSize=batch_size,
            beta=0.0,
            device="cpu",
            optim=torch.optim.SGD([agent.w], lr=0.0),
            monitor=types.SimpleNamespace(saveProgress=lambda *a, **k: steps.append(1)),
        )
        SequenceExplorer.policy_gradient(
            stub, [f"C{'C' * i}" for i in range(molecules)], torch.zeros(molecules, 4).long()
        )
        return scored, len(steps)

    def test_batch_size_splits_the_gradient_not_the_scoring(self):
        for batch_size, expected_steps in ((16, 32), (128, 4), (500, 1)):
            scored, steps = self._policy_gradient(500, batch_size)
            self.assertEqual(scored, [500], f"batch_size={batch_size} must not chunk scoring")
            self.assertEqual(steps, expected_steps, f"batch_size={batch_size} splits the gradient")


class _WedgedPool:

    def __init__(self):
        self.terminated = False

    def get_start_method(self):
        return "forkserver"

    def Pool(self, workers):
        return self

    def map_async(self, function, items):
        return self

    def get(self, timeout=None):
        raise mp.TimeoutError()

    def terminate(self):
        self.terminated = True

    def join(self):
        pass


class StartMethod(unittest.TestCase):

    def test_a_working_context_is_reported_by_its_name(self):
        with patch.dict("os.environ", {"DRUGEX_MP_CONTEXT": "fork"}):
            self.assertEqual(scorer_utils.check_pool(timeout=30.0), "fork")

    def test_a_pool_that_cannot_be_created_is_reported_not_raised(self):
        class _Dead:
            def get_start_method(self):
                return "forkserver"

            def Pool(self, workers):
                raise OSError(24, "Too many open files")

        with (
            patch.object(scorer_utils, "_POOL_BROKEN", False),
            patch.object(scorer_utils, "pool_context", return_value=_Dead()),
        ):
            self.assertIn("broken", scorer_utils.check_pool(timeout=1.0))

    def test_a_pool_that_hangs_is_bounded_and_terminated(self):
        wedged = _WedgedPool()
        started = time.perf_counter()
        with (
            patch.object(scorer_utils, "_POOL_BROKEN", False),
            patch.object(scorer_utils, "pool_context", return_value=wedged),
        ):
            result = scorer_utils.check_pool(timeout=0.2)
        self.assertIn("broken", result)
        self.assertTrue(wedged.terminated, "a wedged pool must be terminated, not left running")
        self.assertLess(time.perf_counter() - started, 10.0)

    def test_a_broken_probe_disables_production_pools(self):
        with (
            patch.object(scorer_utils, "_POOL_BROKEN", False),
            patch.object(scorer_utils, "pool_context", return_value=_WedgedPool()),
        ):
            self.assertIn("broken", scorer_utils.check_pool(timeout=0.01))
            self.assertFalse(scorer_utils.pool_is_healthy())
            # Forty records would pool, and the wedged stub has no map().
            self.assertEqual(
                scorer_utils.map_tasks(scorer_utils._touch, list(range(40)), small(n_jobs=2)),
                list(range(40)),
            )

    def test_a_multi_core_run_refuses_to_start_without_workers(self):
        with patch.object(scorer_utils, "_POOL_BROKEN", True):
            self.assertIsNone(
                runner.serial_run_refusal(small(n_jobs=1)),
                "a one-core run is serial anyway and must still run",
            )
            refusal = runner.serial_run_refusal(small(n_jobs=16))
            self.assertIsNotNone(refusal, "16 cores and no workers cannot finish a run")
            self.assertIn("DRUGEX_MP_CONTEXT=fork", refusal)
        with patch.object(scorer_utils, "_POOL_BROKEN", False):
            self.assertIsNone(runner.serial_run_refusal(small(n_jobs=16)))


class ArmCompleteness(unittest.TestCase):
    @staticmethod
    def _arm_frame(rows: int) -> pd.DataFrame:
        molecules = ["CC", "CCC", "CCCC"][:rows]
        return pd.DataFrame(
            {
                "SMILES": molecules,
                "Valid": [1] * rows,
                "ROCS": [0.5] * rows,
                "SA": [1.0] * rows,
                "Desired": [0] * rows,
                "ShapeTanimoto": [0.4] * rows,
                "ColorTanimoto": [0.1] * rows,
                "Canonical": molecules,
                "SA_raw": [2.5] * rows,
            }
        )

    def test_missing_short_and_unreadable_tables_are_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            self.assertFalse(arm_is_complete(path, 2))
            self._arm_frame(1).to_csv(path, sep="\t", index=False)
            self.assertFalse(arm_is_complete(path, 2), "a one-row table cannot satisfy n=2")
            path.write_text("SMILES\tCanonical\tValid\tDesired\nnot a complete row")
            self.assertFalse(arm_is_complete(path, 1), "a truncated row must be rebuilt")
            self._arm_frame(2).to_csv(path, sep="\t", index=False)
            self.assertTrue(arm_is_complete(path, 2))

    def test_a_row_cut_off_after_desired_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            table = self._arm_frame(2)
            table.to_csv(path, sep="\t", index=False)
            lines = path.read_text().splitlines()
            lines[-1] = "\t".join(lines[-1].split("\t")[:5])  # the writer died after Desired
            path.write_text("\n".join(lines) + "\n")
            self.assertFalse(
                arm_is_complete(path, 2),
                "a row truncated after Desired has no Canonical and must be rebuilt",
            )

    def test_an_empty_generated_smiles_is_still_a_complete_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            table = self._arm_frame(2)
            table.loc[1, ["SMILES", "Canonical"]] = ""
            table.loc[1, "Valid"] = 0
            table.to_csv(path, sep="\t", index=False)
            self.assertTrue(arm_is_complete(path, 2))

    def test_an_arm_records_where_its_molecules_came_from(self):
        frame = self._arm_frame(2).drop(columns="SMILES")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            rebuilt = write_arm(path, frame, ["CC", "CCC"], origin="rebuilt", seed=42)
            self.assertEqual(rebuilt["origin"], "rebuilt")
            self.assertEqual(rebuilt["seed"], 42)
            self.assertTrue(rebuilt["sampled"], "a rebuild samples a checkpoint")
            prior = write_arm(path, frame, ["CC", "CCC"], origin="prior", seed=42)
            self.assertEqual(prior["origin"], "prior")
            self.assertFalse(prior["sampled"], "the prior arm is not a checkpoint draw")
            with self.assertRaises(ValueError, msg="an origin names a known kind of arm"):
                write_arm(path, frame, ["CC", "CCC"], origin="guessed")

    def test_a_recorded_arm_size_survives_a_change_of_n_final(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._arm_frame(2).to_csv(
                run_dir / "final_desired_ratio_at_stop.tsv", sep="\t", index=False
            )
            self.assertEqual(
                incomplete_arms(run_dir, 5000),
                ["desired_ratio_at_stop"],
                "without a recorded size the arm is judged against today's n_final",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"arms": {"desired_ratio_at_stop": {"n": 2}}})
            )
            self.assertEqual(recorded_arm_size(run_dir, "desired_ratio_at_stop"), 2)
            self.assertEqual(
                incomplete_arms(run_dir, 5000),
                [],
                "a recorded n pins the arm to the size it was written at",
            )
            # The recorded size is not a way to bless a damaged table.
            table = pd.read_csv(run_dir / "final_desired_ratio_at_stop.tsv", sep="\t")
            table.iloc[:1].to_csv(
                run_dir / "final_desired_ratio_at_stop.tsv", sep="\t", index=False
            )
            self.assertEqual(incomplete_arms(run_dir, 5000), ["desired_ratio_at_stop"])

    def test_the_original_run_cannot_finish_complete_with_a_missing_arm(self):
        cfg = small(n_final=2, fixed_thresholds=(0.5, 0.5))
        env = types.SimpleNamespace(
            getScorerKeys=lambda: ["ROCS", "SA"],
            getScores=lambda smiles: pd.DataFrame({"ROCS": [0.5] * len(smiles)}),
        )
        explorer = unittest.mock.Mock()
        explorer.fit.return_value = {"epochs_run": 1}

        def complete_file(path, n):
            return path.name != "final_desired_ratio_at_stop.tsv"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest = provenance.Manifest(run_dir / "manifest.json")
            resolved = {"thresholds": [0.5, 0.5]}
            with (
                patch.object(runner, "build_reference", return_value=[Chem.MolFromSmiles("CC")]),
                patch.object(Config, "pretrained", return_value=("agent", "vocabulary")),
                patch.object(runner.VocSmiles, "fromFile"),
                patch.object(runner, "load_model"),
                patch.object(runner, "sample", return_value=["CC", "CC"]),
                patch.object(
                    runner,
                    "calibrate_thresholds",
                    return_value=(
                        [0.5, 0.5],
                        pd.DataFrame({"ROCS": [0.6, 0.4], "SA": [1.0, 1.0], "Valid": [1, 1]}),
                    ),
                ),
                patch.object(runner, "build_environment", return_value=env),
                patch.object(runner, "build_scorers"),
                patch.object(runner, "write_arm", return_value={}),
                patch.object(runner, "components", return_value={}),
                patch.object(runner, "StopCheckpointExplorer", return_value=explorer),
                patch.object(runner, "arm_is_complete", side_effect=complete_file),
                patch.object(analysis, "analyse"),
            ):
                status = runner._pipeline(cfg, run_dir, resolved, manifest, time.monotonic())
            self.assertEqual(status, 1)
            self.assertEqual(manifest.data["state"], "failed")
            self.assertIn("desired_ratio_at_stop", manifest.data["error"])


def _arm(scores: list[float], desired: list[int]) -> pd.DataFrame:
    molecules = ["CCO", "CCCO", "c1ccccc1O", "CC(=O)O", "CCN"][: len(scores)]
    return pd.DataFrame(
        {
            "SMILES": molecules,
            "Canonical": molecules,
            "Valid": [1] * len(scores),
            "Desired": desired,
            "ROCS": scores,
        }
    )


class BestModelGallery(unittest.TestCase):

    def test_the_prior_is_never_the_best_model(self):
        arms = {
            "prior": _arm([0.9, 0.9], [1, 1]),
            "gmean": _arm([0.1, 0.1], [0, 0]),
        }
        self.assertEqual(figures.best_arm(arms, ["ROCS"]), "gmean")

    def test_the_desired_ratio_decides_and_the_median_breaks_the_tie(self):
        arms = {
            "desired_ratio": _arm([0.9, 0.9], [1, 0]),
            "gmean": _arm([0.2, 0.2], [1, 1]),
        }
        self.assertEqual(figures.best_arm(arms, ["ROCS"]), "gmean", "ratio first")
        tied = {
            "desired_ratio": _arm([0.2, 0.2], [1, 1]),
            "gmean": _arm([0.9, 0.9], [1, 1]),
        }
        self.assertEqual(figures.best_arm(tied, ["ROCS"]), "gmean", "then the median")

    def test_the_gallery_draws_the_arm_it_was_asked_for(self):
        arms = {
            "prior": _arm([0.9, 0.9], [1, 1]),
            "gmean": _arm([0.5, 0.4], [1, 0]),
            "desired_ratio": _arm([0.6, 0.3], [1, 1]),
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            primary = figures.lead_structures_gallery(arms, ["ROCS"], out, name="05_gallery")
            secondary = figures.lead_structures_gallery(
                arms, ["ROCS"], out, arm="desired_ratio", name="05b_gallery"
            )
            for path in (primary, secondary):
                self.assertIsNotNone(path)
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(primary.name, "05_gallery.png")
            self.assertEqual(secondary.name, "05b_gallery.png")

    def test_the_shown_arm_falls_back_when_the_wanted_one_is_empty(self):
        arms = {"prior": _arm([0.9], [1]), "gmean": _arm([0.5, 0.4], [1, 0])}
        self.assertEqual(figures.shown_arm(arms, ["ROCS"], "gmean"), "gmean")
        # desired_ratio was never written, so the figure shows what the run has.
        self.assertEqual(figures.shown_arm(arms, ["ROCS"], "desired_ratio"), "gmean")
        self.assertIsNone(figures.shown_arm({"prior": _arm([0.9], [1])}, ["ROCS"], "gmean"))

    def test_no_trained_arm_means_no_figure(self):
        self.assertIsNone(figures.best_arm({"prior": _arm([0.9], [1])}, ["ROCS"]))

    def test_all_invalid_trained_arms_do_not_promise_a_gallery(self):
        invalid = _arm([0.0], [0])
        invalid["Valid"] = 0
        self.assertIsNone(figures.best_arm({"desired_ratio": invalid}, ["ROCS"]))


class MetaCentrumJobs(unittest.TestCase):
    """What the adapter refuses locally, before a queue wait can hide the mistake."""

    def test_a_walltime_is_hh_mm_ss(self):
        self.assertEqual(submit_module.walltime_seconds("24:00:00"), 86400)
        self.assertEqual(submit_module.walltime_seconds("01:02:03"), 3723)
        for bad in ("24", "24:00", "24:60:00", "1d", "-01:00:00"):
            with self.assertRaises(ValueError, msg=bad):
                submit_module.walltime_seconds(bad)

    def test_the_walltime_routes_the_queue(self):
        self.assertEqual(submit_module.routed_queue("02:00:00"), "q_2h")
        self.assertEqual(submit_module.routed_queue("24:00:00"), "q_1d")
        self.assertEqual(submit_module.routed_queue("48:00:00"), "q_2d")
        with self.assertRaises(ValueError):
            submit_module.routed_queue("400:00:00")
        with self.assertRaises(ValueError):
            Job(name="probe", walltime="400:00:00")
        Job(name="probe", walltime="400:00:00", queue="special")

    def test_the_expected_queue_is_printed_not_guessed(self):
        self.assertEqual(
            Job(name="probe", walltime="24:00:00").queue_label, "q_1d (default router)"
        )
        self.assertEqual(
            Job(name="probe", walltime="24:00:00", queue="gpu_long").queue_label, "gpu_long"
        )

    def test_a_job_name_is_a_pbs_name_and_a_run_tag(self):
        Job(name="egcg_gold-2")
        for bad in ("1leading_digit", "has.dot", "has space", "x" * 65, ""):
            with self.assertRaises(ValueError, msg=bad):
                Job(name=bad)

    def test_cores_cannot_be_raised_without_memory(self):
        Job(name="probe", ncpus=16, mem_gb=32)
        with self.assertRaises(ValueError):
            Job(name="probe", ncpus=16, mem_gb=31)
        with self.assertRaises(ValueError):
            Job(name="probe", ncpus=32, mem_gb=63)

    def test_every_campaign_run_is_a_legal_job_at_the_planned_memory(self):
        for batch in campaign.BATCHES.values():
            for name, _ in campaign.runs(batch):
                Job(name=name, ncpus=16, mem_gb=campaign.MEMORY_GB)
                self.assertRegex(name, submit_module._NAME_RE)

    def test_a_rebuild_job_names_runs_and_takes_no_deltas(self):
        job = Job(name="rebuild_1", mode="rebuild", runs=("20261111_0444_w_e05q0",))
        self.assertEqual(job.mode, "rebuild")
        with self.assertRaises(ValueError, msg="no runs"):
            Job(name="rebuild_1", mode="rebuild")
        with self.assertRaises(ValueError, msg="a delta would change the yardstick"):
            Job(name="rebuild_1", mode="rebuild", runs=("run_a",), args=("--set", "epsilon=0.1"))
        with self.assertRaises(ValueError, msg="runs belong to a rebuild"):
            Job(name="train_1", runs=("run_a",))
        with self.assertRaises(ValueError, msg="unknown mode"):
            Job(name="probe", mode="rescore")
        with self.assertRaises(ValueError, msg="a run name is also a path segment"):
            Job(name="rebuild_1", mode="rebuild", runs=("../escape",))

    def test_runs_are_split_into_nearly_equal_contiguous_groups(self):
        runs = [f"run_{index}" for index in range(7)]
        self.assertEqual(
            submit_module.slices(runs, 3),
            [("run_0", "run_1", "run_2"), ("run_3", "run_4"), ("run_5", "run_6")],
        )
        self.assertEqual(len(submit_module.slices(runs, 99)), len(runs), "one job per run")

    def test_a_rebuild_argv_carries_each_runs_own_conformer_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            for run, conformers in (("run_a", 30), ("run_b", 10)):
                (storage / run).mkdir()
                (storage / run / "config.json").write_text(
                    json.dumps({"max_conformers": conformers})
                )
            spec = {
                "name": "rebuild_1",
                "mode": "rebuild",
                "runs": ["run_a", "run_b"],
                "args": [],
                "mem_gb": 32,
                "revision": "abc",
            }
            commands = job_module.commands_for(
                Path("/storage/.venv/bin/python"),
                Path("/scratch/output"),
                storage,
                16,
                spec,
            )
        self.assertEqual(len(commands), 2)
        self.assertIn("_david.code.reconstruct.score_arms", commands[0])
        self.assertEqual(commands[1][-3], f"{storage}/run_b")
        self.assertEqual(commands[0][-1], "8", "30 conformers: 8 workers in 32 GB")
        self.assertEqual(commands[1][-1], "16", "10 conformers fit all 16 cores")

    def test_a_train_spec_without_mode_still_trains(self):
        spec = {"name": "probe", "args": ["--set", "epochs=3"], "revision": "abc"}
        commands = job_module.commands_for(
            Path("/storage/.venv/bin/python"),
            Path("/scratch/output"),
            Path("/storage/_david/output"),
            16,
            spec,
        )
        self.assertEqual(len(commands), 1)
        self.assertIn("_david.code.run", commands[0])
        self.assertIn("n_jobs=16", commands[0])

    def test_a_rebuild_spec_without_runs_is_refused_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            base = {"name": "rebuild_1", "args": [], "source": "/rel", "revision": "abc"}
            path.write_text(json.dumps(base | {"mode": "rebuild"}))
            with self.assertRaises(job_module.JobError):
                job_module.read_spec(path)
            path.write_text(json.dumps(base | {"mode": "rebuild", "runs": ["run_a"]}))
            self.assertEqual(job_module.read_spec(path)["runs"], ["run_a"])
            path.write_text(json.dumps(base | {"mode": "teleport"}))
            with self.assertRaises(job_module.JobError):
                job_module.read_spec(path)

    def test_long_format_qstat_keeps_long_names_and_resolvable_nodes(self):
        text = """Job Id: 25256877.pbs-m1.metacentrum.cz
    Job_Name = ws_e05_noqed
    job_state = R
    queue = q_1d
    exec_host2 = eluo1-2.hw.elixir-czech.cz:15002/4*4
    resources_used.walltime = 01:02:03
    Resource_List.walltime = 24:00:00

Job Id: 25256880.pbs-m1.metacentrum.cz
    Job_Name = p_m1e10r4c10q0
    job_state = Q
    queue = q_1d
"""
        jobs = submit_module.parse_qstat_full(text)
        self.assertEqual([job.name for job in jobs], ["ws_e05_noqed", "p_m1e10r4c10q0"])
        running = jobs[0]
        self.assertEqual(running.short_id, "25256877")
        self.assertTrue(running.running)
        self.assertEqual(running.node, "eluo1-2.hw.elixir-czech.cz")
        self.assertEqual(running.elapsed, "01:02:03")
        self.assertFalse(jobs[1].running)
        self.assertEqual(jobs[1].node, "")

    def test_the_monitor_reads_the_epoch_table_by_its_sentinel_frames(self):
        text = (
            "SCRATCH=/scratch/job_1\n"
            "RUN=/scratch/job_1/output/20261112_1200_probe/\n"
            "COUNT=3\n"
            "HEADER<<@@@DRUGEX@@@\n"
            "Epoch\tDesired\n"
            "@@@DRUGEX@@@\n"
            "ROWS<<@@@DRUGEX@@@\n"
            "1\t0.2\n"
            "2\t0.3\n"
            "@@@DRUGEX@@@\n"
            "EPOCHS=70\n"
            "STATE=running\n"
        )
        probe = monitor.parse_probe(text)
        self.assertEqual(probe["header"], ["Epoch\tDesired"])
        self.assertEqual(probe["rows"], ["1\t0.2", "2\t0.3"])
        self.assertEqual(probe["epochs"], "70")
        self.assertEqual(probe["state"], "running")

    def test_a_32gb_request_caps_the_scoring_pool_below_the_oom_killer(self):
        # The measurement (memreport) says a worker needs 0.1 GB per conformer x 1.15;
        # at the default 20 conformers that is 2.3 GB, so 32 GB runs 13 workers.
        self.assertEqual(job_module.workers_for({"mem_gb": 32}, 16), 13)
        self.assertEqual(job_module.workers_for({"mem_gb": 64}, 16), 16)
        self.assertEqual(job_module.workers_for({"mem_gb": 32}, 8), 8)
        self.assertEqual(job_module.workers_for({}, 16), 16, "old specs stay all-cores")
        self.assertEqual(job_module.workers_for({"mem_gb": 4}, 16), 1)

    def test_the_worker_budget_follows_the_conformer_count(self):
        def workers(conformers: int, memory: int) -> int:
            return job_module.workers_for(
                {"mem_gb": memory}, 32, conformers=conformers
            )

        # 30 conformers measured 2.8 GB/worker, 10 measured 1.0: the same 32 GB must not
        # run the same pool for both.
        self.assertEqual(workers(10, 32), 26, "10 conformers: 1.15 GB per worker")
        self.assertEqual(workers(20, 32), 13, "20 conformers: 2.3 GB per worker")
        self.assertEqual(workers(30, 32), 8, "30 conformers: 3.45 GB per worker")
        self.assertEqual(workers(30, 64), 17, "64 GB allows more, not all, cores")
        self.assertAlmostEqual(job_module.gb_per_worker(30), 3.45)
        self.assertAlmostEqual(job_module.gb_per_worker(10), 1.15)

    def test_a_spec_delta_sets_the_conformer_budget_and_the_last_one_wins(self):
        spec = {"mem_gb": 32, "args": ["--set", "max_conformers=30"]}
        self.assertEqual(job_module.spec_max_conformers(spec), 30)
        self.assertEqual(job_module.workers_for(spec, 16), 8)
        spec["args"] += ["--set", "max_conformers=10"]
        self.assertEqual(job_module.spec_max_conformers(spec), 10)
        self.assertEqual(
            job_module.spec_max_conformers({"args": []}),
            job_module.DEFAULT_MAX_CONFORMERS,
        )
        self.assertEqual(
            job_module.spec_max_conformers({"args": ["--set", "max_conformers=lots"]}),
            job_module.DEFAULT_MAX_CONFORMERS,
            "an unparseable delta must not crash a job body",
        )

    def test_the_duplicated_conformer_default_matches_the_config(self):
        self.assertEqual(
            Config().max_conformers,
            job_module.DEFAULT_MAX_CONFORMERS,
            "job.py must not import Config (torch), so the test guards the duplicate",
        )

    def test_the_queue_script_watches_only_the_three_useful_lanes(self):
        self.assertEqual(queues.QUEUES, ("q_4h", "q_1d", "q_2d"))
        self.assertIn("NOT for training", queues.ROLE["q_4h"])
        self.assertIn("DEFAULT", queues.ROLE[submit_module.DEFAULT_QUEUE])

    def test_long_format_queue_state_is_parsed(self):
        text = """Queue: q_1d
    total_jobs = 5454
    state_count = Transit:0 Queued:3873 Held:0 Waiting:0 Running:1581 Exiting:0 Begun:0
    resources_max.walltime = 24:00:00

Queue: q_2d
    total_jobs = 5253
    state_count = Transit:0 Queued:4986 Held:0 Waiting:0 Running:267 Exiting:0 Begun:0
    resources_max.walltime = 48:00:00
"""
        depth = {entry.name: entry for entry in queues.parse_queues(text)}
        self.assertEqual(depth["q_1d"].running, 1581)
        self.assertEqual(depth["q_1d"].queued, 3873)
        self.assertEqual(depth["q_1d"].max_walltime, "24:00:00")
        self.assertAlmostEqual(depth["q_2d"].pressure, 4986 / 267)
        self.assertEqual(depth["q_1d"].total, 5454)


class MemoryReport(unittest.TestCase):
    """The worker budget in ``job.py`` is checked against what PBS measured, not trusted."""

    #: Three finished jobs in the exact shape ``qstat -x -f`` prints: one killed at its
    #: cgroup, one healthy 30-conformer job, one 20-conformer job that nearly filled 32 GB.
    QSTAT = """Job Id: 25435602.pbs-m1.metacentrum.cz
    Job_Name = p_m1e10r4c30q0
    resources_used.mem = 33554432kb
    resources_used.ncpus = 16
    resources_used.walltime = 15:37:05
    job_state = F
    Resource_List.mem = 32gb
    Resource_List.ncpus = 16
    Resource_List.walltime = 24:00:00
    Exit_status = 137
Job Id: 25446304.pbs-m1.metacentrum.cz
    Job_Name = p_m1e00r1c30q1
    resources_used.mem = 38676060kb
    resources_used.ncpus = 16
    resources_used.walltime = 08:46:40
    job_state = F
    Resource_List.mem = 64gb
    Resource_List.ncpus = 16
    Resource_List.walltime = 48:00:00
    Exit_status = 0
Job Id: 25446500.pbs-m1.metacentrum.cz
    Job_Name = p_m1e10r1c20q0
    resources_used.mem = 31641600kb
    resources_used.ncpus = 16
    resources_used.walltime = 07:10:00
    job_state = F
    Resource_List.mem = 32gb
    Resource_List.ncpus = 16
    Resource_List.walltime = 24:00:00
    Exit_status = 0
"""

    #: Old specs carry no ``workers``, so the pool is the allocated core count.
    SPECS = """### p_m1e10r4c30q0/spec.json
{"name": "p_m1e10r4c30q0", "args": ["--set", "max_conformers=30"], "mem_gb": 32, "source": "/rel", "revision": "abc"}
### p_m1e00r1c30q1/spec.json
{"name": "p_m1e00r1c30q1", "args": ["--set", "max_conformers=30"], "mem_gb": 64, "source": "/rel", "revision": "abc"}
### p_m1e10r1c20q0/spec.json
{"name": "p_m1e10r1c20q0", "args": [], "mem_gb": 32, "source": "/rel", "revision": "abc"}
### gone/spec.json
{not json
"""

    def _rows(self):
        jobs = submit_module.parse_qstat_full(self.QSTAT)
        return memreport.finished_from_jobs(jobs, memreport.specs_from_stream(self.SPECS))

    def test_both_pbs_memory_spellings_are_gigabytes(self):
        self.assertEqual(memreport.parse_mem("32gb"), 32.0)
        self.assertEqual(memreport.parse_mem("33554432kb"), 32.0)
        self.assertEqual(memreport.parse_mem("512mb"), 0.5)
        self.assertEqual(memreport.parse_mem("1024b"), 1024 / 1024**3)
        for empty in ("", "  ", "nonsense", "32 goats", None):
            self.assertIsNone(memreport.parse_mem(empty), empty)

    def test_an_exit_137_at_the_request_is_an_oom_kill_not_a_tight_fit(self):
        oom, healthy, tight = self._rows()
        self.assertEqual(oom.exit_status, 137)
        self.assertEqual(oom.verdict, "OOM (killed at the request)")
        self.assertAlmostEqual(oom.requested_gb, 32.0)
        self.assertAlmostEqual(oom.peak_gb, 32.0)
        self.assertAlmostEqual(oom.fill, 1.0)
        self.assertEqual(oom.conformers, 30)
        self.assertEqual(oom.workers, 16, "an old spec ran every allocated core")
        self.assertAlmostEqual(oom.per_worker_gb, 2.0)
        self.assertAlmostEqual(healthy.per_worker_gb, 38676060 / 1024**2 / 16, places=6)
        self.assertEqual(healthy.verdict, "ok")
        self.assertEqual(tight.verdict, "tight", "94% of the request has no headroom")

    def test_a_job_without_a_staged_spec_reports_no_workers_rather_than_guessing(self):
        jobs = submit_module.parse_qstat_full(self.QSTAT)
        rows = memreport.finished_from_jobs(jobs, {})
        self.assertEqual([row.workers for row in rows], [None, None, None])
        self.assertEqual([row.conformers for row in rows], [None, None, None])
        self.assertIsNone(rows[0].per_worker_gb)
        self.assertIsNone(rows[0].budget_gb)

    def test_a_rebuild_spec_measures_fill_but_not_the_budget(self):
        jobs = submit_module.parse_qstat_full(self.QSTAT)
        specs = {"p_m1e00r1c30q1": {"mode": "rebuild", "args": [], "mem_gb": 64}}
        row = memreport.finished_from_jobs(jobs, specs)[1]
        self.assertIsNone(row.workers, "a rebuild spans runs of different shapes")
        self.assertIsNone(row.conformers)
        self.assertIsNone(row.per_worker_gb)
        self.assertIsNotNone(row.fill)

    def test_the_summary_calibrates_the_budget_from_the_worst_measured_job(self):
        stats = memreport.summary(self._rows())
        self.assertEqual(
            stats["measured"], 2, "the OOM job's peak is truncated by its cgroup"
        )
        self.assertEqual(stats["oom"], ["p_m1e10r4c30q0"])
        self.assertAlmostEqual(
            stats["per_conformer"]["30"]["worst_gb_per_worker"], 38676060 / 1024**2 / 16
        )
        self.assertAlmostEqual(
            stats["per_conformer"]["20"]["worst_gb_per_worker"],
            31641600 / 1024**2 / 16,
            places=6,
        )
        self.assertEqual(
            stats["suggested_gb_per_conformer"],
            job_module.MEM_GB_PER_CONFORMER,
            "a measurement below the budget must not lower it",
        )

    def test_a_measurement_above_the_budget_raises_the_coefficient(self):
        heavy = memreport.Finished(
            job_id="1.pbs",
            name="heavy",
            state="F",
            exit_status=0,
            requested_gb=128.0,
            peak_gb=80.0,
            ncpus=16,
            workers=16,
            conformers=40,
        )
        self.assertEqual(heavy.verdict, "over its budget")
        stats = memreport.summary([heavy])
        self.assertEqual(stats["over_budget"], ["heavy"])
        self.assertAlmostEqual(stats["suggested_gb_per_conformer"], 0.13)

    def test_the_reported_pool_matches_what_the_cap_would_choose(self):
        spec = {"mem_gb": 32, "args": ["--set", "max_conformers=30"]}
        row = memreport.Finished(
            job_id="1.pbs",
            name="p_m1e10r4c30q0",
            state="F",
            exit_status=0,
            requested_gb=32.0,
            peak_gb=20.0,
            ncpus=16,
            workers=job_module.workers_for(spec, 16),
            conformers=job_module.spec_max_conformers(spec),
        )
        self.assertEqual(row.workers, 8)
        self.assertLessEqual(row.per_worker_gb, row.budget_gb)

    def test_the_table_and_summary_name_the_verdict(self):
        rows = self._rows()
        text = f"{memreport._table(rows)}\n\n{memreport._summary_text(memreport.summary(rows))}"
        self.assertIn("OOM (killed at the request)", text)
        self.assertIn("needs a larger --mem", text)
        self.assertIn("25435602", text)
        self.assertIn("conf", text, "the per-shape table is what the budget is read from")


class Campaign(unittest.TestCase):

    def test_every_delta_is_a_setting_the_pipeline_accepts(self):
        for batch in campaign.BATCHES.values():
            for name, settings in campaign.runs(batch):
                cfg = config_from_args(campaign.deltas(settings))
                for key, wanted in settings.items():
                    self.assertEqual(getattr(cfg, key), wanted, f"{name}: {key}")
                scorers.reward_scheme(cfg.scheme)

    def test_a_batch_changes_the_reward_scheme_only_by_being_about_it(self):
        for name, batch in campaign.BATCHES.items():
            self.assertNotIn("scheme", batch.axes, name)
            schemes = {settings["scheme"] for _, settings in campaign.runs(batch)}
            self.assertEqual(len(schemes), 1, f"{name} mixes reward schemes: {schemes}")

    def test_job_names_are_unique_across_every_batch(self):
        names = [name for batch in campaign.BATCHES.values() for name, _ in campaign.runs(batch)]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        self.assertFalse(duplicated, f"a repeated name overwrites a spec on storage: {duplicated}")

    def test_no_run_sets_a_setting_the_job_owns(self):
        for batch in campaign.BATCHES.values():
            for job, settings in campaign.runs(batch):
                for reserved in submit_module.RESERVED:
                    self.assertNotIn(reserved, settings, job)

    def test_every_run_asks_for_more_memory_than_its_cell_was_measured_to_use(self):
        # True peaks from resources_used.mem, by (minimize, max_conformers).
        measured = {
            (False, 10): 22.7,
            (False, 20): 28.0,
            (False, 30): 27.8,
            (True, 10): 22.1,
            (True, 20): 31.2,
        }
        for cell, peak in measured.items():
            # Twice the peak: siblings of one configuration differ by ~5 GB.
            self.assertGreaterEqual(campaign.MEMORY_GB, peak * 2.0, f"minimize/c{cell}")
        # The ad-hoc default is the fast-to-place 32 GB request; the planned sweep includes
        # the ~40 GB heavy cell and deliberately buys headroom instead of queue speed.
        self.assertEqual(submit_module.Job.mem_gb, submit_module.DEFAULT_MEM_GB)
        self.assertLess(submit_module.Job.mem_gb, campaign.MEMORY_GB)

    def test_a_repeated_configuration_is_reported_as_a_replicate(self):
        by_name = {
            name: settings
            for batch in campaign.BATCHES.values()
            for name, settings in campaign.runs(batch)
        }
        for repeat, original in campaign.duplicates():
            self.assertNotEqual(repeat, original)
            self.assertEqual(by_name[repeat], by_name[original], f"{repeat} is not a replicate")


class CollapseVerdict(unittest.TestCase):
    """The triage labels collapse.py hands out, and the statistics they are built on."""

    @staticmethod
    def _stop_arm(unique_ratio, desired=0.9, top10=0.1, valid=0.99):
        return {
            "n": 5000, "valid_ratio": valid, "unique_ratio": unique_ratio,
            "top1_share": top10 / 10, "top10_share": top10, "desired_ratio": desired,
            "rocs_median": None, "rocs_max": None,
        }

    PRIOR = {"n": 5000, "valid_ratio": 0.95, "unique_ratio": 0.95, "top1_share": 0.0,
             "top10_share": 0.003, "desired_ratio": 0.13, "rocs_median": None, "rocs_max": None}
    CHEMISTRY = {"hac_median": 27.0, "mw_median": 365.0, "scaffolds": 100, "scaffold_top": 0.1}

    def _verdict(self, stop, prior=None, n=5000):
        return collapse.verdict(
            stop, prior or self.PRIOR, self.CHEMISTRY, self.CHEMISTRY, n
        )

    def test_a_unique_arm_that_learned_is_clean(self):
        result = self._verdict(self._stop_arm(0.9, desired=0.8))
        self.assertEqual(result["label"], "clean")
        self.assertFalse(result["flags"], "a clean run needs no caveats attached")

    def test_a_unique_arm_that_never_left_the_prior_is_unlearned(self):
        # High uniqueness with no desired gain is the quiet failure: still the prior.
        result = self._verdict(self._stop_arm(0.9, desired=0.14))
        self.assertEqual(result["label"], "unlearned")
        self.assertIn("no_learning", result["flags"])

    def test_uniqueness_below_the_cut_is_collapse_whatever_the_reward(self):
        # The collapsed cluster sits at 0.25-0.33 unique with desired ~0.9: one series.
        result = self._verdict(self._stop_arm(0.25, desired=0.9))
        self.assertEqual(result["label"], "collapsed")
        self.assertIn("collapse", result["flags"])

    def test_a_series_lock_collapses_even_above_the_uniqueness_cut(self):
        # 0.5 unique but the top 10 are 70 % of the draw: it passes the cut, one finding.
        result = self._verdict(self._stop_arm(0.5, desired=0.9, top10=0.7))
        self.assertEqual(result["label"], "collapsed")
        self.assertIn("series_lock", result["flags"])

    def test_between_the_cuts_is_concentrated_not_collapsed(self):
        result = self._verdict(self._stop_arm(0.5, desired=0.9, top10=0.2))
        self.assertEqual(result["label"], "concentrated")

    def test_validity_loss_alone_cannot_label_a_run_clean(self):
        result = self._verdict(self._stop_arm(0.9, desired=0.8, valid=0.6))
        self.assertEqual(result["label"], "collapsed")
        self.assertIn("validity_collapse", result["flags"])

    def test_fragment_drift_is_flagged_from_the_heavy_atom_medians(self):
        thin = {"hac_median": 18.0, "mw_median": 250.0, "scaffolds": 50, "scaffold_top": 0.1}
        result = collapse.verdict(
            self._stop_arm(0.9, desired=0.8), self.PRIOR, thin, self.CHEMISTRY, 5000
        )
        self.assertIn("fragment_drift", result["flags"])

    def test_a_pilot_arm_is_never_judged(self):
        # Uniqueness at n=10 means nothing, or the smoke runs read as the most diverse.
        result = self._verdict(self._stop_arm(1.0, desired=0.5), n=10)
        self.assertEqual(result["label"], "pilot")

    def test_a_short_arm_is_incomplete_not_collapsed(self):
        # A walltime kill mid-write leaves a partial table; that is a rebuild, not a verdict.
        stop = self._stop_arm(0.2, desired=0.9)
        stop["n"] = 2000
        result = self._verdict(stop, n=5000)
        self.assertEqual(result["label"], "incomplete")

    def test_arm_stats_follow_write_arms_conventions(self):
        # The manifest's own definition, so the atlas and the manifest cannot disagree.
        frame = pd.DataFrame(
            {
                "Valid": [1, 1, 1, 1, 0],
                "Desired": [1, 1, 1, 0, 0],
                "Canonical": ["CCO", "CCO", "CCO", "CCN", ""],
                "RDKit_Aggregate_1refs_TanimotoCombo": [1.4, 1.4, 1.4, 1.0, 0.0],
            }
        )
        stats = collapse.arm_stats(frame, "RDKit_Aggregate_1refs_TanimotoCombo", 2.0)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["unique_ratio"], 0.4)  # CCO and CCN of five draws
        self.assertAlmostEqual(stats["desired_ratio"], 0.2)  # CCO alone, once
        self.assertAlmostEqual(stats["rocs_median"], 2.4)  # published scale, deduped: median of 2.8 and 2.0
        self.assertAlmostEqual(stats["top1_share"], 0.6)

    def test_the_frontier_ignores_runs_the_verdict_cannot_trust(self):
        # Pilots would dominate (desired, unique) for free, and the frontier is the highlight.
        def run(name, desired, unique, verdict="clean", n=5000):
            return {
                "run": name, "verdict": verdict, "n_expected": n, "arms": {"stop": {
                    "n": n, "desired_ratio": desired, "unique_ratio": unique}},
                "axes": {},
            }

        records = [
            run("real", 0.6, 0.8),
            run("pilot", 1.0, 1.0, verdict="pilot", n=10),
            run("short", 0.9, 0.9, verdict="incomplete", n=2000),
            run("dominated", 0.5, 0.7),
        ]
        self.assertEqual(collapse._frontier(records), ["real"])


class TreeAtlas(unittest.TestCase):
    """What the tree atlas celebrates, what it refuses to, and which runs it counts."""

    @staticmethod
    def _scan_run(name, verdict="clean", desired=0.5, unique=0.8, qed=False,
                  gmean_desired=0.0, n=5000):
        return {
            "run": name, "repo": "current", "legacy": False, "state": "complete",
            "verdict": verdict, "flags": [], "reasons": [],
            "axes": {"scheme": "PRCD", "epsilon": 0.05, "minimize": True,
                     "reference_conformers": 4, "max_conformers": 10,
                     "include_qed": qed, "color_ff": "fdef", "alignment_shape_weight": 0.5},
            "n_expected": n, "hours": 1.0, "epochs_run": 70,
            "arms": {"stop": {"desired_ratio": desired, "unique_ratio": unique},
                     "gmean": {"desired_ratio": gmean_desired}},
            "chemistry": {}, "curve": {}, "frontier": False,
        }

    def test_medals_rank_by_stop_arm_desired_and_skip_the_unmedallable(self):
        # "Read this run first": it follows the stop arm, never gmean, and never a collapsed
        # arm whose desired ratio is one series counted 5000 times.
        scan = {"generated": "now", "roots": [], "since": "20261110", "totals": {},
                "counts": {}, "runs": [
                    self._scan_run("winner", desired=0.8),
                    self._scan_run("second", desired=0.7, gmean_desired=0.95),
                    self._scan_run("collapsed_but_shiny", verdict="collapsed",
                                   desired=0.99, gmean_desired=1.0),
                    self._scan_run("pilot", verdict="pilot", desired=1.0, n=10),
                    self._scan_run("third", desired=0.6),
                    self._scan_run("fourth", desired=0.55),
                ]}
        payload = atlas_v2.tree_payload(scan)
        by_name = {r["run_id"]: r for r in payload["runs"]}
        self.assertEqual(by_name["winner"]["rank"], 1)  # gmean 0.95 on "second" loses
        self.assertEqual(by_name["second"]["rank"], 2)
        self.assertEqual(by_name["third"]["rank"], 3)
        self.assertNotIn("rank", by_name["collapsed_but_shiny"])
        self.assertNotIn("rank", by_name["pilot"])
        self.assertNotIn("rank", by_name["fourth"])  # three medals per family, no more

    def test_factors_fold_the_axes_onto_the_tree_floors(self):
        factors = atlas_v2._factors(
            {"scheme": "WS", "epsilon": 0.05, "minimize": False, "reference_conformers": 1,
             "max_conformers": 20, "color_ff": "oe_mills_dean", "alignment_shape_weight": 0.3}
        )
        self.assertEqual(factors, {
            "reward": "WS", "epsilon": "0.05", "geometry": "K1 · raw", "budget": "C20",
            "color": "oe_mills_dean", "shape_color": "w 0.3",
        })

    def test_the_scan_counts_only_runs_since_the_restart(self):
        # The 10.11 rule: WITH_BUGS runs and first probes stay out, whatever root holds them.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("20261108_1200_old_regime", "20261111_0400_new",
                         "20261112_0100_newer"):
                run = root / name
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({
                    "state": "complete",
                    "config": {"scheme": "PRCD", "epsilon": 0.05, "n_final": 10},
                }))
            collapse.build([root], root / "atlas.html", since="20261110")
            payload = json.loads((root / "collapse_scan.json").read_text())
            self.assertEqual([r["run"] for r in payload["runs"]],
                             ["20261111_0400_new", "20261112_0100_newer"])
            self.assertEqual(payload["since"], "20261110")

    def test_the_scanner_writes_scan_data_and_no_retired_v1_atlas(self):
        # The tree atlas owns the HTML; a scan must leave json/tsv behind and nothing else,
        # whatever argv the old callers still pass as their output anchor.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "20261112_0200_nofrills"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "state": "complete", "config": {"scheme": "PRCD", "n_final": 4},
            }))
            collapse.build([root], root / "collapse_atlas.html", since="20261110")
            self.assertTrue((root / "collapse_scan.json").is_file())
            self.assertTrue((root / "collapse_scan.tsv").is_file())
            self.assertFalse((root / "collapse_atlas.html").exists(),
                             "the v1 atlas is retired; atlas_v2 renders the tree atlas")

    def test_the_scan_carries_hierarchical_diversity_and_effective_desired_yield(self):
        # Judged on the stop arm only: the EDY is the stop arm's desired fraction times its
        # effective scaffold families, normalised by the expected draw count.
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261112_0300_hier"
            run.mkdir()
            columns = ["SMILES", "Canonical", "Valid", "Desired", "SA_raw"]
            pd.DataFrame({
                "SMILES": ["CCO", "CCN"], "Canonical": ["CCO", "CCN"],
                "Valid": [1, 1], "Desired": [0, 0], "SA_raw": [1.0, 1.0],
            }).to_csv(run / "prior.tsv", sep="\t", index=False, columns=columns)
            pd.DataFrame({
                "SMILES": ["c1ccccc1C", "c1ccncc1C", "C1CCCCC1C", "CCO"],
                "Canonical": ["c1ccccc1C", "c1ccncc1C", "C1CCCCC1C", "CCO"],
                "Valid": [1, 1, 1, 1], "Desired": [1, 1, 0, 0], "SA_raw": [1.0] * 4,
            }).to_csv(run / "final_desired_ratio_at_stop.tsv", sep="\t", index=False,
                      columns=columns)
            (run / "manifest.json").write_text(json.dumps({
                "state": "complete", "config": {"scheme": "PRCD", "n_final": 4},
            }))
            profile = {
                "version": diversity.DIVERSITY_VERSION,
                "arms": {
                    "prior": {},
                    "stop": {
                        "effective_scaffolds": 3.0,
                        "scaffold_internal_diversity": 0.4,
                        "sidechain_internal_diversity": 0.7,
                        "scaffold_entropy_ratio": 0.6,
                    },
                },
                "sources": diversity._sources(run),
            }
            (run / "cache_stats").mkdir()
            (run / "cache_stats" / "diversity.json").write_text(json.dumps(profile))
            record = collapse.scan_run(run, legacy=False)
            levelled = record["diversity"]
            self.assertEqual(levelled["scaffold_internal_diversity"], 0.4)
            self.assertEqual(levelled["sidechain_internal_diversity"], 0.7)
            self.assertEqual(levelled["scaffold_entropy_ratio"], 0.6)
            # Stop arm: 2 desired of 4 draws, 3.0 effective scaffolds -> 0.5 * 3 / 4.
            self.assertAlmostEqual(record["arms"]["stop"]["desired_ratio"], 0.5)
            self.assertAlmostEqual(levelled["effective_desired_yield"], 0.375)


class DiversityProfile(unittest.TestCase):
    """The chemical-diversity metrics and the cache that keeps them cheap."""

    @staticmethod
    def _frame(rows):
        # rows: (canonical, count, desired)
        return pd.DataFrame(
            [r for canonical, count, desired in rows for r in
             [(canonical, desired)] * count],
            columns=["Canonical", "Desired"],
        ).assign(Valid=1)

    def test_effective_molecules_counts_a_repeated_series_once(self):
        # 5,000 draws of 5 distinct molecules, 1,000 each: 5 effective findings, not 5,000.
        uniform = diversity.ArmUniques(self._frame([(f"C{'C' * i}O", 1, 1) for i in range(10)]))
        self.assertAlmostEqual(diversity._entropy_shannon(uniform.counts), 10.0)
        repeated = diversity.ArmUniques(self._frame([("CCO", 5, 1), ("CCCO", 5, 0)]))
        self.assertAlmostEqual(diversity._entropy_shannon(repeated.counts), 2.0)

    def test_a_collapsed_arm_has_no_effective_scaffolds_left(self):
        # The same scaffold wearing different hats: many canonicals, one family.
        hats = [f"c1ccc(C{'C' * i})cc1" for i in range(6)]
        uniques = diversity.ArmUniques(self._frame([(smi, 1, 1) for smi in hats]))
        profile = diversity.arm_profile(uniques)
        self.assertEqual(profile["n_unique"], 6)
        self.assertEqual(profile["scaffolds"], 1)
        self.assertAlmostEqual(profile["effective_scaffolds"], 1.0)
        self.assertLess(profile["internal_diversity"], 0.5)

    def test_pairwise_statistics_match_a_direct_similarity(self):
        # Two molecules: every pairwise statistic is that one similarity, to 4 places.
        methanol, ethanol = "CO", "CCO"
        fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
              for s in (methanol, ethanol)]
        expected = DataStructs.TanimotoSimilarity(*fp)
        profile = diversity._pairwise_profile(fp)
        self.assertAlmostEqual(profile["nn_sim_max"], expected, places=4)
        self.assertAlmostEqual(profile["nn_sim_p50"], expected, places=4)
        self.assertAlmostEqual(profile["nn_sim_p90"], expected, places=4)
        self.assertAlmostEqual(profile["internal_diversity"], 1 - expected, places=4)

    def test_prior_echo_of_a_molecule_the_prior_already_held(self):
        # A stop arm that regurgitates the prior: closest prior similarity is itself.
        prior = diversity.ArmUniques(self._frame([("CCO", 1, 0), ("c1ccccc1O", 1, 0)]))
        stop = diversity.ArmUniques(self._frame([("CCO", 3, 1)]))
        cross = diversity._cross_profile(stop.fingerprints, prior.fingerprints)
        self.assertAlmostEqual(cross["prior_sim_p50"], 1.0)
        self.assertAlmostEqual(cross["prior_echo_rate"], 1.0)
        self.assertAlmostEqual(cross["novelty_rate"], 0.0)

    def test_the_cache_is_fresh_only_while_its_sources_are(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "prior.tsv").write_text(
                self._frame([("CCO", 1, 0)]).to_csv(sep="\t", index=False))
            profile = diversity.profile_run(run)
            self.assertIsNotNone(profile)
            cache = run / "cache_stats"
            cache.mkdir()
            profile["sources"] = diversity._sources(run)
            (cache / "diversity.json").write_text(json.dumps(profile))
            self.assertTrue(diversity.profile_is_fresh(run))
            self.assertEqual(diversity.load_profile(run), profile)
            (run / "prior.tsv").write_text(  # same table plus one molecule: stale now
                self._frame([("CCO", 1, 0), ("CCCO", 1, 0)]).to_csv(sep="\t", index=False))
            self.assertFalse(diversity.profile_is_fresh(run))
            self.assertIsNone(diversity.load_profile(run))

    def test_the_thumbnail_shares_axes_between_stop_and_prior(self):
        prior = diversity.ArmUniques(self._frame(
            [(f"C{'C' * i}N", 1, 0) for i in range(12)]))
        stop = diversity.ArmUniques(self._frame(
            [(f"C{'C' * i}O", 1, 1) for i in range(12)]))
        thumb = diversity._thumb_coordinates(stop, prior)
        self.assertIsNotNone(thumb)
        self.assertEqual(len(thumb["stop"]["x"]), len(thumb["stop"]["desired"]))
        self.assertEqual(len(thumb["prior"]["x"]), len(thumb["stop"]["x"]))
        self.assertLessEqual(sum(thumb["explained"]), 1.0 + 1e-9)
        self.assertTrue(all(d == 1 for d in thumb["stop"]["desired"]))

    def test_scaffold_diversity_is_blind_to_decoration(self):
        # One benzene scaffold, five different side chains: the molecules differ and the
        # whole-molecule layer says so, but the arm holds a single scaffold family.
        hats = [f"c1ccc(C{'C' * i})cc1" for i in range(1, 6)]
        profile = diversity.arm_profile(
            diversity.ArmUniques(self._frame([(smi, 1, 1) for smi in hats]))
        )
        self.assertEqual(profile["scaffolds"], 1)
        self.assertEqual(profile["scaffold_internal_diversity"], 0.0)
        self.assertGreater(profile["internal_diversity"], 0.0)
        self.assertGreater(profile["sidechain_internal_diversity"], 0.0)

    def test_sidechain_diversity_is_blind_to_the_scaffold(self):
        # The mirror case: four ring systems wearing the same ethyl decoration. Breadth in
        # the scaffolds, none in the side chains — the layers must disagree by design.
        skeletons = ["c1ccccc1CC", "c1ccncc1CC", "c1ccc2ccccc2c1CC", "C1CCCCC1CC"]
        profile = diversity.arm_profile(
            diversity.ArmUniques(self._frame([(smi, 1, 1) for smi in skeletons]))
        )
        self.assertEqual(profile["scaffolds"], len(skeletons))
        self.assertGreater(profile["scaffold_internal_diversity"], 0.0)
        self.assertEqual(profile["sidechain_internal_diversity"], 0.0)

    def test_scaffold_entropy_ratio_is_scaffold_breadth_per_effective_molecule(self):
        # Ten uniform draws over one scaffold: 10 effective molecules, 1 effective family.
        hats = [f"c1ccc(C{'C' * i})cc1" for i in range(1, 11)]
        profile = diversity.arm_profile(
            diversity.ArmUniques(self._frame([(smi, 1, 1) for smi in hats]))
        )
        self.assertAlmostEqual(profile["effective_molecules"], 10.0)
        self.assertAlmostEqual(profile["effective_scaffolds"], 1.0)
        self.assertAlmostEqual(profile["scaffold_entropy_ratio"], 0.1)

    def test_effective_desired_yield_prices_reward_in_scaffold_families(self):
        # Same desired fraction (0.4), same draw count (5): the arm with five scaffold
        # families yields five times the distinct desired chemistry of the one with one.
        hats = [f"c1ccc(C{'C' * i})cc1" for i in range(1, 6)]
        skeletons = ["c1ccccc1CC", "c1ccncc1CC", "c1ccc2ccccc2c1CC", "C1CCCCC1CC", "c1ccoc1CC"]
        flags = [1, 1, 0, 0, 0]
        one_family = diversity.arm_profile(diversity.ArmUniques(
            self._frame([(smi, 1, flag) for smi, flag in zip(hats, flags)])))
        five_families = diversity.arm_profile(diversity.ArmUniques(
            self._frame([(smi, 1, flag) for smi, flag in zip(skeletons, flags)])))
        self.assertAlmostEqual(one_family["effective_desired_yield"], 0.4 / 5)
        self.assertAlmostEqual(
            five_families["effective_desired_yield"],
            5 * one_family["effective_desired_yield"],
        )


if __name__ == "__main__":
    unittest.main()
