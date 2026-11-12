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

from _david.code import provenance
from _david.code import run as runner
from _david.code.analysis import _helpers as analysis_helpers, figures, report as analysis
from _david.code.analysis import atlas_v2, collapse, diversity
from _david.code.metacentrum import campaign, submit as submit_module
from _david.code.metacentrum.submit import Job
from _david.code.scorers import conformers, objectives as scorers, rocs
from _david.code.config import PROJECT_ROOT, Config
from _david.code.training.explorer import DualCheckpointExplorer
from _david.code.run import arm_is_complete, config_from_args, write_arm


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

    def __init__(self, ratios, gmeans=None):
        self.ratios = list(ratios)
        self.gmeans = list(gmeans) if gmeans is not None else list(ratios)
        self.epoch = 0

    def sample(self, n):
        return ["CCO"] * n

    def evaluate(self, smiles, evaluator=None, no_multifrag_smiles=True):
        index = min(self.epoch - 1, len(self.ratios) - 1)  # forward() already incremented
        desired = self.ratios[index]
        self.current_gmean = self.gmeans[min(index, len(self.gmeans) - 1)]
        n = len(smiles)
        hits = int(round(desired * n))
        return pd.DataFrame(
            {
                "Valid": [1] * n,
                "Desired": [1] * hits + [0] * (n - hits),
                "ROCS": [0.5] * n,
            }
        )


class _FakeExplorer(DualCheckpointExplorer):

    def __init__(self, ratios, gmeans=None, **kwargs):
        self.agent = _FakeAgent(ratios, gmeans)
        self.env = None
        self.no_multifrag_smiles = True
        self.desired_ratio_path = kwargs["desired_ratio_path"]
        self.desired_ratio_at_stop_path = kwargs["desired_ratio_at_stop_path"]
        self.gmean_path = kwargs["gmean_path"]
        self.training_samples_path = None
        self.n_eval = 100  # a small arm would quantise hits/n_eval
        self.saved = []

    def forward(self):
        self.agent.epoch += 1
        return ["CCO"], None

    def policy_gradient(self, smiles, seqs, epoch=None, epochs=None):
        return 1.0

    def getNovelMoleculeMetrics(self, scores):
        desired = float(scores["Desired"].mean())
        return {
            "desired_ratio": desired,
            "avg_gmean": float(self.agent.current_gmean),
            "valid_ratio": 1.0,
        }

    def getModel(self):
        return {"epoch": self.agent.epoch}

    def logPerformanceAndCompounds(self, epoch, metrics, scores):
        pass


class Checkpoints(unittest.TestCase):

    def _run(self, ratios, directory, gmeans=None, patience=1, **kwargs):
        paths = {
            "desired_ratio_path": Path(directory) / "best.pkg",
            "desired_ratio_at_stop_path": Path(directory) / "at_stop.pkg",
            "gmean_path": Path(directory) / "gmean.pkg",
        }
        explorer = _FakeExplorer(ratios, gmeans, **paths)
        summary = explorer.fit(epochs=len(ratios), patience=patience, min_epochs=1, **kwargs)
        return summary, paths

    def test_best_checkpoint_holds_the_peak_and_at_stop_holds_the_live_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, paths = self._run([1.0, 0.0, 0.0, 1.0], directory)
            self.assertEqual(summary["best_desired_ratio_before_freeze"], 1.0)
            self.assertEqual(summary["best_desired_ratio_epoch"], 1)
            self.assertTrue(summary["desired_ratio_frozen"])
            self.assertGreater(summary["desired_ratio_freeze_epoch"], 1)
            best = torch.load(paths["desired_ratio_path"], weights_only=False)
            at_stop = torch.load(paths["desired_ratio_at_stop_path"], weights_only=False)
            self.assertEqual(best["epoch"], 1, "the best checkpoint must be the peak epoch")
            self.assertNotEqual(at_stop["epoch"], best["epoch"], "both readings must be kept")

    def test_frozen_checkpoint_is_not_rewritten_by_a_later_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            _, paths = self._run([1.0, 0.0, 0.0, 1.0], directory)
            self.assertEqual(
                torch.load(paths["desired_ratio_path"], weights_only=False)["epoch"],
                1,
                "epoch 4 matched the peak and must not have reopened the frozen file",
            )

    def test_run_continues_while_desired_improves_after_gmean_plateaus(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = self._run(
                ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                gmeans=[0.9] * 6,  # best at epoch 1, never improves again
                directory=directory,
                patience=2,
            )
        self.assertEqual(summary["best_avg_gmean_epoch"], 1)
        self.assertEqual(
            summary["epochs_run"], 6, "a plateaued gmean must not cut the desired window short"
        )
        self.assertEqual(summary["best_desired_ratio_epoch"], 6)

    def test_post_freeze_peak_is_reported_even_though_the_checkpoint_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, paths = self._run(
                ratios=[0.5, 0.0, 0.0, 0.0, 0.99],
                gmeans=[0.1, 0.2, 0.3, 0.4, 0.5],
                directory=directory,
                patience=1,
            )
            self.assertTrue(summary["desired_ratio_frozen"])
            self.assertEqual(summary["best_desired_ratio_before_freeze"], 0.5)
            self.assertAlmostEqual(summary["peak_desired_ratio_any_epoch"], 0.99)
            self.assertEqual(
                torch.load(paths["desired_ratio_path"], weights_only=False)["epoch"],
                1,
                "the frozen checkpoint must still hold the pre-freeze best",
            )

    def test_run_does_not_stop_while_desired_patience_is_still_open(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = self._run([0.25, 0.5, 0.75, 1.0], directory)
            self.assertEqual(summary["epochs_run"], 4)
            self.assertEqual(summary["best_desired_ratio_epoch"], 4)

    def test_at_stop_checkpoint_exists_even_when_patience_never_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            summary, paths = self._run([0.25, 0.5, 0.75, 1.0], directory)
            self.assertFalse(summary["desired_ratio_frozen"])
            self.assertTrue(paths["desired_ratio_at_stop_path"].is_file())


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
                for mol in conformers._read(path):
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
            out, reason = conformers._relax((mol, small(minimize=True)))
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
            out, reason = conformers._relax((mol, small()))
        self.assertIsNotNone(out, "the molecule must survive a per-conformer fault")
        self.assertEqual(reason, "relax_raised")

    def test_unkekulizable_molecule_is_kept_not_filtered(self):
        self.assertIsNone(conformers._stereo_key(object()))

    def test_protonation_is_off_and_stated(self):
        self.assertFalse(conformers.PROTONATE)


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
        digests = provenance.source_digests(Path(provenance.__file__).parent)
        for name in (
            "run.py",
            "config.py",
            "provenance.py",
            "scorers/objectives.py",
            "scorers/conformers.py",
            "scorers/rocs.py",
            "training/explorer.py",
            "analysis/report.py",
            "analysis/figures.py",
            "metacentrum/job.py",
            "metacentrum/submit.py",
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
        self.assertEqual(sorted(scorers.SCHEMES), ["PRCD", "PRTD", "WS"])

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
            self.assertEqual(conformers.check_pool(timeout=30.0), "fork")

    def test_a_pool_that_cannot_be_created_is_reported_not_raised(self):
        class _Dead:
            def get_start_method(self):
                return "forkserver"

            def Pool(self, workers):
                raise OSError(24, "Too many open files")

        with (
            patch.object(conformers, "_POOL_BROKEN", False),
            patch.object(conformers, "pool_context", return_value=_Dead()),
        ):
            self.assertIn("broken", conformers.check_pool(timeout=1.0))

    def test_a_pool_that_hangs_is_bounded_and_terminated(self):
        wedged = _WedgedPool()
        started = time.perf_counter()
        with (
            patch.object(conformers, "_POOL_BROKEN", False),
            patch.object(conformers, "pool_context", return_value=wedged),
        ):
            result = conformers.check_pool(timeout=0.2)
        self.assertIn("broken", result)
        self.assertTrue(wedged.terminated, "a wedged pool must be terminated, not left running")
        self.assertLess(time.perf_counter() - started, 10.0)

    def test_a_broken_probe_disables_production_pools(self):
        with (
            patch.object(conformers, "_POOL_BROKEN", False),
            patch.object(conformers, "pool_context", return_value=_WedgedPool()),
        ):
            self.assertIn("broken", conformers.check_pool(timeout=0.01))
            self.assertFalse(conformers.pool_is_healthy())
            # Forty records would pool, and the wedged stub has no map().
            self.assertEqual(
                conformers._map(conformers._touch, list(range(40)), small(n_jobs=2)),
                list(range(40)),
            )

    def test_a_multi_core_run_refuses_to_start_without_workers(self):
        with patch.object(conformers, "_POOL_BROKEN", True):
            self.assertIsNone(
                runner.serial_run_refusal(small(n_jobs=1)),
                "a one-core run is serial anyway and must still run",
            )
            refusal = runner.serial_run_refusal(small(n_jobs=16))
            self.assertIsNotNone(refusal, "16 cores and no workers cannot finish a run")
            self.assertIn("DRUGEX_MP_CONTEXT=fork", refusal)
        with patch.object(conformers, "_POOL_BROKEN", False):
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

    def test_the_original_run_cannot_finish_complete_with_a_missing_arm(self):
        cfg = small(n_final=2, fixed_thresholds=(0.5, 0.5))
        env = types.SimpleNamespace(
            getScorerKeys=lambda: ["ROCS", "SA"],
            getScores=lambda smiles: pd.DataFrame({"ROCS": [0.5] * len(smiles)}),
        )
        explorer = unittest.mock.Mock()
        explorer.fit.return_value = {"epochs_run": 1}

        def complete_file(path, n):
            return path.name != "final_gmean.tsv"

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
                patch.object(runner, "DualCheckpointExplorer", return_value=explorer),
                patch.object(runner, "arm_is_complete", side_effect=complete_file),
                patch.object(analysis, "analyse"),
            ):
                status = runner._pipeline(cfg, run_dir, resolved, manifest, time.monotonic())
            self.assertEqual(status, 1)
            self.assertEqual(manifest.data["state"], "failed")
            self.assertIn("gmean", manifest.data["error"])


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
                arms, ["ROCS"], out, arm=figures.SECONDARY_ARM, name="05b_gallery"
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
        self.assertGreaterEqual(Job.mem, campaign.MEMORY_GB, "the ad-hoc path must match")

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


if __name__ == "__main__":
    unittest.main()
