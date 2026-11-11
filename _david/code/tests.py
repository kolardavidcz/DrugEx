"""Behavioural tests: ``python -m unittest _david.code.tests -v``.

Each test pins a decision that was argued about, not a line of code. The point is that if
someone later "simplifies" one of these behaviours back, a test says which argument they
are reopening. Nothing here runs a real training loop or scores more than a few molecules.
"""

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
from rdkit import Chem
from rdkit.Chem import AllChem

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import FAILURE_VALUES, Property

from _david.code import provenance
from _david.code import run as runner
from _david.code.analysis import _helpers as analysis_helpers, figures, report as analysis
from _david.code.metacentrum import campaign
from _david.code.metacentrum.submit import Job
from _david.code.scorers import conformers, objectives as scorers
from _david.code.config import PROJECT_ROOT, Config
from _david.code.training.explorer import DualCheckpointExplorer
from _david.code.run import _arm_is_complete, _write_arm, config_from_args


def small(**changes) -> Config:
    """A config cheap enough to build a conformer from; ``changes`` may override any default."""
    cheap = {
        "max_conformers": 2,
        "max_isomers": 1,
        "reference_attempts": 4,
        "reference_conformers": 2,
        "n_jobs": 1,
    }
    return replace(Config(), **{**cheap, **changes})


class FailureValues(unittest.TestCase):
    """The bug that started the rewrite: a failed score outranking a real molecule."""

    def test_sa_failure_is_the_worst_value_not_the_best(self):
        mols = [Chem.MolFromSmiles("CCO"), None]
        raw = Property("SA").getScores(mols)
        self.assertAlmostEqual(raw[1], 10.0, msg="an unscorable SA must sit at the worst value")
        reward = Property("SA", modifier=ClippedScore(lower_x=5.0, upper_x=3.0))(mols)
        self.assertAlmostEqual(reward[1], 0.0, msg="the descending modifier must map it to 0")
        self.assertGreater(reward[0], reward[1], "ethanol must beat an unscorable molecule")

    def test_upward_properties_keep_zero_on_failure(self):
        # QED is [0, 1] and improves upwards, so 0.0 is already its worst. A blanket 10.0
        # would make a failure better than the best possible molecule.
        for prop in ("QED", "MW", "logP"):
            self.assertEqual(FAILURE_VALUES.get(prop, 0.0), 0.0, prop)
            self.assertAlmostEqual(Property(prop).getScores([None])[0], 0.0, msg=prop)

    def test_unparameterized_energy_sorts_last_not_first(self):
        # A fabricated 0.0 kcal/mol would sort as the lowest-energy conformer and win the
        # reference bank. Infinity sorts last, which is what an unscorable pose deserves.
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
        # This is what keeps p85 out of the policy gradient by construction rather than by
        # convention: the reward scheme never reads the thresholds it is handed.
        scheme = ParetoCrowdingDistance()
        values = np.array([[0.9, 0.2], [0.2, 0.9], [0.5, 0.5]])
        loose = scheme(["C", "CC", "CCC"], values, [0.0, 0.0])
        tight = scheme(["C", "CC", "CCC"], values, [9.9, 9.9])
        np.testing.assert_allclose(loose, tight)

    def test_invalid_rows_are_never_desired(self):
        # An invalid molecule has every objective zeroed, so it would clear an all-zero
        # threshold set and be counted as a success.
        frame = pd.DataFrame({"Valid": [1, 0], "ROCS": [0.5, 0.0], "SA": [0.5, 0.0]})
        marked = scorers.mark_desired(frame, ["ROCS", "SA"], [0.0, 0.0])
        self.assertEqual(marked["Desired"].tolist(), [1, 0])

    def test_fixed_thresholds_bypass_the_percentile(self):
        cfg = small(fixed_thresholds=(1.5, 0.25))
        frame = pd.DataFrame({"Valid": [1, 1], "A": [0.1, 2.0], "B": [0.1, 0.9]})
        with (
            patch.object(DrugExEnvironment, "getScores", return_value=frame),
            patch.object(DrugExEnvironment, "getScorerKeys", return_value=["A", "B"]),
        ):
            thresholds, _ = scorers.calibrate_thresholds(cfg, [None, None], ["C", "CC"])
        self.assertEqual(thresholds, [1.5, 0.25])

    def test_fixed_thresholds_must_match_the_objective_count(self):
        cfg = small(fixed_thresholds=(1.0,))
        frame = pd.DataFrame({"Valid": [1], "A": [0.1], "B": [0.1]})
        with (
            patch.object(DrugExEnvironment, "getScores", return_value=frame),
            patch.object(DrugExEnvironment, "getScorerKeys", return_value=["A", "B"]),
        ):
            with self.assertRaises(ValueError):
                scorers.calibrate_thresholds(cfg, [None, None], ["C"])


class Metrics(unittest.TestCase):
    def test_desired_ratio_counts_a_molecule_once(self):
        # Under mode collapse the same desirable molecule is sampled repeatedly. DrugEx's
        # definition deduplicates; counting duplicates inflates the headline number.
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
            ratios = _write_arm(
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
    """Enough of a generator for fit() to run without a network or a GPU.

    ``gmeans`` may differ from ``ratios``: with one series driving both criteria no test
    can express the scenario the dual-checkpoint design exists for, and the "stop only
    when both windows expire" rule becomes untestable.
    """

    def __init__(self, ratios, gmeans=None):
        self.ratios = list(ratios)
        self.gmeans = list(gmeans) if gmeans is not None else list(ratios)
        self.epoch = 0

    def sample(self, n):
        return ["CCO"] * n

    def evaluate(self, smiles, evaluator=None, no_multifrag_smiles=True):
        # forward() has already incremented, so epoch 1 reads ratios[0].
        index = min(self.epoch - 1, len(self.ratios) - 1)
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
    """Replaces every expensive part of an epoch, keeping only the checkpoint logic."""

    def __init__(self, ratios, gmeans=None, **kwargs):
        self.agent = _FakeAgent(ratios, gmeans)
        self.env = None
        self.no_multifrag_smiles = True
        self.desired_ratio_path = kwargs["desired_ratio_path"]
        self.desired_ratio_at_stop_path = kwargs["desired_ratio_at_stop_path"]
        self.gmean_path = kwargs["gmean_path"]
        self.training_samples_path = None
        # 100, not a handful: desired_ratio is hits/n_eval, so a small arm quantises the
        # ratio and a test asking for 0.99 would silently get 1.0.
        self.n_eval = 100
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
    """The semantics that two implementations read differently out of one sentence."""

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
        import torch

        # desired_ratio peaks at epoch 1, decays, and recovers. patience=1 means the
        # window expires at epoch 3, by which point the live policy is two epochs past
        # its own best -- which is exactly the difference between the two readings.
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
        import torch

        with tempfile.TemporaryDirectory() as directory:
            _, paths = self._run([1.0, 0.0, 0.0, 1.0], directory)
            self.assertEqual(
                torch.load(paths["desired_ratio_path"], weights_only=False)["epoch"],
                1,
                "epoch 4 matched the peak and must not have reopened the frozen file",
            )

    def test_run_continues_while_desired_improves_after_gmean_plateaus(self):
        """The 'both windows' rule, with criteria that genuinely diverge.

        With one series driving both criteria this is untestable: gmean plateaus only when
        desired_ratio does. Here avg_gmean is flat from epoch 1 while desired_ratio keeps
        improving, so a single-criterion rule would stop early and the real rule must not.
        """
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
        # The frozen file must not reopen, but the summary must still show the run reached
        # a higher value -- otherwise the report and rl_fit.tsv contradict each other.
        import torch

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
        # gmean plateaus immediately while desired keeps improving. Stopping on gmean
        # alone would cut the desired criterion short.
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
    def test_minimization_lowers_the_energy_of_the_reference_bank(self):
        cfg = small(minimize=True)
        with tempfile.TemporaryDirectory() as directory:
            relaxed = conformers.build_reference(cfg, Path(directory) / "ref.sdf")
            plain = conformers.build_reference(
                replace(cfg, minimize=False), Path(directory) / "plain.sdf"
            )
        best_relaxed = min(conformers.single_point_energy(m, cfg) for m in relaxed)
        best_plain = min(conformers.single_point_energy(m, cfg) for m in plain)
        self.assertLess(best_relaxed, best_plain, "relaxation must reduce MMFF energy")

    def test_bank_size_does_not_depend_on_the_arm(self):
        """The window must never make the unminimised arm a smaller bank.

        Raw ETKDG energies spread far wider than relaxed ones, so a fixed kcal/mol window
        admitted 2 conformers unminimised against 4 minimised. The score is a max over the
        bank, so that alone would make the minimised arm look better.
        """
        cfg = replace(small(), reference_attempts=8, reference_conformers=3, minimize=True)
        with tempfile.TemporaryDirectory() as directory:
            relaxed = conformers.build_reference(cfg, Path(directory) / "a.sdf")
            plain = conformers.build_reference(
                replace(cfg, minimize=False), Path(directory) / "b.sdf"
            )
        self.assertEqual(
            len(relaxed), len(plain), "the two arms must offer the same number of poses"
        )

    def test_tuple_fields_are_settable_from_the_command_line(self):
        # "tuple[float, ...]" contains "float", so a scalar branch matched first and the
        # knob could not be set at all from --set.
        self.assertEqual(
            config_from_args(["fixed_thresholds=0.8,1.0"]).fixed_thresholds, (0.8, 1.0)
        )
        self.assertEqual(
            config_from_args(["fixed_thresholds=(0.8, 1.0)"]).fixed_thresholds, (0.8, 1.0)
        )
        self.assertEqual(config_from_args(["fixed_thresholds="]).fixed_thresholds, ())

    def test_parallel_embedding_is_identical_to_serial(self):
        """Sharding molecules across processes must not change a single coordinate.

        The conformer title carries the molecule's index, and the scorer maps conformers
        back to input SMILES through it, so a worker numbering from zero would attribute
        every molecule's geometry to the wrong one.
        """
        import numpy as np

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
        # A geometry that has not reached a stationary point is still a geometry. Dropping
        # it would send the molecule's ROCS score to 0 in the minimized arm only, which
        # changes which chemistry the two arms can see.
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)

        class Unconverged:
            """A force field that never reaches a stationary point."""

            def Minimize(self, maxIts=0):
                return 1

            def CalcEnergy(self):
                return 12.5

        with patch.object(conformers, "_force_field", return_value=Unconverged()):
            out, reason = conformers._relax((mol, small(minimize=True)))
        self.assertIsNotNone(out, "an unconverged conformer must survive")
        self.assertEqual(reason, "unconverged")

    def test_a_conformer_fault_never_ends_the_run(self):
        # This fired for real: KekulizeException inside the stereo check aborted a whole
        # pipeline run. One conformer may fail; a run may not.
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
        # Dropping here would filter on "RDKit could not kekulize this" in the minimized
        # arm only, which is a chemistry-dependent confound in the arm comparison.
        self.assertIsNone(conformers._stereo_key(object()))

    def test_protonation_is_off_and_stated(self):
        # Dimorphite-DL is installed, and native DrugEx defaults protonation ON. At pH 7.4
        # it returns an octa-anionic EGCG, which is not the dominant species and destroys
        # every phenol donor the colour channel depends on.
        self.assertFalse(conformers.PROTONATE)


class Rocs(unittest.TestCase):
    """The transplanted pose kernel: it must agree with native and stay one objective."""

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

    def test_matches_native_bit_for_bit_at_equal_weight(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

        from _david.code.scorers.rocs import ComboROCSScorer

        with tempfile.TemporaryDirectory() as directory:
            cfg, path = self._bank(directory)
            mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1O")] + [None]
            common = dict(
                references=str(path),
                score_type="TanimotoCombo",
                color_ff=cfg.color_ff,
                n_jobs=1,
                show_progress=False,
            )
            native = RDKitROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg), **common
            )
            ours = ComboROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg),
                alignment_shape_weight=0.5,
                **common,
            )
            np.testing.assert_allclose(ours.getScores(mols), native.getScores(mols), atol=0)
            self.assertEqual(ours.getKey(), native.getKey())

    def test_components_sum_to_the_reported_combo(self):
        from _david.code.scorers.rocs import ComboROCSScorer

        with tempfile.TemporaryDirectory() as directory:
            cfg, path = self._bank(directory)
            scorer = ComboROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg),
                references=str(path),
                score_type="TanimotoCombo",
                color_ff=cfg.color_ff,
                n_jobs=1,
                show_progress=False,
                alignment_shape_weight=0.5,
            )
            smiles = ["CCO", "CCN"]
            scores = scorer.getScores([Chem.MolFromSmiles(s) for s in smiles]).ravel()
            for value, name in zip(
                scores, [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
            ):
                shape, colour = scorer.components[name]
                self.assertAlmostEqual(shape + colour, float(value), places=9)

    def test_components_never_become_objectives(self):
        # Returning a key per component would make ParetoCrowdingDistance optimise shape
        # and colour as separate objectives -- a two-objective run silently becoming four.
        from _david.code.scorers.rocs import ComboROCSScorer

        with tempfile.TemporaryDirectory() as directory:
            cfg, path = self._bank(directory)
            scorer = ComboROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg),
                references=str(path),
                score_type="TanimotoCombo",
                color_ff=cfg.color_ff,
                n_jobs=1,
                show_progress=False,
                alignment_shape_weight=0.5,
            )
            self.assertEqual(len(scorer.getKey()), 1)
            environment = scorers.build_environment(
                [scorer, Property("SA")], [0.0, 0.0], scorers.reward_scheme("PRCD")
            )
            keys = environment.getScorerKeys()
            self.assertEqual(len(keys), 2, f"the reward must see two objectives, saw {keys}")
            self.assertNotIn("ShapeTanimoto", keys)
            self.assertNotIn("ColorTanimoto", keys)

    def test_components_survive_the_environment_round_trip(self):
        # The scorer keys components by canonical SMILES. Looking them up with the raw
        # strings a generator emits must still line up, or a row stops adding up.
        from _david.code.scorers.rocs import ComboROCSScorer

        with tempfile.TemporaryDirectory() as directory:
            cfg, path = self._bank(directory)
            scorer = ComboROCSScorer(
                conformer_generator=conformers.RelaxingConformerGenerator(cfg),
                references=str(path),
                score_type="TanimotoCombo",
                color_ff=cfg.color_ff,
                n_jobs=1,
                show_progress=False,
                alignment_shape_weight=0.5,
            )
            environment = scorers.build_environment(
                [scorer, Property("SA")], [0.0, 0.0], scorers.reward_scheme("PRCD")
            )
            raw = ["C(C)O", "C(C)N"]  # deliberately non-canonical
            frame = environment.getScores(raw)
            split = scorer.component_frame(raw)
            combo = frame[scorer.getKey()[0]].to_numpy(dtype=float)
            total = np.asarray(split["ShapeTanimoto"]) + np.asarray(split["ColorTanimoto"])
            np.testing.assert_allclose(total, combo, atol=1e-9)

    def test_alignment_weight_is_validated(self):
        with self.assertRaises(ValueError):
            replace(Config(), alignment_shape_weight=1.5)

    def test_malformed_align_result_costs_one_pose_not_the_epoch(self):
        from _david.code.scorers import rocs

        with tempfile.TemporaryDirectory() as directory:
            _, path = self._bank(directory)
            ref = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m][0]
            with patch.object(rocs.rdShapeAlign, "AlignMol", return_value=None):
                shape, colour = rocs.pair_components(ref, ref, opt_param=0.5)
            self.assertEqual((shape, colour), (0.0, 0.0))


class Provenance(unittest.TestCase):
    def test_kernel_hashes_cover_every_module_a_score_depends_on(self):
        digests = provenance.kernel_fingerprint(PROJECT_ROOT)
        self.assertEqual(set(digests), set(provenance.KERNEL_FILES))
        for name, value in digests.items():
            self.assertNotEqual(value, "missing", name)
            self.assertEqual(len(value), 64, name)

    def test_source_digests_cover_this_package(self):
        digests = provenance.source_digests(Path(provenance.__file__).parent)
        # Every module, including the ones inside subpackages -- a flat glob would miss
        # most of the package and still report a plausible-looking manifest.
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
        # kernel_sha256 hashes files on disk; if the import resolved somewhere else the
        # hashes would describe code that never ran.
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
            manifest.finish("failed", error="boom")
            self.assertEqual(json.loads(path.read_text())["state"], "failed")
            self.assertFalse(list(Path(directory).glob("*.tmp")), "no temporary left behind")


class Configuration(unittest.TestCase):
    def test_reversed_sa_clip_is_rejected(self):
        # ClippedScore only asserts low < high, so an ascending pair would silently invert
        # the objective and reward unsynthesizable molecules.
        with self.assertRaises(ValueError):
            replace(Config(), sa_reward_zero=3.0, sa_reward_one=5.0)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(SystemExit):
            config_from_args(["not_a_field=1"])

    def test_override_round_trips_through_the_declared_type(self):
        cfg = config_from_args(["epochs=3", "min_epochs=1", "minimize=False"])
        self.assertEqual(cfg.epochs, 3)
        self.assertIs(cfg.minimize, False)

    def test_min_epochs_cannot_exceed_epochs(self):
        with self.assertRaises(ValueError):
            replace(Config(), epochs=3, min_epochs=50)


class RewardSchemes(unittest.TestCase):
    """What changes when the reward scheme changes, and what must not."""

    def test_every_scheme_name_is_one_of_drugexs_own(self):
        self.assertIsInstance(scorers.reward_scheme("PRCD"), ParetoCrowdingDistance)
        self.assertIsInstance(scorers.reward_scheme("WS"), WeightedSum)
        self.assertEqual(sorted(scorers.SCHEMES), ["PRCD", "PRTD", "WS"])

    def test_an_unknown_scheme_is_rejected_by_name(self):
        # A typo in a submitted delta must cost a second, not an hour of queue plus the
        # reference bank plus 5000 scored prior molecules.
        with self.assertRaises(ValueError) as caught:
            scorers.reward_scheme("pareto")
        self.assertIn("PRCD", str(caught.exception))

    def test_weighted_sum_does_read_the_thresholds(self):
        # The counterpart of test_pareto_reward_ignores_thresholds, and the reason the WS
        # batch is a different experiment rather than a different setting: DrugEx builds
        # WeightedSum's weights from the thresholds, so under WS they are part of the
        # reward. If this ever stops being true, the report prose below is a lie.
        scheme = scorers.reward_scheme("WS")
        values = np.array([[0.9, 0.2], [0.2, 0.9], [0.5, 0.5]])
        loose = scheme(["C", "CC", "CCC"], values, np.array([0.0, 0.0]))
        tight = scheme(["C", "CC", "CCC"], values, np.array([0.8, 0.1]))
        self.assertFalse(np.allclose(loose, tight), "WS must respond to its thresholds")

    def test_every_objective_reaches_the_reward_on_the_same_scale(self):
        # WS dots its weights with the raw columns, so TanimotoCombo on [0, 2] carries
        # twice the leverage of SA and QED and the reward leaves [0, 1]. Pareto ranking
        # is affine-invariant, so the same modifier would be a no-op there and is not
        # applied -- which keeps the reported ROCS of 144 of 148 runs in published units.
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
        # The scorer's modifier halves the stored combo. Figures read ROCS as the
        # unmodified shape + colour sum, so a run from either side of that change is
        # drawn on [0, 2] -- and the manifest's threshold, which is in reward units,
        # has to be scaled to match or the line lands at half its true position.
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
        # The per-epoch table has no components at all; it must still come back on [0, 2].
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
    """What ``batch_size`` chunks, and what it does not."""

    def _policy_gradient(self, molecules: int, batch_size: int) -> tuple[list[int], int]:
        """Native's policy_gradient over a stub, reporting scoring calls and gradient steps."""
        import torch

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
        # The expensive part of an epoch is one call. `batch_size` never reaches the
        # scorer: it is the minibatch of the policy-gradient loop and the chunk
        # `forward()` generates in, so there is no "extra pool per 128 molecules" to
        # remove -- every molecule of the epoch is already scored in one parallel pass.
        for batch_size, expected_steps in ((16, 32), (128, 4), (500, 1)):
            scored, steps = self._policy_gradient(500, batch_size)
            self.assertEqual(scored, [500], f"batch_size={batch_size} must not chunk scoring")
            self.assertEqual(steps, expected_steps, f"batch_size={batch_size} splits the gradient")


class StartMethod(unittest.TestCase):
    """The probe that stops a broken pool from quietly costing a run its parallelism."""

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
        # The real failure mode, measured: a worker that dies while importing __main__
        # (forkserver/spawn under `python - <<EOF`) is respawned forever, so map() never
        # returns. A probe without a timeout would hang the run it exists to protect.
        class _Wedged:
            terminated = False

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

        wedged = _Wedged()
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
        class _Wedged:
            def get_start_method(self):
                return "forkserver"

            def Pool(self, workers):
                return self

            def map_async(self, function, items):
                return self

            def get(self, timeout=None):
                raise mp.TimeoutError()

            def terminate(self):
                pass

            def join(self):
                pass

        with (
            patch.object(conformers, "_POOL_BROKEN", False),
            patch.object(conformers, "pool_context", return_value=_Wedged()),
        ):
            self.assertIn("broken", conformers.check_pool(timeout=0.01))
            self.assertFalse(conformers.pool_is_healthy())
            # Forty records would ordinarily use two workers. Another Pool() here would
            # raise AttributeError because the wedged stub deliberately has no map().
            self.assertEqual(
                conformers._map(conformers._touch, list(range(40)), small(n_jobs=2)),
                list(range(40)),
            )

    def test_a_multi_core_run_refuses_to_start_without_workers(self):
        with patch.object(conformers, "_POOL_BROKEN", True):
            self.assertIsNone(
                runner._serial_run_refusal(small(n_jobs=1)),
                "a one-core run is serial anyway and must still run",
            )
            refusal = runner._serial_run_refusal(small(n_jobs=16))
            self.assertIsNotNone(refusal, "16 cores and no workers cannot finish a run")
            self.assertIn("DRUGEX_MP_CONTEXT=fork", refusal)
        with patch.object(conformers, "_POOL_BROKEN", False):
            self.assertIsNone(runner._serial_run_refusal(small(n_jobs=16)))


class ArmCompleteness(unittest.TestCase):
    @staticmethod
    def _arm_frame(rows: int) -> pd.DataFrame:
        """An arm in the column order run.py writes: Desired is fifth of nine."""
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
            self.assertFalse(_arm_is_complete(path, 2))
            self._arm_frame(1).to_csv(path, sep="\t", index=False)
            self.assertFalse(_arm_is_complete(path, 2), "a one-row table cannot satisfy n=2")
            path.write_text("SMILES\tCanonical\tValid\tDesired\nnot a complete row")
            self.assertFalse(_arm_is_complete(path, 1), "a truncated row must be rebuilt")
            self._arm_frame(2).to_csv(path, sep="\t", index=False)
            self.assertTrue(_arm_is_complete(path, 2))

    def test_a_row_cut_off_after_desired_is_incomplete(self):
        """Desired is the fifth of nine columns, so NaN there cannot be the only test."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            table = self._arm_frame(2)
            table.to_csv(path, sep="\t", index=False)
            lines = path.read_text().splitlines()
            # Everything through Desired survives; the writer died before the rest.
            lines[-1] = "\t".join(lines[-1].split("\t")[:5])
            path.write_text("\n".join(lines) + "\n")
            self.assertFalse(
                _arm_is_complete(path, 2),
                "a row truncated after Desired has no Canonical and must be rebuilt",
            )

    def test_an_empty_generated_smiles_is_still_a_complete_row(self):
        """The RNN can emit "", which round-trips as NaN in SMILES and Canonical."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.tsv"
            table = self._arm_frame(2)
            table.loc[1, ["SMILES", "Canonical"]] = ""
            table.loc[1, "Valid"] = 0
            table.to_csv(path, sep="\t", index=False)
            self.assertTrue(_arm_is_complete(path, 2))

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
                # Config is a frozen dataclass, so the method is patched on the class.
                patch.object(Config, "pretrained", return_value=("agent", "vocabulary")),
                patch.object(runner.VocSmiles, "fromFile"),
                patch.object(runner, "_load"),
                patch.object(runner, "_sample", return_value=["CC", "CC"]),
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
                patch.object(runner, "_write_arm", return_value={}),
                patch.object(runner, "_components", return_value={}),
                patch.object(runner, "DualCheckpointExplorer", return_value=explorer),
                patch.object(runner, "_arm_is_complete", side_effect=complete_file),
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
    """Figure 05b: one grid, from the checkpoint that actually won."""

    def test_the_prior_is_never_the_best_model(self):
        # It is the untrained baseline. A run whose prior scores best is a result to
        # report, not a model to put in a lead gallery.
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
        # desired_ratio was never written, so the figure describes what the run has.
        self.assertEqual(figures.shown_arm(arms, ["ROCS"], "desired_ratio"), "gmean")
        self.assertIsNone(figures.shown_arm({"prior": _arm([0.9], [1])}, ["ROCS"], "gmean"))

    def test_no_trained_arm_means_no_figure(self):
        self.assertIsNone(figures.best_arm({"prior": _arm([0.9], [1])}, ["ROCS"]))

    def test_all_invalid_trained_arms_do_not_promise_a_gallery(self):
        invalid = _arm([0.0], [0])
        invalid["Valid"] = 0
        self.assertIsNone(figures.best_arm({"desired_ratio": invalid}, ["ROCS"]))


class Campaign(unittest.TestCase):
    """The 148 planned runs. Every assertion here is a job that would die on a node."""

    def test_the_plan_is_the_sweep_that_was_asked_for(self):
        counts = {name: len(campaign.runs(batch)) for name, batch in campaign.BATCHES.items()}
        self.assertEqual(counts, {"parent": 96, "ws": 4, "prtd": 4, "color": 48})
        self.assertEqual(sum(counts.values()), 152)

    def test_every_delta_is_a_setting_the_pipeline_accepts(self):
        # config_from_args raises SystemExit on an unknown name and on a value that will
        # not coerce, which is the whole check: a typo here is 148 jobs that queue for
        # hours and then fail on import.
        for batch in campaign.BATCHES.values():
            for name, settings in campaign.runs(batch):
                cfg = config_from_args(campaign.deltas(settings))
                self.assertEqual(cfg.epsilon, settings["epsilon"], name)
                self.assertEqual(cfg.include_qed, settings["include_qed"], name)
                self.assertEqual(cfg.color_ff, settings["color_ff"], name)
                self.assertEqual(cfg.max_conformers, settings["max_conformers"], name)
                for key, wanted in settings.items():
                    self.assertEqual(getattr(cfg, key), wanted, f"{name}: {key}")
                scorers.reward_scheme(cfg.scheme)

    def test_a_batch_changes_the_reward_scheme_only_by_being_about_it(self):
        # PRCD is the grid's scheme. A run on another one is not a cell of the same
        # experiment -- WS reads the thresholds, PRTD ranks a front by chemical distance
        # -- so a scheme may only vary where the whole batch exists to vary it, and
        # never as one axis among several.
        schemes = {"ws": "WS", "prtd": "PRTD"}
        for name, batch in campaign.BATCHES.items():
            wanted = schemes.get(name, "PRCD")
            self.assertNotIn("scheme", batch.axes, name)
            for job, settings in campaign.runs(batch):
                self.assertEqual(settings["scheme"], wanted, job)

    def test_names_are_unique_and_decode_to_their_settings(self):
        names = []
        suffixes = {
            "minimize": lambda v: f"m{int(v)}",
            "epsilon": lambda v: f"e{int(v * 100):02d}",
            "reference_conformers": lambda v: f"r{v}",
            "max_conformers": lambda v: f"c{v}",
            "include_qed": lambda v: f"q{int(v)}",
            "color_ff": lambda v: {"default": "fd", "fdef": "ff", "oe_mills_dean": "fo"}[v],
            "alignment_shape_weight": lambda v: f"s{int(v * 100):02d}",
        }
        for batch in campaign.BATCHES.values():
            for name, settings in campaign.runs(batch):
                expected = (
                    batch.prefix + "_" + "".join(suffixes[key](settings[key]) for key in batch.axes)
                )
                self.assertEqual(name, expected, "every name must encode every varying setting")
                names.append(name)
        self.assertEqual(len(names), len(set(names)), "a repeated name overwrites a spec")

    def test_no_run_sets_a_setting_the_job_owns(self):
        # output_root and n_jobs come from the PBS allocation; a delta would undo them.
        for batch in campaign.BATCHES.values():
            for job, settings in campaign.runs(batch):
                for reserved in ("output_root", "n_jobs"):
                    self.assertNotIn(reserved, settings, job)

    def test_every_run_asks_for_more_memory_than_its_cell_was_measured_to_use(self):
        # A flat 32 GB killed all 16 minimise+c30 runs of the first attempt during
        # threshold calibration, before any checkpoint existed. These are the high-water
        # marks PBS reported for the cells that survived; the c30 row is the linear
        # extrapolation the kills are consistent with.
        # True peaks from resources_used.mem of this campaign's finished jobs. The
        # earlier version of this test used mid-run samples, which understated every
        # one of them and let a 32 GB request through for cells that then died.
        measured = {
            (False, 10): 22.7,
            (False, 20): 28.0,
            (False, 30): 27.8,
            (True, 10): 22.1,
            (True, 20): 31.2,
        }
        for (minimize, conformers), peak in measured.items():
            asked = campaign.MEMORY_GB
            # Twice the peak, not 1.2x: siblings of one configuration differ by ~5 GB.
            self.assertGreaterEqual(asked, peak * 2.0, f"minimize={minimize} c{conformers}")
        self.assertEqual(campaign.MEMORY_GB, 64)
        # The ad-hoc path must not be the one that still asks for the fatal 32 GB.
        self.assertGreaterEqual(Job.mem, campaign.MEMORY_GB)

    def test_the_four_repeated_configurations_are_named(self):
        # The colour batch is centred on a cell the parent batch already holds. They are
        # kept (fixed seed: a free reproducibility check) but must never be read as four
        # independent measurements.
        repeats = campaign.duplicates()
        self.assertEqual(len(repeats), 4)
        for repeat, original in repeats:
            self.assertTrue(repeat.startswith("c_"))
            self.assertTrue(original.startswith("p_"))


if __name__ == "__main__":
    unittest.main()
