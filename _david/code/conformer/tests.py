"""Tests for the self-similarity study: ``python -m unittest _david.code.conformer.tests``.

Two of these exist to stop specific, previously-made mistakes rather than to raise
coverage: that the study's force field is still the pipeline's force field, and that the
reference and candidate conformers really are independent draws.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.config import Config
from _david.code.conformer import figures, study
from _david.code.conformer.config import METHODS, StudyConfig
from _david.code.scorers.conformers import single_point_energy

ETHANOL = "CCO"
#: Small, rigid and fully specified: one stereoisomer, embeds in milliseconds.
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def small(**changes) -> StudyConfig:
    cheap = {
        "smiles": IBUPROFEN,
        "name": "test",
        "methods": ("ETKDG", "MMFF94s_e78.4"),
        "reference_caps": (1, 2),
        "candidate_counts": (3,),
        "repeats": 2,
        "reference_attempts": 6,
        "max_isomers": 1,
        "n_jobs": 1,
    }
    return replace(StudyConfig(), **{**cheap, **changes})


class ForceFieldMatchesThePipeline(unittest.TestCase):
    """The study exists to choose pipeline settings, so it must measure the pipeline."""

    def test_mmff94s_e784_energy_equals_the_pipelines_own(self):
        cfg = small()
        mol = Chem.AddHs(Chem.MolFromSmiles(IBUPROFEN))
        from rdkit.Chem import AllChem

        params = AllChem.ETKDGv3()
        params.randomSeed = 1234
        AllChem.EmbedMolecule(mol, params=params)

        mine = study._energy(mol, METHODS["MMFF94s_e78.4"], cfg)
        # The pipeline's own call, with the settings the pipeline ships.
        theirs = single_point_energy(mol, Config(dielectric=78.4, dielectric_model=1))
        self.assertTrue(math.isfinite(mine), "the study could not score a drug-like molecule")
        self.assertAlmostEqual(
            mine,
            theirs,
            places=6,
            msg="the study's MMFF94s eps=78.4 has drifted from _david/code/scorers/conformers.py",
        )

    def test_mmff94_and_mmff94s_agree_only_when_there_is_no_nitrogen(self):
        """Why the grid's MMFF94 and MMFF94s rows are identical for EGCG, and not a bug.

        MMFF94s differs from MMFF94 in one respect: it planarises delocalised trigonal
        nitrogen. EGCG is C22H18O11 and has none, so the two are the same force field for
        it. The second half of this test is what makes the first half meaningful -- an
        implementation that ignored ``mmffVariant`` entirely would pass the first.
        """
        cfg = small()
        from rdkit.Chem import AllChem

        params = AllChem.ETKDGv3()
        params.randomSeed = 7

        from _david.code.config import EGCG

        nitrogen_free = Chem.AddHs(Chem.MolFromSmiles(EGCG))
        AllChem.EmbedMolecule(nitrogen_free, params=params)
        self.assertEqual(
            sum(1 for a in nitrogen_free.GetAtoms() if a.GetSymbol() == "N"), 0, "EGCG has no N"
        )
        self.assertEqual(
            study._energy(nitrogen_free, METHODS["MMFF94_e1"], cfg),
            study._energy(nitrogen_free, METHODS["MMFF94s_e1"], cfg),
        )

        amide = Chem.AddHs(Chem.MolFromSmiles("c1ccc(cc1)N(C)C(=O)C"))
        AllChem.EmbedMolecule(amide, params=params)
        self.assertNotEqual(
            study._energy(amide, METHODS["MMFF94_e1"], cfg),
            study._energy(amide, METHODS["MMFF94s_e1"], cfg),
            "the variant is being ignored: MMFF94s must differ on a delocalised amide N",
        )

    def test_an_unparameterisable_molecule_scores_infinity_not_zero(self):
        """0.0 would sort first and seed the bank with a failure -- the original bug."""
        cfg = small()
        mol = Chem.AddHs(Chem.MolFromSmiles("[Xe]"))
        value = study._energy(mol, METHODS["MMFF94s_e78.4"], cfg)
        self.assertEqual(value, math.inf)
        self.assertFalse(value < 0.0)

    def test_etkdg_has_no_force_field_and_never_minimises(self):
        cfg = small()
        mols = study._embed(IBUPROFEN, 2, 104729, cfg)
        before = [Chem.Mol(m).GetConformer().GetPositions().copy() for m in mols]
        after = study._relax(mols, METHODS["ETKDG"], cfg)
        for original, relaxed in zip(before, after):
            np.testing.assert_allclose(original, relaxed.GetConformer().GetPositions())


class IndependentStreams(unittest.TestCase):
    def test_candidate_and_reference_conformers_are_different_draws(self):
        """Adjacent seeds would score an ensemble against conformers it already holds."""
        cfg = small()
        seed = study.SEEDS[0]
        references = study._embed(cfg.smiles, 6, seed, cfg)
        candidates = study._embed(cfg.smiles, 6, seed + study.CANDIDATE_OFFSET, cfg)
        self.assertTrue(references and candidates)
        reference_coords = [m.GetConformer().GetPositions() for m in references]
        for candidate in candidates:
            coords = candidate.GetConformer().GetPositions()
            for other in reference_coords:
                if coords.shape == other.shape:
                    self.assertFalse(
                        np.allclose(coords, other),
                        "a candidate conformer is bit-identical to a reference conformer: "
                        "the two embedding streams are not independent",
                    )

    def test_the_seeds_are_pairwise_non_multiples(self):
        """RDKit derives per-conformer seeds multiplicatively, so multiples collide."""
        for i, a in enumerate(study.SEEDS):
            for b in study.SEEDS[i + 1 :]:
                self.assertNotEqual(a, b)
                self.assertTrue(max(a, b) % min(a, b), f"{a} and {b} are multiples")


class BankSelection(unittest.TestCase):
    def test_the_bank_never_exceeds_its_cap_and_is_energy_ordered(self):
        cfg = small()
        method = METHODS["MMFF94s_e78.4"]
        pool = study._relax(study._embed(cfg.smiles, 8, 104729, cfg), method, cfg)
        bank = study.select_bank(pool, 3, method, cfg)
        self.assertLessEqual(len(bank), 3)
        energies = [m.GetDoubleProp("Energy") for m in bank]
        self.assertEqual(energies, sorted(energies), "the bank must be ordered by energy")

    def test_an_empty_pool_gives_an_empty_bank_rather_than_raising(self):
        self.assertEqual(study.select_bank([], 4, METHODS["ETKDG"], small()), [])


class Scoring(unittest.TestCase):
    def test_a_molecule_scored_against_itself_is_the_maximum(self):
        cfg = small()
        mols = study._embed(IBUPROFEN, 1, 104729, cfg)
        shape, colour, failed = study.score(mols, mols, cfg)
        self.assertAlmostEqual(shape + colour, 2.0, places=3, msg="self-overlay must be 2.0")
        self.assertFalse(failed)

    def test_missing_input_is_flagged_as_a_failure_not_a_real_zero(self):
        """An exact zero is unreachable for two conformers of one molecule."""
        cfg = small()
        self.assertEqual(study.score([], [], cfg), (0.0, 0.0, True))


class Parameterisation(unittest.TestCase):
    def test_a_force_field_that_covers_nothing_is_refused_not_relabelled(self):
        """Otherwise the row is the ETKDG control wearing a minimised method's name."""
        # Boron has no MMFF parameters, so MMFF94s covers none of this molecule.
        cfg = small(smiles="B(O)(O)c1ccccc1", methods=("MMFF94s_e78.4",), reference_attempts=4)
        with self.assertRaises(SystemExit) as caught:
            study._one_stream((cfg, "MMFF94s_e78.4", study.SEEDS[0]))
        self.assertIn("cannot parameterise", str(caught.exception))

    def test_a_molecule_the_force_field_covers_runs_normally(self):
        cfg = small(reference_attempts=4, candidate_counts=(2,), reference_caps=(1,))
        rows, reference = study._one_stream((cfg, "MMFF94s_e78.4", study.SEEDS[0]))
        self.assertTrue(rows)
        self.assertEqual(reference["parameterised"], reference["embedded"])


class Grid(unittest.TestCase):
    def test_the_grid_runs_and_every_cell_is_present_once(self):
        cfg = small()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            frame = study.run(cfg, out)
            self.assertEqual(len(frame), cfg.cells)
            self.assertEqual(
                len(
                    frame.drop_duplicates(
                        subset=["method", "reference_cap", "candidate_count", "seed"]
                    )
                ),
                cfg.cells,
            )
            self.assertTrue((frame["self_recovery"] >= 0.0).all())
            self.assertTrue((frame["self_recovery"] <= 1.0 + 1e-6).all())
            # combo is the raw [0, 2] scale and self_recovery is half of it, always.
            np.testing.assert_allclose(frame["self_recovery"], frame["combo"] / 2.0, atol=1e-6)
            self.assertTrue((frame["cell_cpu_seconds"] > 0).all(), "a cell with no cost is a bug")
            self.assertTrue((frame["bank_size"] <= frame["reference_cap"]).all())
            self.assertFalse(frame["score_failed"].any(), "no cell should fail to score")
            self.assertTrue((out / "measurements.tsv").is_file())
            self.assertTrue((out / "references.tsv").is_file())

    def test_figures_draw_from_a_real_grid(self):
        cfg = small()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            frame = study.run(cfg, out)
            written = figures.render_all(frame, cfg, out)
            self.assertEqual(len(written), 4, "every sheet must draw")
            for path in written:
                self.assertGreater(path.stat().st_size, 0)


class Ceiling(unittest.TestCase):
    def test_the_ceiling_tolerance_is_about_pose_not_float_equality(self):
        """ROCS aligns before scoring, so 2.0 means the same pose, not the same numbers.

        A float-exact test reported 2.7% of this study's cells against the 24.7% that are
        actually indistinguishable from a perfect overlay -- a tenfold understatement of
        the caveat the whole study turns on.
        """
        frame = pd.DataFrame({"combo": [2.0, 2.0 - 1e-9, 2.0 - 5e-7, 2.0 - 1e-4, 1.5]})
        self.assertEqual(list(figures.at_ceiling(frame)), [True, True, True, False, False])


class ConfigValidation(unittest.TestCase):
    def test_an_unknown_method_is_rejected_before_anything_runs(self):
        with self.assertRaises(ValueError):
            small(methods=("NOT_A_METHOD",))

    def test_a_cap_larger_than_the_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            small(reference_caps=(99,), reference_attempts=8)

    def test_cells_counts_the_grid(self):
        cfg = small(methods=("ETKDG",), reference_caps=(1, 2), candidate_counts=(3, 4), repeats=5)
        self.assertEqual(cfg.cells, 1 * 2 * 2 * 5)

    def test_overrides_are_typed_from_the_dataclass(self):
        cfg = study.config_from_args(
            ["repeats=3", "reference_caps=1,2,8", "methods=ETKDG", "alignment_shape_weight=0.25"]
        )
        self.assertEqual(cfg.repeats, 3)
        self.assertEqual(cfg.reference_caps, (1, 2, 8))
        self.assertEqual(cfg.methods, ("ETKDG",))
        self.assertAlmostEqual(cfg.alignment_shape_weight, 0.25)


if __name__ == "__main__":
    unittest.main()
