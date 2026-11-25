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
        for original, relaxed in zip(before, after, strict=True):
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


class CrossStudyTests(unittest.TestCase):
    def test_select_bank_ordered_prefixes_match_select_bank_set(self):
        cfg = small(smiles="CCCCCC(=O)NCc1ccccc1", reference_attempts=12)
        method = METHODS["MMFF94s_e78.4"]
        pool = study._relax(study._embed(cfg.smiles, 12, study.SEEDS[0], cfg), method, cfg)
        ordered = study.select_bank_ordered(pool, 4, method, cfg)
        for k in (1, 2, 4):
            direct = study.select_bank(pool, k, method, cfg)
            e_ord = sorted(round(m.GetDoubleProp("Energy"), 6) for m in ordered[:k])
            e_dir = sorted(round(m.GetDoubleProp("Energy"), 6) for m in direct)
            self.assertEqual(e_ord, e_dir)

    def test_classify_flexibility_bounds(self):
        from _david.code.conformer.cross_study import CrossStudyConfig, classify_flexibility

        cfg = CrossStudyConfig()
        self.assertEqual(classify_flexibility(0, cfg), "rigid")
        self.assertEqual(classify_flexibility(2, cfg), "rigid")
        self.assertEqual(classify_flexibility(3, cfg), "intermediate")
        self.assertEqual(classify_flexibility(6, cfg), "intermediate")
        self.assertEqual(classify_flexibility(7, cfg), "flexible")
        self.assertEqual(classify_flexibility(15, cfg), "flexible")

    def test_incremental_grid_matches_brute_force_pair_components_and_renders(self):
        from _david.code.conformer import cross_figures, cross_study
        from _david.code.scorers.rocs import pair_components

        sample_smiles = [
            "c1ccccc1",  # rigid (0 RotB)
            "CC(=O)Nc1ccc(O)cc1",  # rigid (1 RotB)
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # intermediate (4 RotB, chiral)
            "CC(O)CC(N)CC(F)CC",  # multi-isomer chiral, triggers RMSD pruning (M(C) < C)
            "COc1ccccc1OCC(=O)NCCO",  # flexible (7 RotB)
            "Fc1ccc(-c2nc(C3CC3)no2)cc1",  # rigid/inter
            "CCN(CC)CCOC(=O)c1ccccc1",  # flexible (7 RotB)
        ]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            smi_file = tmp / "cohort.smi"
            smi_file.write_text("\n".join(sample_smiles) + "\n")
            out = tmp / "out"
            cfg = cross_study.CrossStudyConfig(
                n_molecules=7,
                n_references=3,
                methods=("ETKDG", "MMFF94s_e78.4"),
                reference_caps=(1, 2),
                candidate_counts=(4, 8),
                reference_attempts=6,
                max_isomers=2,
                representative_k=2,
                representative_c=8,
                n_jobs=1,
            )
            molecules_df, grid_df, timing_df, self_df = cross_study.run(
                cfg, out, smiles_file=smi_file
            )

            self.assertEqual(len(molecules_df), 7)
            self.assertEqual(len(self_df), 3 * 2 * 2 * 2)  # 3 refs x 2 methods x 2 K x 2 C
            self.assertEqual(len(timing_df), 2 * 2 * 2)
            self.assertTrue((timing_df["rocs_cpu_s"] > 0).all())
            self.assertTrue((timing_df["cand_embed_cpu_s"] > 0).all())

            # Check cross band partition invariant on global_cross: <p50 + [p50,p85) + >=p85 == all_cross
            g_cell = grid_df[
                (grid_df["method"] == "MMFF94s_e78.4")
                & (grid_df["reference_cap"] == 2)
                & (grid_df["candidate_count"] == 8)
                & (grid_df["scope"] == "global_cross")
            ]
            n_by_band = dict(zip(g_cell["band"], g_cell["n"], strict=True))
            self.assertEqual(
                n_by_band["lt_p50_noise"] + n_by_band["p50_to_p85"] + n_by_band["ge_p85"],
                n_by_band["all_cross"],
            )
            self.assertEqual(n_by_band["all_cross"], 3 * 7 - 3)

            # Verify bit-exact match against independent oracle _embed_tagged + select_bank + pair_components
            with np.load(out / "pair_scores.npz") as data:
                shapes = data["shape"].copy()
                colours = data["colour"].copy()
                ref_indices = [int(x) for x in data["ref_mol_indices"]]
            study_cfg = cfg.as_study_config("MMFF94s_e78.4")
            method = METHODS["MMFF94s_e78.4"]
            cand_seed = cfg.reference_seed + study.CANDIDATE_OFFSET

            for ri, ref_mol_idx in enumerate(ref_indices):
                ref_smi = molecules_df.loc[ref_mol_idx, "canonical_smiles"]
                ref_pool = study._relax(
                    cross_study._embed_tagged(ref_smi, cfg.reference_attempts, cfg.reference_seed, cfg),
                    method,
                    study_cfg,
                )
                for ki, k_cap in enumerate(cfg.reference_caps):
                    bank = study.select_bank(ref_pool, k_cap, method, study_cfg)
                    for qi, cand_smi in enumerate(molecules_df["canonical_smiles"]):
                        for ci, c_cnt in enumerate(cfg.candidate_counts):
                            cands = study._relax(
                                cross_study._embed_tagged(cand_smi, c_cnt, cand_seed, cfg),
                                method,
                                study_cfg,
                            )
                            best_sc = (0.0, 0.0)
                            for cm in cands:
                                for rm in bank:
                                    sc = pair_components(
                                        cm, rm, opt_param=0.5, use_colors=True, color_ff="default"
                                    )
                                    if sum(sc) > sum(best_sc):
                                        best_sc = sc
                            got_combo = float(shapes[1, ri, qi, ki, ci] + colours[1, ri, qi, ki, ci])
                            self.assertAlmostEqual(
                                got_combo,
                                sum(best_sc),
                                places=5,
                                msg=f"mismatch at ref={ri}, q={qi}, K={k_cap}, C={c_cnt}",
                            )

            written = cross_figures.render_all(out, cfg=cfg)
            self.assertEqual(len(written), 5)
            for p in written:
                self.assertGreater(p.stat().st_size, 0)

    def test_non_default_color_ff_matches_pair_components(self):
        from _david.code.conformer import cross_study
        from _david.code.scorers.rocs import pair_components

        smi_ref = "c1ccccc1OCC(=O)O"
        smi_cand = "c1ccccc1NC(=O)C"
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            smi_file = tmp / "fdef.smi"
            smi_file.write_text(f"{smi_ref}\n{smi_cand}\n")
            cfg = cross_study.CrossStudyConfig(
                n_molecules=2,
                n_references=1,
                methods=("ETKDG",),
                reference_caps=(1,),
                candidate_counts=(2,),
                reference_attempts=2,
                max_isomers=1,
                color_ff="fdef",
                representative_k=1,
                representative_c=2,
                n_jobs=1,
            )
            molecules_df, _, _, _ = cross_study.run(cfg, tmp / "out", smiles_file=smi_file)
            with np.load(tmp / "out" / "pair_scores.npz") as data:
                ref_idx = int(data["ref_mol_indices"][0])
                other_idx = 1 - ref_idx
                got_combo = float(data["shape"][0, 0, other_idx, 0, 0] + data["colour"][0, 0, other_idx, 0, 0])

            study_cfg = cfg.as_study_config("ETKDG")
            ref_smi = molecules_df.loc[ref_idx, "canonical_smiles"]
            cand_smi = molecules_df.loc[other_idx, "canonical_smiles"]
            bank = study.select_bank(
                cross_study._embed_tagged(ref_smi, cfg.reference_attempts, cfg.reference_seed, cfg),
                1,
                METHODS["ETKDG"],
                study_cfg,
            )
            cands = cross_study._embed_tagged(
                cand_smi, 2, cfg.reference_seed + study.CANDIDATE_OFFSET, cfg
            )
            best_sc = (0.0, 0.0)
            for cm in cands:
                for rm in bank:
                    sc = pair_components(cm, rm, opt_param=0.5, use_colors=True, color_ff="fdef")
                    if sum(sc) > sum(best_sc):
                        best_sc = sc
            self.assertAlmostEqual(got_combo, sum(best_sc), places=5)

    def test_cross_study_config_from_args_coerces_types(self):
        from _david.code.conformer import cross_study

        cfg = cross_study.config_from_args(
            [
                "n_molecules=12",
                "n_references=4",
                "methods=ETKDG,MMFF94s_e78.4",
                "reference_caps=1,4",
                "color_ff=fdef",
                "n_jobs=2",
            ]
        )
        self.assertEqual(cfg.n_molecules, 12)
        self.assertEqual(cfg.n_references, 4)
        self.assertEqual(cfg.methods, ("ETKDG", "MMFF94s_e78.4"))
        self.assertEqual(cfg.reference_caps, (1, 4))
        self.assertEqual(cfg.color_ff, "fdef")
        self.assertEqual(cfg.n_jobs, 2)


if __name__ == "__main__":
    unittest.main()
