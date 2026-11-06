"""Empty-color semantics, RL/study parity, and independent search dimensions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import AllChem

from drugex.training.scorers.rocs_rdkit import _score_single_reference
from _david.code.chemistry import CandidateConformerGenerator
from _david.code.conformer.self_recovery import reference_group, reference_job
from _david.code.training.environment import create_rocs_scorer
from _david.code.rocs import has_color, score_pair


def embedded(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=43) != 0:
        raise ValueError("Fixture embedding failed")
    return mol


class TestThesisROCS(unittest.TestCase):
    def test_native_empty_color_behavior_is_preserved(self):
        ethane, butane = embedded("CC"), embedded("CCCC")
        ethanol, benzene = embedded("CCO"), embedded("c1ccccc1")
        for mol in [ethane, butane]:
            self.assertFalse(has_color(mol))
            self.assertAlmostEqual(score_pair(mol, mol)[0], 0.5, places=6)
            self.assertAlmostEqual(score_pair(mol, mol)[2], 0, places=6)
            self.assertAlmostEqual(
                _score_single_reference(mol, mol, "TanimotoCombo", True) / 2,
                0.5,
                places=6,
            )
        self.assertEqual(score_pair(ethane, ethanol)[2], 0)
        self.assertTrue(has_color(ethanol))
        self.assertTrue(has_color(benzene))
        self.assertEqual(score_pair(ethanol, benzene)[2], 0)
        self.assertAlmostEqual(score_pair(ethanol, ethanol)[0], 1, places=6)

    def test_parallel_and_sequential_rl_factory_match_native_pair_maximum(self):
        references = [embedded("CC"), embedded("CCO")]
        for jobs in [1, 2]:
            scorer = create_rocs_scorer(
                references,
                max_conformers=2,
                max_isomers=1,
                conformer_threads=1,
                n_jobs=jobs,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = scorer.conformer_generator.genConformers(["CC"], directory)
                generated = list(Chem.SDMolSupplier(path, removeHs=False))
            expected = max(
                score_pair(c, r)[0] * 2 for c in generated for r in references
            )
            actual = scorer.getScores(["CC", "CC"])
            self.assertAlmostEqual(actual[0, 0], expected, places=10)
            self.assertAlmostEqual(actual[1, 0], expected, places=10)

    def test_colorless_input_is_excluded_before_any_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "checkpoints").mkdir()
            with patch(
                "_david.code.conformer.self_recovery.score_pair",
                side_effect=AssertionError("Eligibility must precede ROCS"),
            ):
                result = reference_job(
                    dict(molecule_id=0, smiles="CC", stereo_upper_bound=1), directory
                )
            self.assertEqual(result["exclusion_reason"], "no_native_color_features")
            self.assertEqual(result["group"], "")
            self.assertEqual(result["error"], "")
            self.assertEqual(result["profiles"], [])
            self.assertFalse(result["embedding"][0]["native_color_present"])

    def test_stereo_search_is_shared_while_geometry_seeds_are_independent(self):
        mol = Chem.MolFromSmiles("Br" + "[CH](Cl)" * 5 + "F")

        def identities(cap, geometry_seed):
            generator = CandidateConformerGenerator(
                max_isomers=cap,
                stereo_seed=42,
                random_seed=geometry_seed,
            )
            return {
                m.GetProp("EnumeratedIsomerSMILES") for m in generator._get_isomers(mol)
            }

        reference = identities(4, 43)
        self.assertEqual(reference, identities(4, 42))
        self.assertTrue(identities(1, 42) <= reference)
        self.assertTrue(reference <= identities(8, 42))

    def test_merged_structure_group_boundaries_are_disjoint(self):
        for counts, expected in [
            ([0], ""),
            ([3, 4], "1-4"),
            ([4, 5], "5-8"),
            ([8, 9], "9-12"),
            ([12], "9-12"),
            ([13], ""),
        ]:
            self.assertEqual(reference_group(counts), expected)


if __name__ == "__main__":
    unittest.main()
