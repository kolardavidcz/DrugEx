"""Contracts tying the diagnostic ROCS helpers to the production scorer.

``score_pair`` / ``align_to_reference`` are used by reference-conformer selection and
by the shape/colour figures, but they are not on the training path. They must still
score with the same pharmacophore typing as :class:`CanonicalROCSScorer`, otherwise
the diagnostics describe a different colour channel than the run optimised.
"""

from __future__ import annotations

import unittest

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from drugex.training.scorers.rocs_rdkit import _score_single_reference

from _david.code.analysis.ranking import lead_rank_score, lead_ranking_weights
from _david.code.config import DEFAULT_MOLECULE
from _david.code.rocs import score_pair
from _david.code.schema import std

#: A flexible, alkyl-rich probe: its fdef and ImplicitMillsDean typings differ more
#: than EGCG's, so a missing annotation cannot pass the agreement test by accident.
PROBE = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _pose(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise AssertionError(f"embedding failed for {smiles}")
    return mol


class TestScorePairMatchesProduction(unittest.TestCase):
    """``score_pair`` must reproduce the production per-reference score exactly."""

    def test_combo_matches_single_reference_production_path(self) -> None:
        reference = _pose(DEFAULT_MOLECULE, 7)
        candidate = _pose(PROBE, 11)

        combo, shape, color = score_pair(candidate, reference)

        production = _score_single_reference(
            Chem.Mol(candidate),
            Chem.Mol(reference),
            std.ROCS_SCORE_TYPE,
            True,
            False,
            std.VARIANT_TANIMOTO,
            "oe_mills_dean",
        )
        self.assertAlmostEqual(combo, production, places=6)
        self.assertAlmostEqual(combo, shape + color, places=9)

    def test_default_typing_is_the_production_oe_mills_dean_scheme(self) -> None:
        reference = _pose(DEFAULT_MOLECULE, 7)
        candidate = _pose(PROBE, 11)

        default = score_pair(candidate, reference)
        explicit = score_pair(candidate, reference, color_ff="oe_mills_dean")
        self.assertEqual(default, explicit)


class TestLeadRankingObjectiveAlignment(unittest.TestCase):
    """Lead ranking must reward the objectives the run actually trained on."""

    def test_qed_weight_follows_the_training_objective(self) -> None:
        _, qed_weight, sa_weight = lead_ranking_weights({"include_qed": False})
        self.assertEqual(qed_weight, 0.0)
        self.assertLess(sa_weight, 0.0)

        _, qed_weight_on, _ = lead_ranking_weights({"include_qed": True})
        self.assertEqual(qed_weight_on, 1.0)

        # A run without a readable manifest keeps the EGCG default (QED excluded).
        self.assertEqual(lead_ranking_weights(None)[1], 0.0)

    def test_high_qed_cannot_outrank_high_rocs_when_qed_is_disabled(self) -> None:
        frame = pd.DataFrame(
            {
                "ROCS": [0.80, 0.40],
                "QED": [0.10, 0.95],
                "Raw_SA": [3.0, 3.0],
            }
        )

        disabled = lead_rank_score(frame, {"include_qed": False})
        self.assertGreater(disabled.iloc[0], disabled.iloc[1])

        enabled = lead_rank_score(frame, {"include_qed": True})
        self.assertGreater(enabled.iloc[1], enabled.iloc[0])

    def test_missing_sa_falls_back_to_the_desirability_channel(self) -> None:
        frame = pd.DataFrame(
            {"ROCS": [0.80, 0.70], "QED": [0.2, 0.2], "SA": [0.3, 0.9]}
        )
        score = lead_rank_score(frame, {"include_qed": False})
        self.assertLess(score.iloc[1], score.iloc[0])


if __name__ == "__main__":
    unittest.main()
