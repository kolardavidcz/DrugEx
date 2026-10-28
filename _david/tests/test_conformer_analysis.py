"""Check the analysis unit, paired comparisons and disjoint stratum boundaries."""

import unittest

import numpy as np
import pandas as pd

from _david.code.conformer.analyze import (
    MEASURES,
    aggregate_parents,
    prepare_analysis,
    summarize_strata,
)


class TestConformerAnalysis(unittest.TestCase):
    def test_seed_averages_precede_pairing_and_strata_include_boundaries(self):
        bounds = [1, 2, 3, 4, 5, 8, 9]
        panel = pd.DataFrame({"molecule_id": range(7), "isomer_upper_bound": bounds})
        rows = []
        for molecule_id in range(7):
            for cap in [1, 30]:
                for seed, score in enumerate([0.2, 0.4, 0.6]):
                    row = dict.fromkeys(MEASURES, 1.0)
                    row.update(
                        molecule_id=molecule_id,
                        method=1,
                        conformers=1,
                        isomer_cap=cap,
                        reference_cap=0,
                        actual_references=3,
                        available=True,
                        seed=seed,
                        failure=False,
                        score=score + (0.1 if cap == 30 and molecule_id else 0),
                    )
                    if molecule_id == 6 and cap == 30 and seed == 0:
                        row.update(score=0, failure=True)
                    rows.append(row)
        parents, selected = prepare_analysis(
            aggregate_parents(pd.DataFrame(rows)), panel
        )
        self.assertEqual(
            selected.stratum.tolist(), ["1", "2", "3-4", "3-4", "5-8", "5-8", ">=9"]
        )
        self.assertEqual(len(parents), 14)
        np.testing.assert_allclose(parents[parents.isomer_cap.eq(1)].score, 0.4)
        ordinary = parents[
            parents.molecule_id.between(1, 5) & parents.isomer_cap.eq(30)
        ]
        np.testing.assert_allclose(ordinary.isomer_gain, 0.1)
        failed = parents[parents.molecule_id.eq(6) & parents.isomer_cap.eq(30)].iloc[0]
        self.assertAlmostEqual(failed.failure_fraction, 1 / 3)
        self.assertTrue(np.isnan(failed.successful_pair_gain))
        summary = summarize_strata(parents)
        self.assertEqual(summary[summary.stratum.eq("3-4")].molecules.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
