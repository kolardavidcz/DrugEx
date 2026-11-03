"""Tests for study expansion, execution, provenance, and derived tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from _david.code.conformer.core.contracts import (
    ArmSpec,
    MoleculeSpec,
    SeedPlan,
    StudySpec,
    default_candidate_spec,
    default_reference_spec,
)
from _david.code.conformer.studies import clustering, optimization_methods
from _david.code.conformer.studies.base import (
    CellRequest,
    load_study_frame,
    run_study,
    study_requests,
)


@dataclass(frozen=True)
class EchoExecutor:
    """Picklable stand-in for the real measurement executor."""

    multiplier: int = 1
    fail_on: str = ""

    def __call__(self, request: CellRequest) -> dict[str, object]:
        if self.fail_on and request.molecule.name == self.fail_on:
            raise RuntimeError(f"boom: {request.label}")
        return {
            "molecule": request.molecule.name,
            "arm": request.arm.name,
            "repeat": request.repeat,
            "value": self.multiplier * len(request.arm.name),
        }


def small_study(output_dir: Path, *, arms: int = 2, molecules: int = 2) -> StudySpec:
    molecules_spec = tuple(
        MoleculeSpec(name=f"m{index}", smiles="CCO")
        for index in range(molecules)
    )
    arms_spec = tuple(
        ArmSpec(
            name=f"a{index}",
            reference=default_reference_spec(),
            candidate=default_candidate_spec(),
        )
        for index in range(arms)
    )
    return StudySpec(
        name="tiny",
        molecules=molecules_spec,
        arms=arms_spec,
        output_dir=output_dir,
        seeds=SeedPlan(repeats=(0, 1)),
    )


class TestStudyExpansion(unittest.TestCase):
    def test_cartesian_product_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = small_study(Path(tmp))
            requests = study_requests(study)
        self.assertEqual(len(requests), 2 * 2 * 2)
        self.assertEqual(requests[0].label, "m0::a0::r0")
        self.assertEqual(requests[-1].label, "m1::a1::r1")
        self.assertEqual({request.repeat for request in requests}, {0, 1})


class TestRunStudy(unittest.TestCase):
    def test_serial_run_writes_results_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = small_study(Path(tmp))
            result = run_study(study, EchoExecutor())
            self.assertTrue(result.ok)
            self.assertEqual(len(result.frame), 8)
            self.assertIn("results.csv", result.artifacts)
            self.assertTrue((Path(tmp) / "manifest.json").is_file())
            reloaded = load_study_frame(Path(tmp) / "results.csv")
        self.assertEqual(len(reloaded), 8)
        self.assertEqual(result.manifest["study"], "tiny")
        self.assertEqual(result.manifest["completed_cells"], 8)
        self.assertEqual(result.manifest["failed_cells"], 0)
        self.assertIn("versions", result.manifest)

    def test_failures_are_recorded_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = small_study(Path(tmp))
            result = run_study(study, EchoExecutor(fail_on="m1"))
        self.assertEqual(len(result.errors), 4)
        self.assertIn("errors.csv", result.artifacts)
        self.assertEqual({error["molecule"] for error in result.errors}, {"m1"})
        self.assertEqual(result.errors[0]["error_type"], "RuntimeError")

    def test_parallel_and_serial_agree(self) -> None:
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as pool_dir:
            serial = run_study(small_study(Path(serial_dir)), EchoExecutor(), workers=1)
            pooled = run_study(small_study(Path(pool_dir)), EchoExecutor(), workers=2)
        serial_rows = serial.frame.sort_values("cell").reset_index(drop=True)
        pooled_rows = pooled.frame.sort_values("cell").reset_index(drop=True)
        pd.testing.assert_frame_equal(serial_rows, pooled_rows)
        self.assertEqual(pooled.manifest["workers"], 2)

    def test_zero_arm_matrix_is_rejected_by_run_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            study = StudySpec(
                name="tiny",
                molecules=(MoleculeSpec(name="m", smiles="CCO"),),
                arms=(ArmSpec(name="a", reference=default_reference_spec(), candidate=default_candidate_spec()),),
                output_dir=Path(tmp),
            )
            empty = StudySpec(
                name="tiny",
                molecules=study.molecules,
                arms=study.arms,
                output_dir=Path(tmp),
                seeds=SeedPlan(repeats=(0,)),
            )
            from _david.code.conformer.studies.base import study_requests

            self.assertEqual(len(study_requests(empty)), 1)


class TestStudyArms(unittest.TestCase):
    def test_symmetric_pairs_cover_every_method(self) -> None:
        pairs = optimization_methods.symmetric_pairs()
        self.assertEqual(len(pairs), len(optimization_methods.METHODS))
        self.assertEqual({pair.reference for pair in pairs}, set(optimization_methods.METHODS))

    def test_deployable_pairs_share_one_reference(self) -> None:
        pairs = optimization_methods.deployable_pairs()
        self.assertEqual({pair.reference for pair in pairs}, {"mmff94s_water"})
        self.assertEqual(len(pairs), len(optimization_methods.METHODS))

    def test_unknown_method_is_rejected(self) -> None:
        from _david.code.conformer.core.errors import StudyError

        with self.assertRaises(StudyError):
            optimization_methods.deployable_pairs("nope")

    def test_arm_names_are_unique_and_descriptive(self) -> None:
        arms = optimization_methods.build_arms(
            pairs=[optimization_methods.MethodPair("mmff94s_water", "etkdg")],
            selections=["diverse", "rmsd_0.35"],
            caps=[1, 4],
            candidate_attempts=[10],
        )
        self.assertEqual(len(arms), 4)
        self.assertEqual(len({spec.name for spec in arms}), 4)
        self.assertIn("ref-mmff94s_water__cand-etkdg", arms[0].name)
        self.assertEqual(arms[0].score.scale.value, "normalized")

    def test_clustering_study_varies_selection_not_physics(self) -> None:
        study = clustering.build_study(
            [MoleculeSpec(name="m", smiles="CCO")],
            "/tmp/ignored",
            selections=("diverse", "rmsd_0.35"),
            caps=(4,),
            candidate_attempts=(10,),
        )
        names = {spec.name for spec in study.arms}
        self.assertEqual(len(names), 2)
        self.assertTrue(any("rmsd_0.35" in name for name in names))


class TestDerivedTables(unittest.TestCase):
    def test_pareto_front_keeps_only_non_dominated_arms(self) -> None:
        frame = pd.DataFrame(
            {
                "arm": ["cheap-bad", "cheap-good", "expensive-best", "expensive-bad"],
                "combo": [0.5, 0.7, 0.9, 0.6],
                "total_seconds": [1.0, 2.0, 10.0, 9.0],
            }
        )
        front = optimization_methods.pareto_front(frame)
        # "expensive-bad" is dominated by "cheap-good"; the cheapest arm stays
        # on the front even when a better-scoring arm exists, because it is not
        # dominated (it costs less).
        self.assertEqual(
            set(front["arm"]), {"cheap-bad", "cheap-good", "expensive-best"}
        )
        self.assertEqual(list(front["cost"]), [1.0, 2.0, 10.0])

    def test_pareto_front_of_empty_frame_is_empty(self) -> None:
        self.assertTrue(optimization_methods.pareto_front(pd.DataFrame()).empty)

    def test_stratified_summary_uses_canonical_bins(self) -> None:
        frame = pd.DataFrame(
            {
                "molecule": ["a", "b", "c"],
                "isomer_upper_bound": [1, 6, 128],
                "reference_cap": [4, 4, 4],
                "combo": [0.9, 0.7, 0.5],
            }
        )
        summary = optimization_methods.stratified_summary(frame)
        labels = list(summary["isomer_bin"].astype(str))
        self.assertEqual(labels, ["1–2", "5–8", "33+"])

    def test_cluster_counts_summarises_per_selection(self) -> None:
        from _david.code.conformer.analysis.clustering import cluster_counts

        frame = pd.DataFrame(
            {
                "ref_sel": ["rmsd_0.35", "rmsd_0.35", "rmsd_1.25"],
                "k_label": ["all", "all", "all"],
                "n_clusters": [10, 20, 2],
            }
        )
        summary = cluster_counts(frame).set_index("ref_sel")
        self.assertEqual(int(summary.loc["rmsd_0.35", "median"]), 15)
        self.assertEqual(int(summary.loc["rmsd_1.25", "maximum"]), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
