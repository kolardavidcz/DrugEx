"""Scientific invariants and production-scorer parity for the conformer benchmark."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer, _serialize_worker_molecule

from _david.code.conformer.benchmark import (
    CandidateBuilder,
    EnsembleIO,
    MoleculePanel,
    ReferenceBuilder,
    ReferenceProtocol,
    TrainingKernel,
)
from _david.code.conformer.benchmark_analysis import BenchmarkAnalysis
from _david.code.config import DEFAULT_MOLECULE
from _david.code.environment import PrepareMolecule


class TestConformerBenchmark(unittest.TestCase):
    """Verify measured scores correspond to training and statistics use correct units."""

    def test_reference_baseline_matches_training_preparation(self) -> None:
        """Check the EGCG cutoff and first three geometries against real preparation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = ReferenceProtocol.suite()[0]
            molecule, _ = PrepareMolecule().load_molecule(DEFAULT_MOLECULE)
            self.assertEqual(
                protocol.rmsd_cutoff,
                PrepareMolecule.determine_ensemble_parameters(molecule)[0],
            )
            measured = ReferenceBuilder.run(
                {
                    "protocol": asdict(protocol),
                    "budget": 32,
                    "seed": 42,
                    "replicate": 0,
                    "path": str(root / "measured.sdf"),
                }
            )
            self.assertEqual(measured["status"], "ok")
            PrepareMolecule().generate_reference_conformers(
                DEFAULT_MOLECULE,
                root / "training.sdf",
                num_conformers=32,
                random_seed=42,
            )
            expected = EnsembleIO.read(root / "training.sdf")
            actual = EnsembleIO.read(root / "measured.sdf")[:3]
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self.assertLess(
                    AllChem.GetBestRMS(Chem.RemoveHs(first), Chem.RemoveHs(second)),
                    0.001,
                )

    def test_training_kernel_matches_full_scorer(self) -> None:
        """Match all three metrics against full production SDF preparation and scoring."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
            AllChem.EmbedMolecule(reference, randomSeed=42)
            reference_file = root / "reference.sdf"
            EnsembleIO.write(reference_file, reference, [0])
            smile = "CC(C)NCC(O)COc1cccc2ccccc12"
            candidate = CandidateBuilder.run(
                {
                    "path": str(root / "candidate.sdf"),
                    "count": 2,
                    "seed": 0xC0FFEE,
                    "timeout": 20,
                    "smiles": smile,
                }
            )
            self.assertEqual(candidate["status"], "ok")
            queries = EnsembleIO.read(root / "candidate.sdf")
            for metric in ("tanimoto", "tversky_ref", "tversky_subshape"):
                scorer = RDKitROCSScorer(
                    RDKitConformerGenerator(
                        max_conformers=2,
                        max_isomers=4,
                        max_heavy_atoms=45,
                        max_rotatable_bonds=15,
                        num_threads=1,
                        timeout=20,
                    ),
                    str(reference_file),
                    n_jobs=1,
                    show_progress=False,
                    score_variant=metric,
                )
                expected = float(scorer.getScores([Chem.MolFromSmiles(smile)])[0, 0])
                measured = TrainingKernel.score(
                    queries, EnsembleIO.read(reference_file), metric
                )
                self.assertAlmostEqual(measured, expected, places=9)
                self.assertGreater(measured, 0)
                self.assertLessEqual(measured, 2)
                scorer.n_jobs = 2
                with patch.dict("os.environ", {"DRUGEX_MP_CONTEXT": "spawn"}):
                    parallel = float(scorer.getScores([Chem.MolFromSmiles(smile)])[0, 0])
                self.assertAlmostEqual(parallel, expected, places=12)

    def test_parallel_payload_preserves_coordinates(self) -> None:
        """Keep native double coordinates exact after the multiprocessing pickle step."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        payload = pickle.loads(pickle.dumps(_serialize_worker_molecule(mol)))
        restored = Chem.Mol(payload)
        np.testing.assert_array_equal(
            mol.GetConformer().GetPositions(), restored.GetConformer().GetPositions()
        )

    def test_filtered_candidate_is_recorded_as_failure(self) -> None:
        """Keep a valid but out-of-domain molecule visible rather than dropping it."""
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateBuilder.run(
                {
                    "path": str(Path(directory) / "filtered.sdf"),
                    "count": 2,
                    "seed": 42,
                    "timeout": 1,
                    "smiles": "C" * 46,
                }
            )
            self.assertEqual(result["status"], "no_conformers")
            self.assertEqual(result["actual_conformers"], 0)
            self.assertGreater(result["generation_seconds"], 0)

    def test_negative_embedding_id_never_becomes_a_2d_record(self) -> None:
        """Treat RDKit's timeout sentinel as failure instead of writing a 2D depiction."""
        with tempfile.TemporaryDirectory() as directory:
            generator = RDKitConformerGenerator(
                max_isomers=1, protonate=False, canonicalize_tautomer=False
            )
            with patch.object(AllChem, "EmbedMultipleConfs", return_value=[-1]):
                result = Path(generator.genConformers(["CCO"], directory))
            self.assertEqual(result.stat().st_size, 0)
            self.assertEqual(generator.rejected_embedding_results, 1)

    def test_incremental_prefixes_match_independent_ensembles(self) -> None:
        """Reuse exact pair work while preserving every reference-cap score and timer."""
        reference = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
        AllChem.EmbedMultipleConfs(reference, numConfs=4, randomSeed=42)
        references: list[Chem.Mol] = []
        for conf in reference.GetConformers():
            single = Chem.Mol(reference)
            single.RemoveAllConformers()
            single.AddConformer(conf, assignId=True)
            references.append(single)
        queries = references[:2]
        for metric in ("tanimoto", "tversky_ref", "tversky_subshape"):
            measured = TrainingKernel.prefixes(queries, references, metric)
            for count in range(1, 5):
                self.assertAlmostEqual(
                    measured[count][0],
                    TrainingKernel.score(queries, references[:count], metric),
                    places=12,
                )
                self.assertGreater(measured[count][1], measured[count - 1][1])
                self.assertGreaterEqual(measured[count][0], measured[count - 1][0])

    def test_panel_has_disjoint_holdout_and_no_score_dependence(self) -> None:
        """Freeze canonical identities and select using chemistry, without saved scores."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "samples.csv"
            pd.DataFrame({"SMILES": ["C" * n + "O" for n in range(3, 15)]}).to_csv(
                source, index=False
            )
            panel = MoleculePanel.build([source], count=6, seed=42)
            model = panel[panel.source != "control"]
            self.assertEqual(len(model), 6)
            self.assertEqual(model.panel_split.eq("holdout").sum(), 2)
            self.assertEqual(panel.smiles.nunique(), len(panel))
            self.assertFalse(
                set(model[model.panel_split == "screening"].molecule_id)
                & set(model[model.panel_split == "holdout"].molecule_id)
            )

    def test_bootstrap_does_not_count_replicates_as_molecules(self) -> None:
        """Repeating identical seed observations must not shrink molecule uncertainty."""
        original = pd.DataFrame(
            {"molecule_id": ["a", "b", "c"], "loss": [0.01, 0.10, 0.20]}
        )
        repeated = pd.concat([original] * 20, ignore_index=True)
        self.assertEqual(
            BenchmarkAnalysis.confidence_interval(original, "loss"),
            BenchmarkAnalysis.confidence_interval(repeated, "loss"),
        )

    def test_holdout_failure_does_not_reselect_a_setting(self) -> None:
        """Report failed generalization rather than selecting again on the holdout."""
        keys = dict(
            protocol="test",
            candidate_method="training_etkdg",
            matching="tanimoto",
            reference_budget=100,
            k_ref_requested=3,
            k_candidate_requested=5,
        )
        train = pd.DataFrame(
            [
                {
                    **keys,
                    "mean_loss": 0.01,
                    "p95_loss": 0.03,
                    "failure_fraction": 0.0,
                    "median_candidate_seconds": 0.2,
                    "mean_reference_seconds": 1.0,
                }
            ]
        )
        holdout = pd.DataFrame(
            [
                {
                    **keys,
                    "mean_loss": 0.08,
                    "p95_loss": 0.1,
                    "failure_fraction": 0.0,
                    "loss_ci_low": 0.05,
                    "loss_ci_high": 0.1,
                    "rank_spearman": 0.9,
                    "n_molecules": 8,
                }
            ]
        )
        selected = BenchmarkAnalysis.select(train, holdout).iloc[0]
        self.assertEqual(selected.selection_status, "holdout_did_not_confirm")
        self.assertEqual(selected.k_candidate_requested, 5)
        self.assertTrue(np.isclose(selected.holdout_mean_loss, 0.08))


if __name__ == "__main__":
    unittest.main()
