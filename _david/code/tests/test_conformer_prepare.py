"""Tests for molecule preparation, stereoisomer enumeration, and registry tables."""

from __future__ import annotations

import unittest

import pandas as pd

from _david.code.conformer.core.contracts import EmbedSpec, MoleculeSpec
from _david.code.conformer.core.errors import PreparationError
from _david.code.conformer.prepare.standardize import (
    molecule_properties,
    prepare_many,
    prepare_molecule,
)

EGCG = (
    "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)"
    "c1cc(O)c(O)c(O)c1"
)
#: Achiral with one unassigned centre: enumeration must produce more than one.
UNASSIGNED = "CC(N)C(=O)O"
#: Tautomer normalization destroys this molecule's assigned double-bond stereo.
STEREO_FALLBACK = (
    "Nc1nc(/C(=C/CC(=O)O)C(=O)N[C@@H]2C(=O)N3C(C(=O)O)=CCS[C@H]23)cs1"
)


class TestPrepareMolecule(unittest.TestCase):
    def test_egcg_preparation_is_deterministic(self) -> None:
        first = prepare_molecule(MoleculeSpec(name="egcg", smiles=EGCG))
        second = prepare_molecule(MoleculeSpec(name="egcg", smiles=EGCG))
        self.assertEqual(first.canonical_smiles, second.canonical_smiles)
        self.assertEqual(first.n_isomers, second.n_isomers)
        self.assertEqual(first.n_isomers, 1)

    def test_egcg_descriptors_match_thesis_values(self) -> None:
        properties = prepare_molecule(
            MoleculeSpec(name="egcg", smiles=EGCG)
        ).properties
        self.assertEqual(properties["heavy_atoms"], 33)
        self.assertAlmostEqual(properties["molecular_weight"], 458.37, places=2)
        self.assertEqual(properties["rotatable_bonds"], 3)
        self.assertEqual(properties["h_bond_donors"], 8)
        self.assertEqual(properties["h_bond_acceptors"], 11)
        self.assertEqual(properties["stereo_centers"], 2)
        self.assertAlmostEqual(properties["qed"], 0.21, places=2)

    def test_default_preparation_keeps_only_specified_identities(self) -> None:
        prepared = prepare_molecule(MoleculeSpec(name="ala", smiles=UNASSIGNED))
        self.assertEqual(prepared.n_isomers, 1)
        self.assertGreater(prepared.properties["isomer_upper_bound"], 1)

    def test_isomer_cap_enumerates_and_reports_truncation(self) -> None:
        prepared = prepare_molecule(
            MoleculeSpec(name="ala", smiles=UNASSIGNED), EmbedSpec(isomer_cap=8)
        )
        self.assertEqual(prepared.n_isomers, 2)
        self.assertEqual(prepared.properties["enumerated_isomers"], 2)
        self.assertEqual(len(set(prepared.isomer_smiles)), 2)
        self.assertFalse(
            any("truncated" in warning for warning in prepared.warnings),
            prepared.warnings,
        )

    def test_isomer_cap_truncation_is_disclosed(self) -> None:
        prepared = prepare_molecule(
            MoleculeSpec(name="ala", smiles=UNASSIGNED), EmbedSpec(isomer_cap=1)
        )
        self.assertEqual(prepared.n_isomers, 1)
        self.assertTrue(any("truncated" in warning for warning in prepared.warnings))

    def test_stereochemistry_is_never_silently_dropped(self) -> None:
        prepared = prepare_molecule(
            MoleculeSpec(name="ceftibuten", smiles=STEREO_FALLBACK)
        )
        self.assertTrue(
            any(warning.startswith("chemical-state-fallback") for warning in prepared.warnings),
            prepared.warnings,
        )
        self.assertIn("@", prepared.canonical_smiles)

    def test_heavy_atom_cap_is_enforced(self) -> None:
        with self.assertRaises(PreparationError):
            prepare_molecule(
                MoleculeSpec(name="egcg", smiles=EGCG),
                EmbedSpec(heavy_atom_cap=10),
            )

    def test_invalid_smiles_raises_preparation_error(self) -> None:
        with self.assertRaises(PreparationError):
            prepare_molecule(MoleculeSpec(name="bad", smiles="not-a-smiles"))

    def test_prepare_many_records_failures_without_aborting(self) -> None:
        prepared, errors = prepare_many(
            [
                MoleculeSpec(name="ok", smiles="CCO"),
                MoleculeSpec(name="bad", smiles="not-a-smiles"),
            ]
        )
        self.assertEqual(len(prepared), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["molecule"], "bad")

    def test_properties_on_bare_mol(self) -> None:
        from rdkit import Chem

        properties = molecule_properties(Chem.MolFromSmiles("CCO"))
        self.assertEqual(properties["heavy_atoms"], 3)
        self.assertEqual(properties["formal_charge"], 0)
        self.assertEqual(properties["isomer_upper_bound"], 1)


class TestRegistryTables(unittest.TestCase):
    def test_registry_columns_and_flags(self) -> None:
        from _david.code.conformer.analysis.preparation import (
            REGISTRY_COLUMNS,
            REGISTRY_ISOMER_CAP,
            build_registry,
        )

        frame, errors = build_registry(
            [
                MoleculeSpec(name="egcg", smiles=EGCG),
                MoleculeSpec(name="ala", smiles=UNASSIGNED),
            ]
        )
        self.assertEqual(errors, [])
        self.assertEqual(list(frame.columns), list(REGISTRY_COLUMNS))
        self.assertEqual(len(frame), 2)
        self.assertFalse(bool(frame["hit_enumeration_cap"].any()))
        row = frame[frame["name"] == "ala"].iloc[0]
        self.assertEqual(row["isomers_enumerated"], 2)
        self.assertEqual(int(row["enumeration_cap"]), REGISTRY_ISOMER_CAP)

    def test_reference_count_summary_derives_ratios(self) -> None:
        from _david.code.conformer.analysis.preparation import reference_count_summary

        counts = pd.DataFrame(
            {
                "molecule_id": [1, 2],
                "phase": ["reference", "reference"],
                "method": ["MMFF94s", "MMFF94s"],
                "enumerated_isomers": [1, 4],
                "generated_valid": [100, 100],
                "converged_valid": [100, 50],
                "energy_survivors": [50, 25],
                "chemical_preparation_ok": [True, False],
            }
        )
        summary = reference_count_summary(counts)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary["convergence"].iloc[0]), 0.75)
        self.assertAlmostEqual(float(summary["survival"].iloc[0]), 0.375)
        self.assertEqual(int(summary["preparation_failures"].iloc[0]), 1)

    def test_isomer_audit_flags_truncation(self) -> None:
        from _david.code.conformer.analysis.preparation import isomer_audit

        audit = pd.DataFrame(
            {
                "molecule_id": [1, 2],
                "name": ["a", "b"],
                "split": ["development", "development"],
                "specified_stereo": [True, False],
                "original_upper_bound": [1, 16],
                "masked_upper_bound": [1, 16],
                "masked_enumerated_up_to8": [1, 8],
            }
        )
        frame = isomer_audit(audit)
        self.assertFalse(bool(frame.loc[frame["molecule_id"] == 1, "truncated"].iloc[0]))
        self.assertTrue(bool(frame.loc[frame["molecule_id"] == 2, "truncated"].iloc[0]))
        self.assertEqual(frame.iloc[0]["molecule_id"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
