"""Regression checks for scientific meaning and failure handling in the helper."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any
from unittest.mock import patch

SCRIPT_PATH: Path = Path(__file__).resolve().parents[1] / "scripts/drug_design_suite.py"
SPEC = importlib.util.spec_from_file_location("drug_design_suite", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


class DrugDesignSuiteTests(unittest.TestCase):
    """Prevent descriptor mislabeling and loss of assay/source context."""

    @staticmethod
    def response(data: dict[str, Any]) -> io.BytesIO:
        """Create a context-managed API response without a network dependency."""
        return io.BytesIO(json.dumps(data).encode("utf-8"))

    @staticmethod
    def observation() -> dict[str, Any]:
        """Build an exact assay observation containing curation-relevant flags."""
        return {
            "activity_id": 123,
            "molecule_chembl_id": "CHEMBL25",
            "canonical_smiles": None,
            "target_chembl_id": "CHEMBL1993",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "standard_value": "42",
            "assay_chembl_id": "CHEMBL123",
            "document_chembl_id": "CHEMBL456",
            "data_validity_comment": "Outside typical range",
            "potential_duplicate": 1,
        }

    def test_average_weight_and_exact_mass_are_distinct(self) -> None:
        """Benzene's average molecular weight must not contain its exact mass."""
        profile = SUITE.PubChemResolver.profile_molecule("c1ccccc1")
        self.assertAlmostEqual(profile.molecular_weight, 78.114, places=3)
        self.assertAlmostEqual(profile.exact_molecular_mass, 78.046950, places=5)
        self.assertNotEqual(profile.to_dict()["molecular_weight"], profile.to_dict()["exact_molecular_mass"])

    def test_profile_preserves_charge_salt_and_stereochemistry(self) -> None:
        """Profiling does not silently perform the old promised pH/parent cleanup."""
        smiles = "[Na+].C[C@H](O)C(=O)[O-]"
        profile = SUITE.PubChemResolver.profile_molecule(smiles)
        self.assertIn("[Na+]", profile.canonical_smiles)
        self.assertIn("[O-]", profile.canonical_smiles)
        self.assertIn("@", profile.canonical_smiles)

    def test_empty_smiles_is_an_error(self) -> None:
        """An empty molecule must not produce plausible all-zero descriptors."""
        with self.assertRaises(ValueError):
            SUITE.PubChemResolver.profile_molecule("")

    def test_name_resolution_does_not_fall_back_to_connectivity(self) -> None:
        """A response lacking stereo-preserving SMILES is an explicit failure."""
        payload = {"PropertyTable": {"Properties": [{"ConnectivitySMILES": "CC(O)C(=O)O"}]}}
        with (
            patch.object(SUITE.urllib.request, "urlopen", return_value=self.response(payload)),
            self.assertRaises(ValueError),
        ):
            SUITE.PubChemResolver.resolve_name_to_smiles("test ligand")

    def test_activity_query_retains_curation_fields_and_missing_structure(self) -> None:
        """Flags, source IDs, and missing structures remain visible to curation."""
        with patch.object(
            SUITE.urllib.request, "urlopen",
            return_value=self.response({"activities": [self.observation()]}),
        ) as request:
            records = SUITE.ChEMBLMiner.query_target_actives("chembl1993", limit=5)
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(request.call_args.args[0].full_url).query)
        self.assertEqual(params["standard_relation"], ["="])
        self.assertEqual(params["standard_value__gt"], ["0"])
        self.assertEqual(records[0]["standard_value_nm"], 42.0)
        self.assertIsNone(records[0]["canonical_smiles"])
        self.assertEqual(records[0]["data_validity_comment"], "Outside typical range")
        self.assertEqual(records[0]["potential_duplicate"], 1)
        self.assertEqual(records[0]["document_chembl_id"], "CHEMBL456")

    def test_censored_or_malformed_activity_is_not_reported_as_exact(self) -> None:
        """API/schema mismatches fail instead of silently dropping observations."""
        for changes in ({"standard_relation": ">"}, {"standard_value": "nan"}, {"target_chembl_id": "CHEMBL2016"}):
            with self.subTest(changes=changes):
                activity = self.observation() | changes
                with patch.object(
                    SUITE.urllib.request, "urlopen",
                    return_value=self.response({"activities": [activity]}),
                ), self.assertRaises(ValueError):
                    SUITE.ChEMBLMiner.query_target_actives("CHEMBL1993")

    def test_invalid_query_budget_or_cutoff_never_calls_network(self) -> None:
        """Reject unbounded/invalid requests rather than relying on API defaults."""
        with patch.object(SUITE.urllib.request, "urlopen") as request:
            for kwargs in ({"limit": 0}, {"limit": 1001}, {"max_ic50_nm": float("nan")}, {"max_ic50_nm": -1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    SUITE.ChEMBLMiner.query_target_actives("CHEMBL1993", **kwargs)
            request.assert_not_called()

    def test_pdb_identifier_cannot_escape_download_directory(self) -> None:
        """Reject path components before any filesystem or network work."""
        with self.assertRaises(ValueError):
            SUITE.PDBTargetManager.normalize_id("../3PTA")
        self.assertEqual(SUITE.PDBTargetManager.normalize_id(" 3pta "), "3PTA")

    def test_failed_download_preserves_existing_coordinate_file(self) -> None:
        """A read timeout must not truncate an already downloaded structure."""
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "3pta.pdb"
            destination.write_bytes(b"existing coordinates")
            with patch.object(SUITE.urllib.request, "urlopen") as request:
                request.return_value.__enter__.return_value.read.side_effect = TimeoutError("read timed out")
                with self.assertRaises(TimeoutError):
                    SUITE.PDBTargetManager.download_structure("3PTA", folder)
            self.assertEqual(destination.read_bytes(), b"existing coordinates")

    def test_literature_preserves_source_type_and_core_journal(self) -> None:
        """Core records use nested journal metadata and retain publication types."""
        payload = {"resultList": {"result": [{
            "id": "PPR1", "source": "PPR", "title": "Example preprint",
            "pubTypeList": {"pubType": ["preprint"]},
            "journalInfo": {"journal": {"title": "Example journal"}},
        }]}}
        with patch.object(SUITE.urllib.request, "urlopen", return_value=self.response(payload)):
            record = SUITE.LiteratureDiscoverer.search_europe_pmc("example")[0]
        self.assertEqual(record["source"], "PPR")
        self.assertEqual(record["publication_types"], ["preprint"])
        self.assertEqual(record["journal"], "Example journal")

    def test_cli_rejects_ambiguous_input_and_misleading_export(self) -> None:
        """Input conflicts and TSV filenames fail before fetching any data."""
        for args in (
            ["profile", "--name", "EGCG", "--smiles", "CC"],
            ["profile"],
            ["chembl", "--target", "CHEMBL1993", "--output", "actives.tsv"],
        ):
            with self.subTest(args=args), patch.object(sys, "argv", [str(SCRIPT_PATH), *args]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    SUITE.DrugDesignSuiteCLI.run()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
