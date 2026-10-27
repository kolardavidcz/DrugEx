"""Small public-API retrieval utility; outputs are not curated scientific evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class PhysicochemicalProfile:
    """Descriptors of the supplied structure without parent/pH standardization."""

    smiles: str
    canonical_smiles: str
    molecular_weight: float
    exact_molecular_mass: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    heavy_atom_count: int
    formal_charge: int
    fraction_sp3: float

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "molecular_weight": round(self.molecular_weight, 3),
            "exact_molecular_mass": round(self.exact_molecular_mass, 6),
            "logp": round(self.logp, 3),
            "tpsa": round(self.tpsa, 2),
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "heavy_atom_count": self.heavy_atom_count,
            "formal_charge": self.formal_charge,
            "fraction_sp3": round(self.fraction_sp3, 3),
        }


class PubChemResolver:
    """Robust compound resolver leveraging PubChem PUG REST API and RDKit."""

    PUG_BASE: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    @classmethod
    def resolve_name_to_smiles(cls, compound_name: str) -> str:
        """Resolve a name to PubChem SMILES, preserving specified stereochemistry."""
        if not compound_name.strip():
            raise ValueError("Compound name must not be empty")
        encoded_name = urllib.parse.quote(compound_name.strip(), safe="")
        url = f"{cls.PUG_BASE}/name/{encoded_name}/property/SMILES/JSON"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DrugDesignScience/1.0 (academic-thesis)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        properties = data["PropertyTable"]["Properties"][0]
        smi = properties.get("SMILES")
        if not smi:
            raise ValueError(f"No SMILES found for compound '{compound_name}'")
        return str(smi)

    @staticmethod
    def profile_molecule(smiles_or_mol: str | Chem.Mol) -> PhysicochemicalProfile:
        """Compute RDKit descriptors without changing salts, charge, or stereochemistry."""
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f"Failed to parse SMILES: '{smiles_or_mol}'")
            raw_smiles = smiles_or_mol
        else:
            mol = smiles_or_mol
            raw_smiles = Chem.MolToSmiles(mol)

        canon_smi = Chem.MolToSmiles(mol, canonical=True)

        return PhysicochemicalProfile(
            smiles=raw_smiles,
            canonical_smiles=canon_smi,
            molecular_weight=float(Descriptors.MolWt(mol)),
            exact_molecular_mass=float(Descriptors.ExactMolWt(mol)),
            logp=float(Descriptors.MolLogP(mol)),
            tpsa=float(Descriptors.TPSA(mol)),
            hbd=int(Lipinski.NumHDonors(mol)),
            hba=int(Lipinski.NumHAcceptors(mol)),
            rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
            heavy_atom_count=int(mol.GetNumHeavyAtoms()),
            formal_charge=int(Chem.GetFormalCharge(mol)),
            fraction_sp3=float(rdMolDescriptors.CalcFractionCSP3(mol)),
        )


class PDBTargetManager:
    """Interface for querying structural targets and downloading experimentally resolved PDB/mmCIF files."""

    RCSB_DATA_URL: str = "https://data.rcsb.org/rest/v1/core/entry"
    RCSB_FILES_URL: str = "https://files.rcsb.org/download"

    @staticmethod
    def normalize_id(pdb_id: str) -> str:
        """Validate a classic four-character PDB ID before URL/path construction."""
        clean_id = pdb_id.strip().upper()
        if re.fullmatch(r"[0-9][A-Z0-9]{3}", clean_id) is None:
            raise ValueError("Expected a classic four-character PDB ID, e.g. 3PTA")
        return clean_id

    @classmethod
    def fetch_entry_summary(cls, pdb_id: str) -> dict[str, Any]:
        """Fetch entry-level metadata; ligand/chain details need entity endpoints."""
        clean_id = cls.normalize_id(pdb_id)
        url = f"{cls.RCSB_DATA_URL}/{clean_id}"

        req = urllib.request.Request(url, headers={"User-Agent": "DrugDesignScience/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        expt = data.get("exptl", [{}])[0]
        refine = data.get("rcsb_entry_info", {})

        return {
            "pdb_id": clean_id,
            "title": data.get("struct", {}).get("title", "Unknown Title"),
            "method": expt.get("method", "Unknown"),
            "resolution_angstrom": refine.get("resolution_combined", [None])[0],
            "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date"),
            "molecular_weight_kda": refine.get("molecular_weight"),
        }

    @classmethod
    def download_structure(
        cls,
        pdb_id: str,
        dest_dir: Path | str,
        file_format: str = "pdb",
    ) -> Path:
        """Download experimental 3D coordinate file (.pdb or .cif) for molecular visualization."""
        clean_id = cls.normalize_id(pdb_id).lower()
        if file_format.lower() not in ("pdb", "cif", "mmcif"):
            raise ValueError("Structure format must be pdb, cif, or mmcif")
        ext = "cif" if file_format.lower() in ("cif", "mmcif") else "pdb"
        target_dir = Path(dest_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / f"{clean_id}.{ext}"

        url = f"{cls.RCSB_FILES_URL}/{clean_id}.{ext}"
        req = urllib.request.Request(url, headers={"User-Agent": "DrugDesignScience/1.0"})

        with urllib.request.urlopen(req, timeout=20) as resp:
            coordinates = resp.read()
        out_file.write_bytes(coordinates)

        return out_file


class ChEMBLMiner:
    """Retrieve bounded activity observations; assay curation is a separate step."""

    CHEMBL_API_BASE: str = "https://www.ebi.ac.uk/chembl/api/data"

    @staticmethod
    def normalize_target_id(target_chembl_id: str) -> str:
        """Validate a target identifier before constructing API URLs."""
        clean_id = target_chembl_id.strip().upper()
        if re.fullmatch(r"CHEMBL[0-9]+", clean_id) is None:
            raise ValueError("Expected a ChEMBL target identifier, e.g. CHEMBL1993")
        return clean_id

    @classmethod
    def resolve_target(cls, target_chembl_id: str) -> dict[str, Any]:
        """Return target identity so callers can check species/name before mining."""
        clean_id = cls.normalize_target_id(target_chembl_id)
        req = urllib.request.Request(
            f"{cls.CHEMBL_API_BASE}/target/{clean_id}.json",
            headers={"User-Agent": "DrugDesignScience/1.0"},
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            return json.load(response)

    @classmethod
    def query_target_actives(
        cls,
        target_chembl_id: str,
        max_ic50_nm: float = 10000.0,  # 10 uM cutoff
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve one page of exact positive IC50/Ki records, retaining provenance.

        No assay-confidence join, mechanism inference, or completeness is implied.
        The legacy max_ic50_nm argument applies to both returned endpoint types.
        """
        clean_id = cls.normalize_target_id(target_chembl_id)
        if not math.isfinite(max_ic50_nm) or max_ic50_nm <= 0:
            raise ValueError("Activity cutoff must be finite and positive")
        if not 1 <= limit <= 1000:
            raise ValueError("ChEMBL discovery limit must be between 1 and 1000")
        query_params = {
            "target_chembl_id": clean_id,
            "standard_type__in": "IC50,Ki",
            "standard_value__lte": str(max_ic50_nm),
            "standard_value__gt": "0",
            "standard_relation": "=",
            "standard_units": "nM",
            "format": "json",
            "limit": str(limit),
        }
        encoded = urllib.parse.urlencode(query_params)
        url = f"{cls.CHEMBL_API_BASE}/activity?{encoded}"

        req = urllib.request.Request(url, headers={"User-Agent": "DrugDesignScience/1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))

        records: list[dict[str, Any]] = []
        for act in data.get("activities", []):
            value = float(act["standard_value"])
            if (
                not math.isfinite(value)
                or not 0 < value <= max_ic50_nm
                or act.get("standard_relation") != "="
                or act.get("standard_units") != "nM"
                or act.get("standard_type") not in ("IC50", "Ki")
                or act.get("target_chembl_id") != clean_id
            ):
                raise ValueError(f"ChEMBL returned an observation outside the query: {act.get('activity_id')}")
            # Keep missing structures and validity flags visible for downstream curation.
            records.append(
                {
                    key: act.get(key)
                    for key in (
                        "activity_id", "molecule_chembl_id", "canonical_smiles",
                        "standard_type", "standard_relation", "standard_units",
                        "pchembl_value", "assay_chembl_id", "assay_type",
                        "target_chembl_id", "target_organism", "document_chembl_id",
                        "data_validity_comment", "potential_duplicate",
                    )
                }
            )
            records[-1]["standard_value_nm"] = value

        return records


class LiteratureDiscoverer:
    """Retrieve Europe PMC discovery metadata; no full-text review is performed."""

    EUROPE_PMC_URL: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @classmethod
    def search_europe_pmc(cls, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Europe PMC records without retrieving or appraising article text."""
        if not query.strip() or not 1 <= limit <= 1000:
            raise ValueError("Supply a nonempty literature query and a limit between 1 and 1000")
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(limit),
            "resultType": "core",
        }
        url = f"{cls.EUROPE_PMC_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "DrugDesignScience/1.0"})

        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results: list[dict[str, Any]] = []
        for item in data.get("resultList", {}).get("result", []):
            results.append(
                {
                    "title": item.get("title"),
                    "author_string": item.get("authorString"),
                    "journal": item.get("journalInfo", {}).get("journal", {}).get("title"),
                    "year": item.get("pubYear"),
                    "doi": item.get("doi"),
                    "pmcid": item.get("pmcid"),
                    "pmid": item.get("pmid"),
                    "source": item.get("source"),
                    "id": item.get("id"),
                    "publication_types": item.get("pubTypeList", {}).get("pubType", []),
                    "is_open_access": item.get("isOpenAccess") == "Y",
                }
            )
        return results


class DrugDesignSuiteCLI:
    """Unified command-line interface for the drug-design-science meta-skill."""

    @staticmethod
    def run() -> None:
        """Parse CLI arguments and dispatch task."""
        parser = argparse.ArgumentParser(
            description="Universal Computational Drug Design & Cheminformatics Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        # 1. Profile / Resolve Ligand
        profile_parser = subparsers.add_parser("profile", help="Resolve compound and calculate physicochemical envelope")
        profile_input = profile_parser.add_mutually_exclusive_group(required=True)
        profile_input.add_argument("--name", type=str, help="Common chemical name (e.g., 'EGCG', 'aspirin')")
        profile_input.add_argument("--smiles", type=str, help="SMILES string to evaluate")

        # 2. PDB Structure Fetch
        pdb_parser = subparsers.add_parser("pdb", help="Fetch PDB metadata and download 3D coordinates")
        pdb_parser.add_argument("--id", type=str, required=True, help="4-letter PDB ID (e.g. '3PTA', '4WXX')")
        pdb_parser.add_argument("--download", type=str, help="Directory to save downloaded .pdb file")

        # 3. ChEMBL Target Actives
        chembl_parser = subparsers.add_parser("chembl", help="Retrieve an uncurated page of activity observations")
        chembl_parser.add_argument("--target", type=str, required=True, help="Target ChEMBL ID (e.g., 'CHEMBL203' for EGFR)")
        chembl_parser.add_argument("--cutoff-nm", type=float, default=10000.0, help="IC50/Ki cutoff in nM (default: 10000 nM)")
        chembl_parser.add_argument("--limit", type=int, default=25, help="Maximum records to retrieve")
        chembl_parser.add_argument("--output", type=str, help="Save uncurated records to a .json file")

        # 4. Literature Search
        lit_parser = subparsers.add_parser("literature", help="Search peer-reviewed and preprint literature")
        lit_parser.add_argument("--query", type=str, required=True, help="Literature search terms")
        lit_parser.add_argument("--limit", type=int, default=5, help="Number of citations to retrieve")

        args = parser.parse_args()
        if args.command == "chembl" and args.output and Path(args.output).suffix.lower() != ".json":
            parser.error("ChEMBL output supports .json only; TSV export is not implemented")

        if args.command == "profile":
            smiles = args.smiles
            if args.name is not None:
                print(f"Resolving compound name '{args.name}' via PubChem...")
                smiles = PubChemResolver.resolve_name_to_smiles(args.name)
                print(f"  -> Resolved SMILES: {smiles}")

            profile = PubChemResolver.profile_molecule(smiles)
            print("\nPHYSICOCHEMICAL PROFILE:")
            print(json.dumps(profile.to_dict(), indent=2))

        elif args.command == "pdb":
            meta = PDBTargetManager.fetch_entry_summary(args.id)
            print(f"\nPDB ENTRY: {meta['pdb_id']}")
            print(f"  Title: {meta['title']}")
            print(f"  Method: {meta['method']}")
            print(f"  Resolution: {meta['resolution_angstrom']} A")
            if args.download:
                out_path = PDBTargetManager.download_structure(args.id, args.download)
                print(f"  -> Downloaded 3D Coordinate File: {out_path}")

        elif args.command == "chembl":
            target = ChEMBLMiner.resolve_target(args.target)
            print(f"Target: {target['target_chembl_id']} | {target['pref_name']} | {target['organism']} | {target['target_type']}")
            print(f"Retrieving exact IC50/Ki observations (cutoff <= {args.cutoff_nm} nM)...")
            records = ChEMBLMiner.query_target_actives(args.target, max_ic50_nm=args.cutoff_nm, limit=args.limit)
            print(f"  -> Retrieved {len(records)} uncurated observations (one page; not a complete dataset).")
            if records:
                print(f"  Sample observation: {records[0]['molecule_chembl_id']} | {records[0]['standard_type']} = {records[0]['standard_value_nm']} nM")
            if args.output:
                out_p = Path(args.output).resolve()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
                print(f"  -> Saved to {out_p}")

        elif args.command == "literature":
            print(f"Searching Europe PMC for: '{args.query}'...")
            results = LiteratureDiscoverer.search_europe_pmc(args.query, limit=args.limit)
            print(f"\nFound {len(results)} discovery records; peer review and full text not verified:")
            for idx, r in enumerate(results, start=1):
                doi_str = f"https://doi.org/{r['doi']}" if r['doi'] else "No DOI"
                print(f"\n{idx}. {r['title']} ({r['year']})")
                print(f"   Authors: {r['author_string']}")
                print(f"   Journal: {r['journal']} | DOI: {doi_str} | OpenAccess: {r['is_open_access']}")
                print(f"   Source/ID: {r['source']}/{r['id']} | PMCID: {r['pmcid']}")


if __name__ == "__main__":
    DrugDesignSuiteCLI.run()
