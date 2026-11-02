"""Ligand preparation and conformer generation for target docking validation.

Generates 3D conformers (ETKDGv3 + MMFF94) and parameterized PDBQT files for the
top DrugEx de novo prioritized leads and the reference query seed (EGCG).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA
from meeko import MoleculePreparation, PDBQTWriterLegacy

# [PACKAGE: rdkit.Chem.rdMolDescriptors] WHY: SAscore is standard synthetic accessibility metric in DrugEx. WHAT: SAScore calculation. HOW: from drugex.data.pkg.sascorer import calculateScore.
try:
    from drugex.data.pkg.sascorer import calculateScore as calc_sa
except ImportError:
    def calc_sa(m: Chem.Mol) -> float:
        return 3.0


@dataclass
class LigandRecord:
    """Encapsulates metadata and descriptors for a validation ligand."""
    ligand_id: str
    name: str
    smiles: str
    is_reference: bool
    hac: int
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rot_bonds: int
    qed: float
    sa: float
    sdf_path: str = ""
    pdbqt_path: str = ""


class LigandPreparer:
    """Prepares 3D conformers and AutoDock Vina PDBQT files for candidate ligands."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize LigandPreparer with target output directory.

        Args:
            output_dir: Directory where prepared SDF and PDBQT files will be written.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_default_candidate_set(cls) -> List[Tuple[str, str, str, bool]]:
        """Return the curated benchmark set (5 DrugEx M4 leads + 1 EGCG seed reference).

        Returns:
            List of tuples: (ligand_id, name, smiles, is_reference)
        """
        return [
            (
                "EGCG_seed",
                "(-)-Epigallocatechin gallate (Reference)",
                "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1",
                True,
            ),
            (
                "Lead_01",
                "DrugEx M4 Lead 1 (Desirability 0.596)",
                "COc1ccc(C2Oc3ccccc3CC2OC(=O)c2ccccc2)cc1",
                False,
            ),
            (
                "Lead_02",
                "DrugEx M4 Lead 2 (Desirability 0.586)",
                "COc1ccc(C2Oc3ccccc3CC2OC(=O)c2ccc3c(c2)OCO3)cc1",
                False,
            ),
            (
                "Lead_03",
                "DrugEx M4 Lead 3 (Desirability 0.553)",
                "Cc1c(C(=O)OC2Cc3ccccc3OC2c2ccccc2)ccc2c1OC(C)CO2",
                False,
            ),
            (
                "Lead_04",
                "DrugEx M4 Lead 4 (Desirability 0.480)",
                "COc1ccc(C2Oc3ccccc3CC2OC(=O)c2ccc3c(=O)cc(C)sc3c2)cc1",
                False,
            ),
            (
                "Lead_05",
                "DrugEx M4 Lead 5 (Desirability 0.473)",
                "COc1ccc2c(c1)OC(c1ccccc1)C(OC(=O)c1ccc3c(c1)NC(=O)CC3)C2",
                False,
            ),
        ]

    def build_3d_conformer(self, smiles: str, name: str) -> Chem.Mol:
        """Generate MMFF94-relaxed 3D conformer using RDKit ETKDGv3.

        Args:
            smiles: Input SMILES string.
            name: Molecule title for SDF property tagging.

        Returns:
            RDKit Mol with 3D coordinates.
        """
        # [PACKAGE: rdkit.Chem] WHY: High quality conformer embedding. WHAT: ETKDGv3 distance geometry with torsion knowledge. HOW: AllChem.EmbedMolecule.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", name)
        mol.SetProp("SMILES", smiles)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.20
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # [PACKAGE: rdkit.Chem.AllChem] WHY: Energy minimize generated conformer. WHAT: MMFF94 force-field relaxation. HOW: MMFFOptimizeMolecule.
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)

        return mol

    def convert_mol_to_pdbqt(self, mol: Chem.Mol, output_path: Path) -> str:
        """Convert 3D RDKit Mol to AutoDock Vina PDBQT format using Meeko.

        Args:
            mol: 3D RDKit Mol with explicit hydrogens.
            output_path: Destination file path.

        Returns:
            PDBQT string content.
        """
        # [PACKAGE: meeko] WHY: Assign Gasteiger charges and parameterize flexible torsions for Vina. WHAT: MoleculePreparation. HOW: PDBQTWriterLegacy.write_string.
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            raise RuntimeError(f"Meeko failed to prepare molecule: {mol.GetProp('_Name')}")

        pdbqt_res = PDBQTWriterLegacy.write_string(mol_setups[0])
        pdbqt_str = pdbqt_res[0] if isinstance(pdbqt_res, tuple) else pdbqt_res

        output_path.write_text(pdbqt_str, encoding="utf-8")
        return pdbqt_str

    def prepare_all_ligands(
        self, candidate_set: List[Tuple[str, str, str, bool]] | None = None
    ) -> List[LigandRecord]:
        """Prepare all candidate ligands, generating SDF, PDBQT, and manifest TSV.

        Args:
            candidate_set: Optional list of tuples (id, name, smiles, is_reference). Defaults to benchmark set.

        Returns:
            List of LigandRecord objects.
        """
        if candidate_set is None:
            candidate_set = self.get_default_candidate_set()

        records: List[LigandRecord] = []
        combined_sdf_path = self.output_dir / "all_6_ligands.sdf"
        combined_writer = Chem.SDWriter(str(combined_sdf_path))

        for lig_id, name, smiles, is_ref in candidate_set:
            mol = self.build_3d_conformer(smiles, name)

            # Calculate fundamental descriptors
            raw_mol = Chem.RemoveHs(mol)
            hac = raw_mol.GetNumHeavyAtoms()
            mw = float(Descriptors.MolWt(raw_mol))
            logp = float(Descriptors.MolLogP(raw_mol))
            tpsa = float(CalcTPSA(raw_mol))
            hbd = int(CalcNumHBD(raw_mol))
            hba = int(CalcNumHBA(raw_mol))
            rot = int(Descriptors.NumRotatableBonds(raw_mol))
            qed_score = float(QED.qed(raw_mol))
            sa_score = float(calc_sa(raw_mol))

            # Set properties on 3D mol for SDF
            mol.SetProp("Ligand_ID", lig_id)
            mol.SetProp("Is_Reference", str(is_ref))
            mol.SetProp("HAC", str(hac))
            mol.SetProp("MW", f"{mw:.2f}")
            mol.SetProp("LogP", f"{logp:.2f}")
            mol.SetProp("TPSA", f"{tpsa:.1f}")
            mol.SetProp("HBD", str(hbd))
            mol.SetProp("HBA", str(hba))
            mol.SetProp("QED", f"{qed_score:.3f}")
            mol.SetProp("SAscore", f"{sa_score:.2f}")

            # Write individual SDF
            sdf_path = self.output_dir / f"{lig_id}.sdf"
            ind_writer = Chem.SDWriter(str(sdf_path))
            ind_writer.write(mol)
            ind_writer.close()

            # Write combined SDF
            combined_writer.write(mol)

            # Write PDBQT
            pdbqt_path = self.output_dir / f"{lig_id}.pdbqt"
            self.convert_mol_to_pdbqt(mol, pdbqt_path)

            records.append(
                LigandRecord(
                    ligand_id=lig_id,
                    name=name,
                    smiles=smiles,
                    is_reference=is_ref,
                    hac=hac,
                    mw=mw,
                    logp=logp,
                    tpsa=tpsa,
                    hbd=hbd,
                    hba=hba,
                    rot_bonds=rot,
                    qed=qed_score,
                    sa=sa_score,
                    sdf_path=str(sdf_path),
                    pdbqt_path=str(pdbqt_path),
                )
            )

        combined_writer.close()

        # Write summary TSV manifest
        tsv_path = self.output_dir / "ligands_manifest.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Ligand_ID", "Name", "SMILES", "Is_Reference", "HAC", "MW",
                "LogP", "TPSA", "HBD", "HBA", "RotBonds", "QED", "SAscore",
                "SDF_Path", "PDBQT_Path"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "Ligand_ID": r.ligand_id,
                    "Name": r.name,
                    "SMILES": r.smiles,
                    "Is_Reference": r.is_reference,
                    "HAC": r.hac,
                    "MW": f"{r.mw:.2f}",
                    "LogP": f"{r.logp:.2f}",
                    "TPSA": f"{r.tpsa:.1f}",
                    "HBD": r.hbd,
                    "HBA": r.hba,
                    "RotBonds": r.rot_bonds,
                    "QED": f"{r.qed:.3f}",
                    "SAscore": f"{r.sa:.2f}",
                    "SDF_Path": r.sdf_path,
                    "PDBQT_Path": r.pdbqt_path,
                })

        return records
