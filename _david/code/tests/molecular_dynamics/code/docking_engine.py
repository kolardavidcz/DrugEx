"""Docking engine and interaction profiler for multi-target validation.

Executes AutoDock Vina across all candidate ligands and targets, converts docked
poses to complex structures, and profiles residue-level interactions with PLIP.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from meeko import PDBQTMolecule, RDKitMolCreate
from plip.structure.preparation import PDBComplex
from rdkit import Chem
from vina import Vina

from _david.validation.code.prepare_ligands import LigandRecord
from _david.validation.code.prepare_targets import TargetMetadata


@dataclass
class DockingInteractionProfile:
    """Residue-level interaction metrics extracted via PLIP."""
    num_hbonds_prot_donor: int = 0
    num_hbonds_lig_donor: int = 0
    num_hydrophobic: int = 0
    num_pi_stacking: int = 0
    num_salt_bridges: int = 0
    interacting_residues: List[str] = field(default_factory=list)
    hbond_donor_details: List[str] = field(default_factory=list)


@dataclass
class DockingResult:
    """Complete docking score, pose paths, and PLIP profile for a ligand-target pair."""
    ligand_id: str
    target_id: str
    target_name: str
    biological_class: str
    affinity_kcal_mol: float
    is_reference: bool
    docked_pdbqt_path: str
    complex_pdb_path: str
    interactions: DockingInteractionProfile


class VinaDockingEngine:
    """Manages multi-target AutoDock Vina runs and PLIP interaction characterization."""

    def __init__(self, output_dir: Path | str, exhaustiveness: int = 4, cpu_cores: int = 4) -> None:
        """Initialize VinaDockingEngine.

        Args:
            output_dir: Directory to save docked poses and results.
            exhaustiveness: Vina search exhaustiveness (default 4 for agile validation).
            cpu_cores: Number of parallel CPU threads for Vina.
        """
        self.output_dir: Path = Path(output_dir)
        self.exhaustiveness: int = exhaustiveness
        self.cpu_cores: int = cpu_cores

        self.poses_dir: Path = self.output_dir / "poses"
        self.complexes_dir: Path = self.output_dir / "complexes"
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.complexes_dir.mkdir(parents=True, exist_ok=True)

    def _convert_docked_pose_to_hetatm_block(self, docked_pdbqt: Path) -> str:
        """Convert docked PDBQT pose to standard PDB HETATM records with LIG resname."""
        # [PACKAGE: meeko] WHY: Decode Vina output PDBQT into 3D RDKit Mol coordinates. WHAT: PDBQTMolecule.from_file + RDKitMolCreate. HOW: Generate clean PDB HETATM block.
        pdbqt_mol = PDBQTMolecule.from_file(str(docked_pdbqt), is_dlg=False, skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if not rdkit_mols:
            raise RuntimeError(f"Meeko failed to parse docked pose: {docked_pdbqt}")

        mol = rdkit_mols[0]
        pdb_block = Chem.MolToPDBBlock(mol)

        hetatm_lines: List[str] = []
        for line in pdb_block.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                # Format as HETATM, chain Z, residue LIG 1
                formatted = f"HETATM{line[6:17]}LIG Z   1{line[26:]}"
                hetatm_lines.append(formatted)

        return "\n".join(hetatm_lines)

    def _build_complex_pdb(
        self, receptor_clean_pdb: Path, docked_pdbqt: Path, complex_out_path: Path
    ) -> None:
        """Concatenate receptor PDB and ligand HETATM records into a unified complex for PLIP."""
        rec_lines = [
            l for l in receptor_clean_pdb.read_text(encoding="utf-8").splitlines()
            if not l.startswith("END")
        ]
        lig_block = self._convert_docked_pose_to_hetatm_block(docked_pdbqt)

        complex_text = "\n".join(rec_lines) + "\n" + lig_block + "\nEND\n"
        complex_out_path.write_text(complex_text, encoding="utf-8")

    def _analyze_interactions_with_plip(self, complex_pdb_path: Path) -> DockingInteractionProfile:
        """Extract atomic interactions from complex PDB using PLIP."""
        # [PACKAGE: plip] WHY: Rigorous biophysical contact profiling (H-bonds, pi-stacking, hydrophobic). WHAT: PDBComplex analysis. HOW: Extract interaction sets for ligand.
        c = PDBComplex()
        c.load_pdb(str(complex_pdb_path))
        c.analyze()

        profile = DockingInteractionProfile()
        if not c.interaction_sets:
            return profile

        # Take first characterized ligand interaction set
        bs = next(iter(c.interaction_sets.values()))

        profile.num_hbonds_prot_donor = len(bs.hbonds_pdon)
        profile.num_hbonds_lig_donor = len(bs.hbonds_ldon)
        profile.num_hydrophobic = len(bs.hydrophobic_contacts)
        profile.num_pi_stacking = len(bs.pistacking)
        profile.num_salt_bridges = len(bs.saltbridge_lneg) + len(bs.saltbridge_pneg)

        residues = set()
        donor_details = []

        for hb in bs.hbonds_pdon:
            res_str = f"{hb.restype}{hb.resnr}"
            residues.add(res_str)
            donor_details.append(f"Protein donor {res_str} -> Ligand {hb.atype}")

        for hb in bs.hbonds_ldon:
            res_str = f"{hb.restype}{hb.resnr}"
            residues.add(res_str)
            donor_details.append(f"Ligand donor {hb.dtype} -> Protein {res_str}")

        for hp in bs.hydrophobic_contacts:
            residues.add(f"{hp.restype}{hp.resnr}")

        for pi in bs.pistacking:
            residues.add(f"{pi.restype}{pi.resnr}")

        profile.interacting_residues = sorted(list(residues))
        profile.hbond_donor_details = donor_details

        return profile

    def dock_ligand_to_target(
        self, ligand: LigandRecord, target: TargetMetadata, receptor_clean_pdb: Path
    ) -> DockingResult:
        """Dock a single ligand into a specific target and analyze contacts.

        Args:
            ligand: Candidate LigandRecord.
            target: Target metadata containing grid box and receptor PDBQT.
            receptor_clean_pdb: Path to cleaned protein PDB.

        Returns:
            DockingResult with binding energy and PLIP profile.
        """
        # [PACKAGE: vina] WHY: High-accuracy empirical binding affinity calculation. WHAT: AutoDock Vina Python API. HOW: compute_vina_maps + dock.
        v = Vina(sf_name="vina", cpu=self.cpu_cores, verbosity=0)
        v.set_receptor(target.receptor_pdbqt)
        v.set_ligand_from_file(ligand.pdbqt_path)

        center_list = list(target.box_center)
        size_list = list(target.box_size)
        v.compute_vina_maps(center=center_list, box_size=size_list)

        v.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
        energies = v.energies(n_poses=1)
        affinity = float(energies[0][0]) if len(energies) > 0 else 0.0

        # Save docked pose
        pose_pdbqt = self.poses_dir / f"{target.target_id}_{ligand.ligand_id}_docked.pdbqt"
        v.write_poses(str(pose_pdbqt), n_poses=1, overwrite=True)

        # Build complex and analyze PLIP interactions
        complex_pdb = self.complexes_dir / f"{target.target_id}_{ligand.ligand_id}_complex.pdb"
        self._build_complex_pdb(receptor_clean_pdb, pose_pdbqt, complex_pdb)

        interactions = self._analyze_interactions_with_plip(complex_pdb)

        return DockingResult(
            ligand_id=ligand.ligand_id,
            target_id=target.target_id,
            target_name=target.common_name,
            biological_class=target.biological_class,
            affinity_kcal_mol=affinity,
            is_reference=ligand.is_reference,
            docked_pdbqt_path=str(pose_pdbqt),
            complex_pdb_path=str(complex_pdb),
            interactions=interactions,
        )

    def run_full_validation_matrix(
        self,
        ligands: List[LigandRecord],
        targets: Dict[str, TargetMetadata],
        receptors_root: Path | str,
    ) -> List[DockingResult]:
        """Execute complete cross-docking matrix (6 ligands x 3 targets = 18 experiments).

        Args:
            ligands: List of prepared ligands.
            targets: Dictionary of prepared targets.
            receptors_root: Root directory containing target subfolders.

        Returns:
            List of DockingResult objects.
        """
        results: List[DockingResult] = []
        rec_root = Path(receptors_root)

        for tid, meta in targets.items():
            clean_pdb = rec_root / tid / "receptor_clean.pdb"
            for lig in ligands:
                print(f"  [Docking] {lig.ligand_id} -> {meta.target_id} ({meta.common_name})...")
                res = self.dock_ligand_to_target(lig, meta, clean_pdb)
                print(
                    f"    Affinity: {res.affinity_kcal_mol:.2f} kcal/mol | "
                    f"H-bonds (P/L): {res.interactions.num_hbonds_prot_donor}/{res.interactions.num_hbonds_lig_donor} | "
                    f"Hydrophobic: {res.interactions.num_hydrophobic}"
                )
                results.append(res)

        # Save collective results JSON
        json_path = self.output_dir / "docking_results_matrix.json"
        serializable = [
            {
                "ligand_id": r.ligand_id,
                "target_id": r.target_id,
                "target_name": r.target_name,
                "biological_class": r.biological_class,
                "affinity_kcal_mol": r.affinity_kcal_mol,
                "is_reference": r.is_reference,
                "docked_pdbqt_path": r.docked_pdbqt_path,
                "complex_pdb_path": r.complex_pdb_path,
                "interactions": asdict(r.interactions),
            }
            for r in results
        ]
        json_path.write_text(json.dumps(serializable, indent=4), encoding="utf-8")

        return results
