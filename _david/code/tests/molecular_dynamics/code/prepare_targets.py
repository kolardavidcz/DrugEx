"""Target receptor preparation and grid box calibration for docking validation.

Parses experimental structures (mmCIF / PDB), isolates the native co-crystallized
ligand (KDH/EGCG) to establish ground-truth binding envelopes, strips solvents/ions,
and parameterizes receptor PDBQT files using Meeko.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gemmi
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem


@dataclass
class TargetMetadata:
    """Scientific classification and metadata for a macromolecular target."""
    target_id: str
    pdb_id: str
    common_name: str
    uniprot_id: str
    biological_class: str
    ec_number: str | None
    resolution_angstrom: float
    experimental_method: str
    functional_role: str
    native_ligand_resname: str
    native_ligand_chain: str
    native_ligand_seqid: int
    box_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    receptor_pdbqt: str = ""
    native_kdh_pdb: str = ""
    native_kdh_pdbqt: str = ""


class ReceptorPreparer:
    """Handles parsing, cleaning, and PDBQT grid preparation across all 3 targets."""

    def __init__(self, receptors_root: Path | str) -> None:
        """Initialize ReceptorPreparer.

        Args:
            receptors_root: Directory containing 'raw/' subfolder and target output folders.
        """
        self.root: Path = Path(receptors_root)
        self.raw_dir: Path = self.root / "raw"
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_target_registry(cls) -> Dict[str, TargetMetadata]:
        """Return the strict academic target classification registry.

        Addresses the distinct biological classes of the 3 targets:
        - Pin1: Class 5 Isomerase (EC 5.2.1.8)
        - Transthyretin: Non-enzymatic plasma transport protein (no EC)
        - SIRT6: Class 3 Hydrolase (EC 3.5.1.-)
        """
        return {
            "3OOB_pin1": TargetMetadata(
                target_id="3OOB_pin1",
                pdb_id="3OOB",
                common_name="Pin1 (Peptidyl-prolyl cis-trans isomerase)",
                uniprot_id="Q13526",
                biological_class="Isomerase (EC 5.2.1.8)",
                ec_number="5.2.1.8",
                resolution_angstrom=1.89,
                experimental_method="X-ray Diffraction",
                functional_role="Oncogenic signaling isomerase; competitive active-site inhibition (Lys63/Arg69 contacts)",
                native_ligand_resname="KDH",
                native_ligand_chain="A",
                native_ligand_seqid=402,
            ),
            "3NG5_ttr": TargetMetadata(
                target_id="3NG5_ttr",
                pdb_id="3NG5",
                common_name="Transthyretin V30M (TTR)",
                uniprot_id="P02766",
                biological_class="Transport / Carrier Protein (Non-enzymatic)",
                ec_number=None,
                resolution_angstrom=1.70,
                experimental_method="X-ray Diffraction",
                functional_role="Plasma amyloid transport tetramer; kinetic quaternary stabilization against dissociation",
                native_ligand_resname="KDH",
                native_ligand_chain="A",
                native_ligand_seqid=128,
            ),
            "8ZVP_sirt6": TargetMetadata(
                target_id="8ZVP_sirt6",
                pdb_id="8ZVP",
                common_name="Sirtuin 6 (SIRT6 Deacylase)",
                uniprot_id="Q8N6T7",
                biological_class="Hydrolase (EC 3.5.1.- / EC 2.3.1.-)",
                ec_number="3.5.1.-",
                resolution_angstrom=2.50,
                experimental_method="X-ray Diffraction",
                functional_role="Chromatin modifier / longevity deacetylase; allosteric pocket modulation with ADP-ribose",
                native_ligand_resname="KDH",
                native_ligand_chain="A",
                native_ligand_seqid=403,
            ),
        }

    def _extract_and_clean_structures(
        self, raw_path: Path, meta: TargetMetadata, target_dir: Path
    ) -> Tuple[Path, Path]:
        """Extract native ligand coordinates and clean protein polymer using gemmi.

        Args:
            raw_path: Path to raw .cif or .pdb file.
            meta: Target metadata containing ligand chain and seqid.
            target_dir: Destination directory.

        Returns:
            Tuple of (clean_protein_pdb_path, native_ligand_pdb_path).
        """
        # [PACKAGE: gemmi] WHY: Fast, robust parsing of macromolecular mmCIF and PDB files without truncation. WHAT: gemmi.read_structure. HOW: Filter models, chains, residues.
        st = gemmi.read_structure(str(raw_path))
        st.remove_waters()

        kdh_st = gemmi.Structure()
        kdh_model = gemmi.Model("1")
        kdh_chain = gemmi.Chain(meta.native_ligand_chain)

        prot_st = gemmi.Structure()
        prot_model = gemmi.Model("1")

        ignored_hetero = {
            "HOH", "SO4", "PE4", "AR6", "EDO", "GOL", "DMS", "ACT", "ZN", "CL", "NA"
        }

        for chain in st[0]:
            prot_chain = gemmi.Chain(chain.name)
            for res in chain:
                if (
                    res.name == meta.native_ligand_resname
                    and chain.name == meta.native_ligand_chain
                    and str(res.seqid.num) == str(meta.native_ligand_seqid)
                ):
                    kdh_chain.add_residue(res)
                elif res.name != meta.native_ligand_resname and res.name not in ignored_hetero:
                    prot_chain.add_residue(res)
            if len(prot_chain) > 0:
                prot_model.add_chain(prot_chain)

        kdh_model.add_chain(kdh_chain)
        kdh_st.add_model(kdh_model)
        prot_st.add_model(prot_model)

        clean_prot_path = target_dir / "receptor_clean.pdb"
        native_kdh_path = target_dir / "native_kdh.pdb"

        prot_st.write_pdb(str(clean_prot_path))
        kdh_st.write_pdb(str(native_kdh_path))

        return clean_prot_path, native_kdh_path

    def _convert_native_ligand_to_pdbqt(self, native_pdb: Path, output_pdbqt: Path) -> None:
        """Convert native extracted ligand to PDBQT for re-docking control."""
        # [PACKAGE: rdkit.Chem] WHY: Read 3D coordinates of crystal ligand. WHAT: MolFromPDBFile. HOW: AddHs and parameterize with Meeko.
        mol = Chem.MolFromPDBFile(str(native_pdb), removeHs=False, sanitize=False)
        if mol is None:
            return

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        try:
            mol = Chem.AddHs(mol, addCoords=True)
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if mol_setups:
                pdbqt_res = PDBQTWriterLegacy.write_string(mol_setups[0])
                pdbqt_str = pdbqt_res[0] if isinstance(pdbqt_res, tuple) else pdbqt_res
                output_pdbqt.write_text(pdbqt_str, encoding="utf-8")
        except Exception:
            # Fallback if PDB coordinates have valence ambiguity
            pass

    def _prepare_receptor_pdbqt(
        self, clean_prot_pdb: Path, native_kdh_pdb: Path, target_dir: Path
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Run mk_prepare_receptor.py to generate receptor PDBQT and Vina grid envelope."""
        # [PACKAGE: meeko.cli] WHY: Standardized AutoDock Vina receptor parameterization with Gasteiger charges. WHAT: mk_prepare_receptor.py. HOW: Subprocess execution.
        out_pdbqt = target_dir / "receptor.pdbqt"
        vina_box_txt = target_dir / "vina_box.txt"

        if not (out_pdbqt.exists() and vina_box_txt.exists() and out_pdbqt.stat().st_size > 0):
            python_bin = sys.executable
            mk_script = Path(python_bin).parent / "mk_prepare_receptor.py"

            cmd = [
                str(mk_script),
                "--read_pdb", str(clean_prot_pdb),
                "--box_enveloping", str(native_kdh_pdb),
                "--padding", "8.0",
                "--default_altloc", "A",
                "-x",
                "--output_basename", str(target_dir / "receptor"),
                "-p", str(out_pdbqt),
                "-v", str(vina_box_txt),
            ]

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not out_pdbqt.exists() or not vina_box_txt.exists():
                raise RuntimeError(
                    f"mk_prepare_receptor failed for {clean_prot_pdb.name}: {res.stderr}"
                )

        # Parse box coordinates from vina_box.txt
        center: Dict[str, float] = {}
        size: Dict[str, float] = {}
        for line in vina_box_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line:
                k, v = [x.strip() for x in line.split("=")]
                if k.startswith("center_"):
                    center[k.split("_")[1]] = float(v)
                elif k.startswith("size_"):
                    size[k.split("_")[1]] = float(v)

        center_tuple = (center.get("x", 0.0), center.get("y", 0.0), center.get("z", 0.0))
        size_tuple = (size.get("x", 20.0), size.get("y", 20.0), size.get("z", 20.0))

        return center_tuple, size_tuple

    def prepare_all_targets(self) -> Dict[str, TargetMetadata]:
        """Prepare all 3 target receptors and generate manifest metadata."""
        registry = self.get_target_registry()
        prepared_targets: Dict[str, TargetMetadata] = {}

        raw_files = {
            "3OOB_pin1": self.raw_dir / "3OOB.cif",
            "3NG5_ttr": self.raw_dir / "3NG5.cif",
            "8ZVP_sirt6": self.raw_dir / "8ZVP.pdb",
        }

        for target_id, meta in registry.items():
            raw_path = raw_files.get(target_id)
            if not raw_path or not raw_path.exists():
                raise FileNotFoundError(f"Raw structure missing for {target_id}: {raw_path}")

            target_dir = self.root / target_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # 1. Clean structures and extract native KDH
            clean_prot, native_kdh = self._extract_and_clean_structures(
                raw_path, meta, target_dir
            )

            # 2. Convert native ligand to PDBQT
            native_pdbqt = target_dir / "native_kdh.pdbqt"
            self._convert_native_ligand_to_pdbqt(native_kdh, native_pdbqt)

            # 3. Generate receptor PDBQT and grid envelope
            center, size = self._prepare_receptor_pdbqt(clean_prot, native_kdh, target_dir)

            meta.box_center = center
            meta.box_size = size
            meta.receptor_pdbqt = str(target_dir / "receptor.pdbqt")
            meta.native_kdh_pdb = str(native_kdh)
            meta.native_kdh_pdbqt = str(native_pdbqt) if native_pdbqt.exists() else ""

            # Save individual target metadata JSON
            meta_json = target_dir / "target_info.json"
            meta_json.write_text(json.dumps(asdict(meta), indent=4), encoding="utf-8")

            prepared_targets[target_id] = meta

        # Save collective targets summary
        summary_path = self.root / "targets_summary.json"
        summary_dict = {tid: asdict(m) for tid, m in prepared_targets.items()}
        summary_path.write_text(json.dumps(summary_dict, indent=4), encoding="utf-8")

        return prepared_targets
