from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from rdkit import Chem


class ConformerIOManager:
    """Manages input/output operations, molecular file serialization, and scorecard persistence.

    Specialized in producing multi-model SDF and PDB structures tailored for direct interactive
    inspection in VS Code molecular visualization extensions (e.g. Mol*, NGL Viewer, 3Dmol.js).
    """

    def __init__(self, base_output_dir: Path | str) -> None:
        self.output_dir = Path(base_output_dir).resolve()
        self.models_dir = self.output_dir / "models_3d"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def export_sdf(
        self,
        mol: Chem.Mol,
        conf_ids: Sequence[int],
        energies: Mapping[int, float],
        rg_values: Mapping[int, float],
        filename: str,
    ) -> Path:
        """Export curated conformers to a multi-model SDF file with embedded energetic properties.

        Args:
            mol: RDKit molecule holding 3D conformers.
            conf_ids: Sequence of selected conformer IDs to export in order.
            energies: Mapping of conf_id to calculated potential energy in kcal/mol.
            rg_values: Mapping of conf_id to heavy-atom radius of gyration in Angstroms.
            filename: Destination filename (e.g., 'egcg_conformers_mmff94s_pocket.sdf').

        Returns:
            Path to the written SDF file.
        """
        target_path = self.models_dir / filename
        min_e = min(energies.values()) if energies else 0.0

        with Chem.SDWriter(str(target_path)) as writer:
            for rank, cid in enumerate(conf_ids, start=1):
                conf_mol = Chem.Mol(mol)
                conf_mol.SetProp("_Name", f"Conformer_{rank}_CID_{cid}")
                conf_mol.SetIntProp("Conformer_Rank", rank)
                conf_mol.SetIntProp("Conformer_ID", cid)

                energy = float(energies.get(cid, 0.0))
                delta_e = float(energy - min_e)
                rg = float(rg_values.get(cid, 0.0))

                conf_mol.SetDoubleProp("Energy_kcal_mol", energy)
                conf_mol.SetDoubleProp("DeltaE_kcal_mol", delta_e)
                conf_mol.SetDoubleProp("Radius_of_Gyration_A", rg)
                writer.write(conf_mol, confId=cid)

        return target_path

    def export_pdb(
        self,
        mol: Chem.Mol,
        conf_ids: Sequence[int],
        filename: str,
    ) -> Path:
        """Export curated conformers to a multi-model PDB file universally readable by 3D viewers.

        Each conformer is wrapped in standard MODEL / ENDMDL blocks, allowing interactive
        browsing between conformational basins in VS Code extensions.

        Args:
            mol: RDKit molecule holding 3D conformers.
            conf_ids: Sequence of selected conformer IDs.
            filename: Destination filename (e.g., 'egcg_conformers_mmff94s_pocket.pdb').

        Returns:
            Path to the written PDB file.
        """
        target_path = self.models_dir / filename

        with open(target_path, "w", encoding="utf-8") as f:
            for rank, cid in enumerate(conf_ids, start=1):
                f.write(f"MODEL     {rank:4d}\n")
                pdb_block = Chem.MolToPDBBlock(mol, confId=cid)
                atom_lines = [
                    line
                    for line in pdb_block.splitlines()
                    if not line.startswith(("MODEL", "ENDMDL", "END"))
                ]
                f.write("\n".join(atom_lines) + "\n")
                f.write("ENDMDL\n")
            f.write("END\n")

        return target_path

    def export_scorecard(
        self,
        summary_rows: Sequence[Mapping[str, object]],
        filename: str = "conformer_convergence_scorecard.csv",
    ) -> Path:
        """Export convergence summary metrics to CSV format.

        Args:
            summary_rows: Sequence of dictionary records.
            filename: Target CSV filename.

        Returns:
            Path to the saved CSV file.
        """
        target_path = self.output_dir / filename
        df = pd.DataFrame(summary_rows)
        df.to_csv(target_path, index=False)
        return target_path
