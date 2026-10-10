"""Template-aligned 2D molecular depiction for consistent visual orientation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

# [PACKAGE: rdkit] WHY: 2D depiction alignment matching reference coordinates. WHAT: Cheminformatics depiction toolkit. HOW: Chem.rdDepictor.GenerateDepictionMatching2DStructure, Chem.rdDepictor.Compute2DCoords, Draw.MolsToGridImage.
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold


class AlignedDepicter:
    """Aligns 2D depiction coordinates of candidate molecules to a reference seed template.

    Ensures that core scaffolds (e.g. flavan-3-ol, benzopyran, core aromatics) are
    depicted with identical rotation and orientation across publication grids.
    """

    def __init__(
        self,
        reference: str | Path | Chem.Mol | None = None,
        core_smarts: str | None = None,
    ) -> None:
        """Initialize AlignedDepicter with query template or core SMARTS pattern."""
        self.ref_mol: Chem.Mol | None = None
        self.core_mol: Chem.Mol | None = None

        if reference is not None:
            if isinstance(reference, Chem.Mol):
                self.ref_mol = Chem.Mol(reference)
            elif isinstance(reference, Path) or (isinstance(reference, str) and Path(reference).is_file()):
                # Load from SDF or SMILES file
                p = Path(reference)
                if p.suffix.lower() in {".sdf", ".sd"}:
                    suppl = Chem.SDMolSupplier(str(p), removeHs=True)
                    for m in suppl:
                        if m is not None:
                            self.ref_mol = m
                            break
                else:
                    with open(p, encoding="utf-8") as f:
                        line = f.readline().strip()
                        smi = line.split()[0] if line else ""
                        self.ref_mol = Chem.MolFromSmiles(smi)
            elif isinstance(reference, str):
                self.ref_mol = Chem.MolFromSmiles(reference)

        if self.ref_mol is not None:
            # Generate reference 2D coordinates
            rdDepictor.Compute2DCoords(self.ref_mol)
            # Extract Murcko core scaffold as secondary alignment fallback
            try:
                core_scaff = MurckoScaffold.GetScaffoldForMol(self.ref_mol)
                if core_scaff is not None and core_scaff.GetNumAtoms() > 0:
                    rdDepictor.Compute2DCoords(core_scaff)
                    self.core_mol = core_scaff
            except Exception:
                self.core_mol = None

        if core_smarts:
            self.core_mol = Chem.MolFromSmarts(core_smarts)
            if self.core_mol is not None:
                rdDepictor.Compute2DCoords(self.core_mol)

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Align 2D coordinates of mol against reference or core scaffold."""
        mol_copy = Chem.Mol(mol)
        aligned = False

        # 1. Attempt depiction matching against intact reference
        if self.ref_mol is not None:
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, self.ref_mol)
                aligned = True
            except Exception:
                aligned = False

        # 2. Attempt depiction matching against core Murcko scaffold
        if not aligned and self.core_mol is not None:
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, self.core_mol)
                aligned = True
            except Exception:
                aligned = False

        # 3. Fallback to standard 2D layout
        if not aligned:
            rdDepictor.Compute2DCoords(mol_copy)

        return mol_copy

    def create_aligned_grid(
        self,
        mols: Sequence[Chem.Mol],
        legends: Sequence[str] | None = None,
        output_path: Path | str | None = None,
        mols_per_row: int = 3,
        sub_img_size: tuple[int, int] = (340, 260),
    ) -> Any:
        """Create publication-quality 2D structure grid with aligned orientations.

        Args:
            mols: List of RDKit molecules to draw.
            legends: Optional captions / property text under each molecule.
            output_path: Optional path to save image PNG.
            mols_per_row: Grid layout columns count (default 3).
            sub_img_size: Size per sub-image cell in pixels (width, height).

        Returns:
            PIL Image instance containing aligned molecular grid.
        """
        aligned_mols = [self.align_molecule(m) for m in mols]
        leg_list = list(legends) if legends is not None else ["" for _ in mols]

        img = Draw.MolsToGridImage(
            aligned_mols,
            molsPerRow=min(mols_per_row, len(aligned_mols)) if aligned_mols else 1,
            subImgSize=sub_img_size,
            legends=leg_list,
            useSVG=False,
        )

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(p))

        return img
