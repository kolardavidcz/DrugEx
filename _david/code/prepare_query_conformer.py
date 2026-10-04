"""Prepares curated 3D reference conformer ensembles, Murcko core fragments, and dynamic envelopes."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# [PACKAGE: numpy] WHY: Median calculation for Radius of Gyration extended conformer curation. WHAT: Scientific computing library. HOW: np.median.
import numpy as np

# [PACKAGE: rdkit] WHY: 3D conformer embedding (ETKDGv3), force field relaxation (MMFF94), Radius of Gyration calculation, and Murcko scaffold extraction. WHAT: Comprehensive cheminformatics toolkit. HOW: rdkit.Chem AllChem, Descriptors3D, MurckoScaffold, Lipinski, Crippen.
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.Scaffolds import MurckoScaffold

# [PACKAGE: drugex] WHY: Native conformer generator, 3D ROCS scorer, and property modifiers. WHAT: Deep learning drug design framework. HOW: drugex.training.scorers.conformer_generators, drugex.training.scorers.rocs_rdkit, drugex.training.scorers.properties.
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

try:
    from dynamic_envelope import (
        DynamicPropertyEnvelope,
        compute_dynamic_envelope,
        create_dynamic_property_scorers,
    )
except (ImportError, ModuleNotFoundError):
    from _david.code.dynamic_envelope import (
        DynamicPropertyEnvelope,
        compute_dynamic_envelope,
        create_dynamic_property_scorers,
    )

try:
    from fallback_monitor import FallbackMonitor
except (ImportError, ModuleNotFoundError):
    from _david.code.fallback_monitor import FallbackMonitor

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
NUM_CONFORMERS: int = 50
RANDOM_SEED: int = 42
PRUNE_RMS_THRESH: float = 0.20
MAX_ENERGY_DELTA: float = 8.0

ROCS_MAX_CONFORMERS: int = 22
ROCS_MAX_HEAVY_ATOMS: int = 55
ROCS_MAX_ROTATABLE_BONDS: int = 25
ROCS_TIMEOUT: int = 30


def get_seed_hash(smiles: str, length: int = 8) -> str:
    """Compute deterministic short hash for query seed SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    can_smi = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else smiles.strip()
    return hashlib.sha256(can_smi.encode("utf-8")).hexdigest()[:length]


def load_query_molecule(query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Load query molecule and detect whether valid 3D coordinates are present.

    Parameters:
        query: SMILES string, Path to SDF/MOL/SMI file, or RDKit Mol object.

    Returns:
        tuple[Chem.Mol, bool]: The parsed RDKit Mol and a boolean flag indicating 3D conformation.
    """
    if isinstance(query, Chem.Mol):
        has_3d = query.GetNumConformers() > 0 and query.GetConformer().Is3D()
        return query, has_3d

    query_str: str = str(query)
    query_path: Path = Path(query_str)

    if query_path.is_file():
        # [PACKAGE: rdkit] WHY: Read 3D SDF structures and preserve coordinates. WHAT: SDMolSupplier reader. HOW: Chem.SDMolSupplier.
        if query_path.suffix.lower() in {".sdf", ".sd"}:
            suppl = Chem.SDMolSupplier(str(query_path), removeHs=False)
            loaded_mols: list[Chem.Mol] = [m for m in suppl if m is not None]
            if not loaded_mols:
                raise ValueError(f"No valid molecule could be read from SDF file: {query_path}")

            base_mol: Chem.Mol = loaded_mols[0]
            # Merge additional conformer records if multiple records exist in SDF
            for extra_mol in loaded_mols[1:]:
                if extra_mol.GetNumAtoms() == base_mol.GetNumAtoms() and extra_mol.GetNumConformers() > 0:
                    base_mol.AddConformer(extra_mol.GetConformer(), assignId=True)

            has_3d = base_mol.GetNumConformers() > 0 and base_mol.GetConformer().Is3D()
            return base_mol, has_3d
        else:
            with open(query_path, "r", encoding="utf-8") as f:
                first_line: str = f.readline().strip()
                smi: str = first_line.split()[0] if first_line else ""
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return mol, False
            raise ValueError(f"Could not parse SMILES from file: {query_path}")

    mol = Chem.MolFromSmiles(query_str)
    if mol is not None:
        return mol, False

    raise ValueError(f"Could not parse query molecule: {query!r}")


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    max_conformers: int = ROCS_MAX_CONFORMERS,
    max_heavy_atoms: int = ROCS_MAX_HEAVY_ATOMS,
    max_rotatable_bonds: int = ROCS_MAX_ROTATABLE_BONDS,
    timeout: int = ROCS_TIMEOUT,
    score_type: str = "TanimotoCombo",
    color_ff: str = "oe_mills_dean",
    n_jobs: int = -1,
    show_progress: bool = False,
) -> RDKitROCSScorer:
    """Construct native DrugEx RDKitROCSScorer with relaxed conformer limits and oe_mills_dean color SMARTS.

    Parameters:
        references: Reference SDF path or RDKit Mol object(s).
        max_conformers: Maximum conformers generated per probe molecule (default 22).
        max_heavy_atoms: Maximum heavy atom count threshold before skipping (relaxed to 55).
        max_rotatable_bonds: Maximum rotatable bond count threshold before skipping (relaxed to 25).
        timeout: Conformer generation timeout in seconds per molecule (default 30s).
        score_type: Alignment score type ('TanimotoCombo', 'shape', 'color').
        color_ff: Pharmacophore feature forcefield ('oe_mills_dean' or 'default').
        n_jobs: Multi-processing core count (-1 for all available cores via capped_n_jobs).
        show_progress: Whether to show progress bar during scoring.

    Returns:
        RDKitROCSScorer: Fully configured native DrugEx ROCS scorer instance.
    """
    conf_gen = RDKitConformerGenerator(
        max_conformers=max_conformers,
        max_heavy_atoms=max_heavy_atoms,
        max_rotatable_bonds=max_rotatable_bonds,
        timeout=timeout,
        show_progress=show_progress,
    )

    ref_arg: str | list[str] | Chem.Mol | list[Chem.Mol]
    if isinstance(references, Path):
        ref_arg = str(references)
    else:
        ref_arg = references  # type: ignore[assignment]

    return RDKitROCSScorer(
        conformer_generator=conf_gen,
        references=ref_arg,
        score_type=score_type,
        color_ff=color_ff,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


class QueryConformerPreparer:
    """Prepares curated 3D conformer ensembles, Murcko core fragments, and dynamic envelopes for reference ligands."""

    get_seed_hash = staticmethod(get_seed_hash)

    def __init__(self, references_dir: Path | str | None = None) -> None:
        """Initialize QueryConformerPreparer with reference output directory."""
        if references_dir is None:
            self.REFERENCES_DIR: Path = SCRIPT_DIR.parent / "references"
        else:
            self.REFERENCES_DIR = Path(references_dir)
        self.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_SDF: Path = self.REFERENCES_DIR / "my_molecule_3d.sdf"

    def generate_reference_sdf(
        self,
        query: str | Path | Chem.Mol = MY_MOLECULE,
        output_path: Path | None = None,
        num_conformers: int = NUM_CONFORMERS,
        prune_rms_thresh: float = PRUNE_RMS_THRESH,
        max_energy_delta: float = MAX_ENERGY_DELTA,
        curate_extended: bool = True,
        **kwargs: Any,
    ) -> Path:
        """Generate low-energy 3D conformer ensemble curated to prevent vacuum intramolecular collapse.

        If a 3D SDF is provided, its 3D Cartesian coordinates are preserved.
        If a SMILES or 2D molecule is provided, an ensemble is embedded using ETKDGv3
        with pruneRmsThresh=0.20 A, relaxed with MMFF94 within Delta E <= 8.0 kcal/mol,
        and curated by Radius of Gyration (Rg >= Rg,median) to retain extended outward-facing geometries.
        """
        # Backward compatibility for smiles parameter
        if "smiles" in kwargs:
            query = kwargs["smiles"]

        target_path: Path = output_path or self.OUTPUT_SDF
        target_path.parent.mkdir(parents=True, exist_ok=True)

        mol, is_3d = load_query_molecule(query)

        # 1. If authentic 3D coordinates provided, preserve them exactly
        if is_3d:
            writer = Chem.SDWriter(str(target_path))
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
            writer.close()
            return target_path

        # 2. Conformer ensemble generation for 2D / SMILES input
        mol_h: Chem.Mol = Chem.AddHs(mol)

        # [PACKAGE: rdkit] WHY: ETKDGv3 distance-geometry embedding with 0.20 A RMSD prune threshold. WHAT: Conformer generator parameters. HOW: AllChem.ETKDGv3.
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        params.pruneRmsThresh = prune_rms_thresh

        conf_ids: list[int] = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if len(conf_ids) == 0:
            FallbackMonitor.record_fallback(
                fallback_type="conformer_embedding_random_coords",
                message=f"Standard ETKDGv3 embedding yielded 0 conformers for query; falling back to useRandomCoords=True",
                context={"query": str(query), "num_conformers": num_conformers},
            )
            params.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
            if len(conf_ids) == 0:
                FallbackMonitor.record_fallback(
                    fallback_type="conformer_embedding_failure",
                    message=f"Conformer embedding failed completely for query: {query}",
                    context={"query": str(query)},
                )
                raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {query}")

        # [PACKAGE: rdkit] WHY: Force-field relaxation using MMFF94 (or UFF fallback if unparameterized). WHAT: Molecular mechanics force field. HOW: AllChem.MMFFGetMoleculeForceField, UFFOptimizeMolecule.
        energies: list[tuple[float, int]] = []
        for cid in conf_ids:
            props = AllChem.MMFFGetMoleculeProperties(mol_h)
            if props is None:
                FallbackMonitor.record_fallback(
                    fallback_type="forcefield_uff_fallback",
                    message="MMFF94 unparameterized for molecule; falling back to UFF force field",
                    context={"query": str(query), "conf_id": cid},
                )
                res = AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
                energy = float(res)
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid)
                if ff is not None:
                    ff.Minimize(maxIts=500)
                    energy = float(ff.CalcEnergy())
                else:
                    FallbackMonitor.record_fallback(
                        fallback_type="forcefield_uff_fallback",
                        message="MMFF forcefield initialization returned None; falling back to UFF",
                        context={"query": str(query), "conf_id": cid},
                    )
                    AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
                    energy = 0.0
            energies.append((energy, cid))

        energies.sort(key=lambda x: x[0])
        min_e: float = energies[0][0]
        accepted_cids: list[int] = [cid for e, cid in energies if (e - min_e) <= max_energy_delta]
        if not accepted_cids:
            accepted_cids = [energies[0][1]]

        # 3. Prevent vacuum intramolecular collapse via Radius of Gyration (Rg >= Rg,median)
        if curate_extended and len(accepted_cids) > 1:
            # [PACKAGE: rdkit] WHY: Measure 3D spatial extent of conformers to prevent vacuum globular collapse. WHAT: 3D descriptor. HOW: Descriptors3D.RadiusOfGyration.
            rgs = [(Descriptors3D.RadiusOfGyration(mol_h, confId=cid), cid) for cid in accepted_cids]
            rg_median: float = float(np.median([rg for rg, cid in rgs]))
            curated_cids: list[int] = [cid for rg, cid in rgs if rg >= rg_median]
            if not curated_cids:
                curated_cids = accepted_cids
        else:
            curated_cids = accepted_cids

        # [PACKAGE: rdkit] WHY: Write curated 3D conformers to multi-record SDF. WHAT: SDF writer. HOW: Chem.SDWriter.
        writer = Chem.SDWriter(str(target_path))
        for cid in curated_cids:
            writer.write(mol_h, confId=cid)
        writer.close()

        return target_path

    def extract_seed_core_scaffold(
        self,
        query: str | Path | Chem.Mol = MY_MOLECULE,
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> str:
        """Extract canonical Bemis-Murcko core ring framework of reference ligand as generative seed fragment.

        Preserves intact ring assemblies and linkers (e.g. flavan-3-ol / Murcko core for EGCG)
        while stripping terminal functional substituents.
        """
        if "smiles" in kwargs:
            query = kwargs["smiles"]

        mol, _ = load_query_molecule(query)
        mol_no_h: Chem.Mol = Chem.RemoveHs(mol)

        try:
            # [PACKAGE: rdkit] WHY: Extract intact Murcko core framework without side-chain decorations. WHAT: Bemis-Murcko scaffold algorithm. HOW: MurckoScaffold.GetScaffoldForMol.
            scaff = MurckoScaffold.GetScaffoldForMol(mol_no_h)
            if scaff is None or scaff.GetNumHeavyAtoms() == 0:
                scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)
            else:
                scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=False)
        except Exception:
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)

        target = output_path or (self.REFERENCES_DIR / "seed_core_scaffold.tsv")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("Scaffold\tQuery\n")
            f.write(f"{scaff_smi}\t{str(query)}\n")

        return scaff_smi

    def get_dynamic_envelope(
        self,
        query: str | Path | Chem.Mol = MY_MOLECULE,
    ) -> dict[str, dict[str, float]]:
        """Compute dynamic physicochemical envelope dictionary for reference ligand."""
        return compute_dynamic_envelope(query)

    def create_dynamic_property_scorers(
        self,
        query: str | Path | Chem.Mol = MY_MOLECULE,
    ) -> list[Property]:
        """Create DrugEx Property scorers configured with SmoothHump modifiers for reference ligand."""
        return create_dynamic_property_scorers(query)
