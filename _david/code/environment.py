from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, ParetoTanimotoDistance, WeightedSum
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from _david.code.utils import FallbackMonitor, get_safe_parallel_jobs

DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)

class PrepareMolecule:

    def __init__(
        self,
        num_conformers: int = 100,
        random_seed: int = 42,
        min_branch_size: int = 2,
        max_heteroatom_chain: int = 2,
        include_unsaturated: bool = False,
    ) -> None:
        self.num_conformers = num_conformers
        self.random_seed = random_seed
        self.min_branch_size = min_branch_size
        self.max_heteroatom_chain = max_heteroatom_chain
        self.include_unsaturated = include_unsaturated

    # [PACKAGE: rdkit.Chem] WHY: Multi-record MDL SD format parsing. WHAT: Conformer assembly. HOW: SDMolSupplier.
    def load_sdf_ensemble(self, path: Path | str) -> tuple[Chem.Mol, bool]:
        # 1. Multi-Record Conformer Extraction
        mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m]
        if not mols:
            raise ValueError(f"No valid molecule in SDF: {path}")

        # 2. Conformer Stitching onto Base Frame
        base = mols[0]
        for extra in mols[1:]:
            base.AddConformer(extra.GetConformer(), assignId=True)

        # 3. 3D Dimensionality Verification
        return base, bool(base.GetNumConformers() and base.GetConformer().Is3D())


    def load_query_molecule(self, query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
        # 1. Direct RDKit Mol instance
        if isinstance(query, Chem.Mol):
            return query, bool(query.GetNumConformers() and query.GetConformer().Is3D())

        # 2. Fast Path: SMILES string (avoids OS filesystem syscalls & filename length limits)
        if isinstance(query, str) and not query.endswith((".sdf", ".sd", ".txt", ".smi")):
            mol = Chem.MolFromSmiles(query)
            if mol:
                return mol, False

        # 3. File Resolution (Path object or path string)
        try:
            path = Path(query)
            if path.is_file():
                if path.suffix.lower() in {".sdf", ".sd"}:
                    return self.load_sdf_ensemble(path)
                tokens = path.read_text(encoding="utf-8").split()
                if not tokens:
                    raise ValueError(f"Empty molecule file: {path}")
                mol = Chem.MolFromSmiles(tokens[0])
                if mol:
                    return mol, False
        except OSError:
            pass

        raise ValueError(f"Could not parse molecule query: {query!r}")


    @staticmethod
    def _calculate_adaptive_rmsd_cutoff(mol: Chem.Mol | None) -> float:
        """Calculate heavy-atom RMSD cutoff for Butina basin curation (Hawkins 2017).

        WHY:
        Rigid molecules have limited degrees of conformational freedom; an RMSD cutoff of
        1.00 Å prevents clustering distinct minimum-energy conformers into a single degenerate basin.
        Standard drug-like molecules occupy moderate conformational space; an RMSD cutoff of
        1.25 Å groups minor vibrational fluctuations while preserving pharmacophore poses.
        Highly flexible ligands have sprawling degrees of freedom where small dihedral variations
        expand RMSD across the molecular diameter; an RMSD cutoff of 1.50 Å prevents fragmentation
        into hundreds of redundant clusters.

        WHAT:
        - Rigid/compact (<= 2 rotatable bonds, <= 20 heavy atoms): 1.00 Å
        - Drug-like (<= 7 rotatable bonds, <= 45 heavy atoms): 1.25 Å
        - Flexible (> 7 rotatable bonds or > 45 heavy atoms): 1.50 Å
        - Empty/null molecule: 1.00 Å

        HOW:
        Removes explicit hydrogens before counting rotatable bonds and heavy atoms to ensure
        topological invariance regardless of whether hydrogens are implicit or explicit.
        """
        if mol is None:
            return 1.00
        # Enforce topological invariance against explicit hydrogens
        mol_no_h = Chem.RemoveHs(mol) if mol.GetNumAtoms() > mol.GetNumHeavyAtoms() else mol
        n_heavy = int(mol_no_h.GetNumHeavyAtoms())
        if n_heavy == 0:
            return 1.00
        n_rot = int(Lipinski.NumRotatableBonds(mol_no_h))
        if n_rot <= 2 and n_heavy <= 20:
            return 1.00
        if n_rot <= 7 and n_heavy <= 45:
            return 1.25
        return 1.50

    def get_adaptive_rmsd_cutoff(self, mol: Chem.Mol | None) -> float:
        """Delegate to _calculate_adaptive_rmsd_cutoff for backwards-compatibility."""
        return self._calculate_adaptive_rmsd_cutoff(mol)

    def generate_reference_conformers(
        self,
        query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
        output_path: Path | str | None = None,
        num_conformers: int = 100,  #! HYP could be bigger, like 200
        random_seed: int = 42,
    ) -> Path:
        output_path_real = Path(output_path) if output_path else Path("reference_3d.sdf")
        output_path_real.parent.mkdir(parents=True, exist_ok=True)

        # 1. Input Loading & 3D Passthrough
        mol, is_3d = self.load_query_molecule(query)
        if is_3d:
            with Chem.SDWriter(str(output_path_real)) as writer:
                for conf in mol.GetConformers():
                    writer.write(mol, confId=conf.GetId())
            return output_path_real

        # 2. ETKDGv3 Conformer Embedding
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = 0.20  # since is single ligand its quite detailed (RDKit papers use 0.35-0.5, 0.1-0.2 is normal thermal vibrations)

        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        )  # based on crystal predictions
        if not conf_ids:
            params.useRandomCoords = True  # based on random initial + gradient optimization
            conf_ids = list(
                AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
            )
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 embedding failed for: {query}")

        # 3. Force Field Minimization & Energy Profiling (MMFF94 with UFF fallback)
        # [PACKAGE: rdkit.Chem.AllChem] WHY: Multi-threaded batch force field minimization with UFF fallback for non-MMFF elements.
        results = []
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            FallbackMonitor.record_fallback(
                "FORCE_FIELD",
                "MMFF94 lacks parameters for query molecule; falling back to UFF.",
                {"query": str(query)},
            )
            results = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
        if not results:
            raise RuntimeError(f"Force field optimization failed for: {query}")

        energies = {cid: float(e) for cid, (_, e) in zip(conf_ids, results)}

        # 4. Energy Window Filtering
        # ~ [Hawkins, Physiological energy window, 2017] <= 8.0 kcal/mol window retains biologically accessible conformational basins
        min_e = min(energies.values())
        for cid, e in list(energies.items()):
            if (e - min_e) > 8.0:
                mol_h.RemoveConformer(cid)
                del energies[cid]

        # 5. Butina RMSD Basin Curation
        # ~ [Hawkins, Basin curation via Butina clustering, 2017] Adaptive cutoff isolates distinct conformational basins
        remaining_cids = list(energies.keys())
        if len(remaining_cids) > 1:
            # [PACKAGE: rdkit.ML.Cluster.Butina] WHY: Cluster conformational basins by heavy-atom RMSD
            cutoff = self._calculate_adaptive_rmsd_cutoff(mol_h)
            # [PACKAGE: rdkit.Chem.AllChem.GetConformerRMSMatrix] Heavy-atom RMSD (Z > 1) prevents rotating Hs from splitting basins
            heavy_atom_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
            rms_matrix = AllChem.GetConformerRMSMatrix(
                mol_h, atomIds=heavy_atom_ids, prealigned=False
            )
            clusters = Butina.ClusterData(
                rms_matrix, len(remaining_cids), distThresh=cutoff, isDistData=True
            )

            # Select lowest-energy conformer within each conformational basin
            curated = []
            for cluster in clusters:
                basin_cids = [remaining_cids[i] for i in cluster]
                curated.append(min(basin_cids, key=energies.get))
            curated.sort(key=energies.get)
        else:
            curated = remaining_cids

        # 6. Export SDF with Energetic Annotations
        with Chem.SDWriter(str(output_path_real)) as writer:
            for i, cid in enumerate(curated):
                mol_h.SetProp("_Name", f"Conformer_{i + 1}")
                mol_h.SetDoubleProp("Energy_MMFF_kcal_mol", energies[cid])
                mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
                writer.write(mol_h, confId=cid)

        return output_path_real

    def extract_seed_core_scaffold(
        self,
        query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
        output_path: Path | str | None = None,
        scaffold_type: str = "BemisMurcko",
        isomeric: bool = False,
        min_cluster_size: int = 2,
        max_heteroatom_chain: int = 2,
        min_branch_size: int | None = None,
        include_unsaturated: bool = False,
    ) -> str:
        """Extract canonical Bemis-Murcko framework preserving all rings and linkers.

        WHY:
        Standardizing on the Bemis & Murcko (1996) molecular framework preserves the core
        pharmacophore architecture (conjugated ring systems, lactones, esters, and linkers)
        without fragile ad-hoc bond-breaking heuristics or aromaticity-stripping kekulization.

        WHAT:
        Extracts MurckoScaffold.GetScaffoldForMol(mol_no_h). For molecules with rings,
        this extracts all rings and connecting linkers (yielding 25 heavy atoms for EGCG).
        For purely acyclic molecules (where MurckoScaffold yields 0 heavy atoms),
        it falls back cleanly to the sanitized molecular SMILES.

        HOW:
        Uses RDKit MurckoScaffold and Chem.SanitizeMol, writing optional TSV metadata.
        """
        # 1. Input Loading & Hydrogen Stripping
        mol, _ = self.load_query_molecule(query)
        mol_no_h = Chem.RemoveHs(mol)

        # 2. Canonical Bemis-Murcko Framework Extraction
        scaff_mol = MurckoScaffold.GetScaffoldForMol(mol_no_h)
        if scaff_mol and scaff_mol.GetNumHeavyAtoms() > 0:
            Chem.SanitizeMol(scaff_mol)
            scaff_smi = Chem.MolToSmiles(scaff_mol, isomericSmiles=isomeric)
        else:
            # Acyclic fallback: Preserve sanitized heavy-atom backbone
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)

        # 3. Optional Metadata & TSV Export
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            res_mol = Chem.MolFromSmiles(scaff_smi)
            n_heavy = res_mol.GetNumHeavyAtoms() if res_mol else 0
            n_rings = res_mol.GetRingInfo().NumRings() if res_mol else 0
            with open(target, "w", encoding="utf-8") as f:
                f.write("Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings\n")
                f.write(
                    f"{scaff_smi}\t{str(query)}\t{scaffold_type}\t{n_heavy}\t{n_rings}\n"
                )

        return scaff_smi


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
) -> RDKitROCSScorer:
    # 1. Reference Input Normalization
    ref_input = str(references) if isinstance(references, Path) else references

    # 2. Conformer Generator Configuration
    conf_gen = RDKitConformerGenerator(
        max_conformers=30,  #! HYP could be bigger, like 50
        max_heavy_atoms=55,
        max_rotatable_bonds=25,
        timeout=60,  #! HYP could be bigger, like 60
    )

    # 3. Multi-Threaded ROCS Alignment Construction
    return RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        color_ff="oe_mills_dean",  # standard
        n_jobs=get_safe_parallel_jobs(),
    )

def create_environment(
    ref_sdf: str | Path,
    epsilon: float = 0.2,
) -> tuple[DrugExEnvironment, float]:
    # 1. Primary 3D ROCS Shape/Color Alignment Objective
    rocs_scorer = create_rocs_scorer(ref_sdf)
    rocs_scorer.modifier = ClippedScore(
        lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0
    )

    # 2. Secondary Synthesizability Feasibility Floor (SA Score)
    sa_scorer = Property(
        prop="SA",
        modifier=ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0),
    )

    # 3. Pareto Multi-Objective DrugEx Environment
    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    thresholds: list[float] = [0.25, 0.25]

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=ParetoCrowdingDistance(),
    )
    return env, epsilon
