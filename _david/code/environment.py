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


    def get_adaptive_rmsd_cutoff(self, mol: Chem.Mol) -> float:
        return 1.25
        # # 1. Null / Heavy Atom Guard
        # if mol is None or mol.GetNumHeavyAtoms() == 0:
        #     return 1.00

        # # 2. Rotatable Bond & Heavy Atom Topology Profiling
        # # [PACKAGE: rdkit.Chem.Lipinski] WHY: Quantify rotatable single bonds dictating conformational volume expansion.
        # n_rot = int(Lipinski.NumRotatableBonds(mol))
        # n_heavy = int(mol.GetNumHeavyAtoms())

        # # 3. Rigid & Compact Scaffolds (<= 2 rotatable bonds, <= 20 heavy atoms; e.g. benzene, aspirin)
        # # ~ [Ebejer et al., Conformer generation benchmarking, 2012]
        # if n_rot <= 2 and n_heavy <= 20:
        #     return 1.00

        # # 4. Standard Drug-Like Ligands (<= 7 rotatable bonds, <= 45 heavy atoms; e.g. EGCG)
        # # ~ [Hawkins, ROCS conformational ensemble benchmark, 2017]
        # #! HYP EGCG has N_rot=3 but N_heavy=33, so it must fall in this drug-like basin
        # if n_rot <= 7 and n_heavy <= 45:
        #     return 1.25

        # # 5. Flexible Chain Expansion Ceiling (capped at 1.50 Å)
        # # ~ [Friedrich et al., Flexible ligand conformer benchmarking, 2017]
        # return 1.50


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
            if any(status == -1 for status, _ in results):
                results = []
        if not results and AllChem.UFFHasAllMoleculeParams(mol_h):
            results = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
            if any(status == -1 for status, _ in results):
                results = []
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
            cutoff = self.get_adaptive_rmsd_cutoff(mol_h)
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


    def _is_passable_bridge(
        self,
        mol: Chem.Mol,
        rw: Chem.RWMol,
        bond: Chem.Bond,
        min_branch_size: int,
    ) -> bool:
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        zu, zv = mol.GetAtomWithIdx(u).GetAtomicNum(), mol.GetAtomWithIdx(v).GetAtomicNum()
        if zu == 6 and zv == 6:
            return True

        # Temporarily cut bond to profile connected component sizes on each side
        rw.RemoveBond(u, v)
        frags = Chem.GetMolFrags(rw, asMols=False)
        fu = next(f for f in frags if u in f)
        fv = next(f for f in frags if v in f)

        u_ok = (zu == 6) or (len(fu) >= min_branch_size and any(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in fu))
        v_ok = (zv == 6) or (len(fv) >= min_branch_size and any(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in fv))

        if u_ok and v_ok:
            rw.AddBond(u, v, bond.GetBondType())
            return True
        return False

    def _has_valid_hetero_chain(
        self,
        mol: Chem.Mol,
        path: list[int],
        max_heteroatom_chain: int,
    ) -> bool:
        consec = max_consec = 0
        for idx in path:
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        return max_consec <= max_heteroatom_chain

    def _append_unsaturated_side_groups(
        self,
        mol: Chem.Mol,
        path: list[int],
    ) -> list[int]:
        scaffold = set(path)
        for idx in path:
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                nbr_idx = b.GetOtherAtomIdx(idx)
                if b.GetBondTypeAsDouble() > 1.0 and nbr_idx not in scaffold:
                    if mol.GetAtomWithIdx(nbr_idx).GetAtomicNum() in (7, 8, 16):
                        scaffold.add(nbr_idx)
        return sorted(scaffold)

    # [PACKAGE: rdkit.Chem.rdmolops] WHY: Continuous heavy-atom backbone with passable heteroatom bridges and unsaturated side groups. WHAT: Topological diameter. HOW: RDKit RWMol bond pruning, distance matrix & GetShortestPath.
    def get_acyclic_backbone_scaffold(
        self,
        mol: Chem.Mol,
        min_branch_size: int = 2,
        max_heteroatom_chain: int = 2,
        include_unsaturated: bool = False,
    ) -> list[int]:
        if not mol or mol.GetNumHeavyAtoms() == 0:
            return []

        # 1. Prune Impassable Heteroatom Bonds
        rw = Chem.RWMol(mol)
        for b in list(mol.GetBonds()):
            self._is_passable_bridge(mol, rw, b, min_branch_size)

        # 2. Extract Longest Passable Path Obeying Max Heteroatom Chain Length
        D = Chem.GetDistanceMatrix(rw)
        D[D > 1000] = 0
        pairs = sorted(np.argwhere(D > 0), key=lambda p: D[p[0], p[1]], reverse=True)

        longest: list[int] = []
        for i, j in pairs:
            path = list(Chem.rdmolops.GetShortestPath(rw, int(i), int(j)))
            if len(path) <= len(longest):
                break
            if self._has_valid_hetero_chain(mol, path, max_heteroatom_chain):
                longest = path
                break

        if not longest:
            longest = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6] or [a.GetIdx() for a in mol.GetAtoms()]

        # 3. Optional Unsaturated Appendages Preservation (=O, =N, =S)
        if include_unsaturated:
            return self._append_unsaturated_side_groups(mol, longest)
        return sorted(longest)

    def extract_seed_core_scaffold(
        self,
        query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
        output_path: Path | str | None = None,
        scaffold_type: str = "core_ring_system",  # active default
        isomeric: bool = False,
        min_branch_size: int = 2,
        max_heteroatom_chain: int = 2,
        include_unsaturated: bool = False,
    ) -> str:
        # 1. Input Loading & Hydrogen Stripping
        mol, _ = self.load_query_molecule(query)
        mol_no_h = Chem.RemoveHs(mol)

        # 2. Ring System Detection & Union
        # [PACKAGE: rdkit.Chem.MolFragmentToSmiles] Extract SMILES of largest contiguous ring system directly from atom indices
        ring_info = mol_no_h.GetRingInfo()
        atom_rings = [set(r) for r in ring_info.AtomRings()]
        if atom_rings:
            ring_systems: list[set[int]] = []
            for ring in atom_rings:
                overlapping = [sys for sys in ring_systems if sys & ring]
                isolated = [sys for sys in ring_systems if not (sys & ring)]
                ring_systems = isolated + [ring.union(*overlapping)]

            # 3. Subgraph Extraction for Largest Contiguous Core
            core_atoms = max(ring_systems, key=len)
            scaff_smi = Chem.MolFragmentToSmiles(
                mol_no_h, atomsToUse=list(core_atoms), isomericSmiles=isomeric
            )
        else:
            # 3. Acyclic Fallback: Continuous Heavy-Atom Backbone Scaffold with Passable Linkers
            chain_atoms = self.get_acyclic_backbone_scaffold(
                mol_no_h,
                min_branch_size=min_branch_size,
                max_heteroatom_chain=max_heteroatom_chain,
                include_unsaturated=include_unsaturated,
            )
            scaff_smi = Chem.MolFragmentToSmiles(
                mol_no_h, atomsToUse=chain_atoms, isomericSmiles=isomeric
            )

        # 4. Bulletproof Sanitization Pass
        scaff_mol = Chem.MolFromSmiles(scaff_smi)
        if scaff_mol is not None:
            Chem.SanitizeMol(scaff_mol)
            scaff_smi = Chem.MolToSmiles(scaff_mol, isomericSmiles=isomeric)

        # Option 2: Bemis-Murcko Framework (Ring systems + linkers; 25 heavy atoms for EGCG)
        # [PACKAGE: rdkit.Chem.Scaffolds.MurckoScaffold]
        # scaff = MurckoScaffold.GetScaffoldForMol(mol_no_h)
        # scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=isomeric) if (scaff and scaff.GetNumHeavyAtoms() > 0) else Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)

        # Option 3: Minimal Elementary Ring (Single smallest ring, e.g. benzene 6 atoms)
        # [PACKAGE: rdkit.Chem.MolFragmentToSmiles]
        # min_ring = min(atom_rings, key=len) if atom_rings else set()
        # scaff_smi = Chem.MolFragmentToSmiles(mol_no_h, atomsToUse=list(min_ring), isomericSmiles=isomeric) if min_ring else Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)

        # Option 4: SequenceRNN (Unconditional Ex Nihilo / 0 atoms)
        # SequenceRNN does not require an input scaffold seed; it samples de novo from <START> or can take leads from a previous GraphTransformer run.
        # scaff_smi = ""

        # 5. Optional Metadata & TSV Export
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            scaff_mol = Chem.MolFromSmiles(scaff_smi)
            n_heavy = scaff_mol.GetNumHeavyAtoms() if scaff_mol else 0
            n_rings = scaff_mol.GetRingInfo().NumRings() if scaff_mol else 0
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
