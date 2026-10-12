from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import WeightedSum
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
ROCS_TIMEOUT: int = 30


def load_query_molecule(query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
    # query: Chem.Mol (with 3D conformers) | Path / file (.sdf, .sd, .txt) | str (SMILES)
    if isinstance(query, Chem.Mol):
        return query, bool(query.GetNumConformers() and query.GetConformer().Is3D())

    try:
        path = Path(query)
        if path.is_file():
            if path.suffix.lower() in {".sdf", ".sd"}:
                mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m]
                if not mols:
                    raise ValueError(f"No valid molecule in SDF: {path}")
                mol = mols[0]
                for extra in mols[1:]:
                    mol.AddConformer(extra.GetConformer(), assignId=True)
                return mol, bool(mol.GetNumConformers() and mol.GetConformer().Is3D())

            content = path.read_text(encoding="utf-8").split()
            if not content:
                raise ValueError(f"Empty SMILES file: {path}")
            query = content[0]
    except OSError:
        pass

    mol = Chem.MolFromSmiles(str(query).strip())
    if mol is None:
        raise ValueError(f"Could not parse query molecule: {query!r}")

    return mol, False


def get_adaptive_rmsd_cutoff(mol: Chem.Mol) -> float:
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return 1.00

    # [PACKAGE: rdkit.Chem.Lipinski] WHY: Quantify rotatable single bonds dictating conformational volume expansion.
    n_rot = int(
        Lipinski.NumRotatableBonds(mol)
    )  # non-terminal single bonds between heavy atoms
    n_heavy = int(mol.GetNumHeavyAtoms())

    # ~ [Ebejer et al., Conformer generation benchmarking, 2012] 1.00 Å for rigid/compact scaffolds (benzene, aspirin)
    if n_rot <= 2 and n_heavy <= 20:
        return 1.00

    # ~ [Hawkins, ROCS conformational ensemble benchmark, 2017] 1.25 Å optimal resolution for standard drug-like ligands (EGCG)
    #! HYP EGCG has N_rot=3 but N_heavy=33, so it must fall in this drug-like basin
    if n_rot <= 7 and n_heavy <= 45:
        return 1.25

    # ~ [Friedrich et al., Flexible ligand conformer benchmarking, 2017] Flexible chain expansion ceiling (capped at 1.50 Å)
    return 1.50


def generate_reference_conformers(
    query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
    num_conformers: int = 100,  #! HYP could be bigger, like 200
    random_seed: int = 42,
) -> Path:
    target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Input Loading & 3D Passthrough
    mol, is_3d = load_query_molecule(query)
    if is_3d:
        with Chem.SDWriter(str(target_path)) as writer:
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
        return target_path

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
    # [PACKAGE: rdkit.Chem.AllChem.MMFFOptimizeMoleculeConfs] Multi-threaded C++ batch minimization across all conformers
    results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
    if not results:  # fallback for unparameterized elements in periodic table
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
        cutoff = get_adaptive_rmsd_cutoff(mol_h)
        # [PACKAGE: rdkit.Chem.AllChem.GetConformerRMSMatrix] Heavy-atom RMSD (Z > 1) prevents rotating Hs from splitting basins
        heavy_atom_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
        rms_matrix = AllChem.GetConformerRMSMatrix(
            mol_h, atomIds=heavy_atom_ids, prealigned=False
        )
        clusters = Butina.ClusterData(
            rms_matrix, len(remaining_cids), distThresh=cutoff, isDistData=True
        )

        # Select thermodynamic minimum within each conformational basin
        curated = [
            min([remaining_cids[i] for i in cluster], key=lambda c: energies[c])
            for cluster in clusters
        ]
        curated.sort(key=lambda c: energies[c])
    else:
        curated = remaining_cids

    # 6. Export SDF with Energetic Annotations
    with Chem.SDWriter(str(target_path)) as writer:
        for i, cid in enumerate(curated):
            mol_h.SetProp("_Name", f"Conformer_{i + 1}")
            mol_h.SetDoubleProp("Energy_MMFF_kcal_mol", energies[cid])
            mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
            writer.write(mol_h, confId=cid)

    return target_path


def extract_seed_core_scaffold(
    query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
    scaffold_type: str = "core_ring_system",  # active default
    isomeric: bool = False,
) -> str:
    mol, _ = load_query_molecule(query)
    mol_no_h = Chem.RemoveHs(mol)

    # Option 1: Core Ring System (Privileged fused/spiro core; e.g. chroman 10 atoms for EGCG - ACTIVE)
    # [PACKAGE: rdkit.Chem.MolFragmentToSmiles] Extract SMILES of largest contiguous ring system directly from atom indices
    ring_info = mol_no_h.GetRingInfo()
    atom_rings = [set(r) for r in ring_info.AtomRings()]
    if atom_rings:
        ring_systems: list[set[int]] = []
        for ring in atom_rings:
            overlapping = [sys for sys in ring_systems if sys & ring]
            isolated = [sys for sys in ring_systems if not (sys & ring)]
            ring_systems = isolated + [ring.union(*overlapping)]

        core_atoms = max(ring_systems, key=len)
        scaff_smi = Chem.MolFragmentToSmiles(
            mol_no_h, atomsToUse=list(core_atoms), isomericSmiles=isomeric
        )
    else:
        scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)

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
    ref_input = str(references) if isinstance(references, Path) else references
    conf_gen = RDKitConformerGenerator(
        max_conformers=30,  #! HYP could be bigger, like 50
        max_heavy_atoms=55,
        max_rotatable_bonds=25,
        timeout=ROCS_TIMEOUT,  #! HYP could be bigger, like 60
    )
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
    rocs_scorer = create_rocs_scorer(ref_sdf)
    rocs_scorer.modifier = ClippedScore(
        lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0
    )

    sa_scorer = Property(
        prop="SA",
        modifier=ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0),
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    thresholds: list[float] = [0.25, 0.25]

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=WeightedSum(),
    )
    return env, epsilon
