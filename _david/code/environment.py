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

DEFAULT_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
ROCS_TIMEOUT: int = 30


def load_query_molecule(query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
    # query: str (SMILES) | Path (.sdf, .sd, .txt) | Chem.Mol (with 3D conformers)
    if isinstance(query, Chem.Mol):
        return query, (query.GetNumConformers() > 0 and query.GetConformer().Is3D())

    query_path = Path(str(query))
    if query_path.is_file():
        if query_path.suffix.lower() in {".sdf", ".sd"}:
            suppl = Chem.SDMolSupplier(str(query_path), removeHs=False)
            mols = [m for m in suppl if m is not None]
            if not mols:
                raise ValueError(f"No valid molecule in SDF: {query_path}")
            base_mol = mols[0]
            for extra in mols[1:]:
                if extra.GetNumAtoms() == base_mol.GetNumAtoms() and extra.GetNumConformers() > 0:
                    base_mol.AddConformer(extra.GetConformer(), assignId=True)
            return base_mol, (base_mol.GetNumConformers() > 0 and base_mol.GetConformer().Is3D())
        else:
            with open(query_path, "r", encoding="utf-8") as f:
                smi = f.readline().strip().split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                return mol, False
            raise ValueError(f"Could not parse SMILES from {query_path}")

    mol = Chem.MolFromSmiles(str(query))
    if mol:
        return mol, False
    raise ValueError(f"Could not parse query: {query!r}")


def get_adaptive_rmsd_cutoff(mol: Chem.Mol) -> float:
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return 1.00

    # [PACKAGE: rdkit.Chem.Lipinski] WHY: Quantify rotatable single bonds dictating conformational volume expansion.
    n_rot = int(Lipinski.NumRotatableBonds(mol))  # non-terminal single bonds between heavy atoms
    n_heavy = int(mol.GetNumHeavyAtoms())

    #~ [Ebejer et al., Conformer generation benchmarking, 2012] 1.00 Å for rigid/compact scaffolds (benzene, aspirin)
    if n_rot <= 2 and n_heavy <= 20:
        return 1.00

    #~ [Hawkins, ROCS conformational ensemble benchmark, 2017] 1.25 Å optimal resolution for standard drug-like ligands (EGCG)
    #! HYP EGCG has N_rot=3 but N_heavy=33, so it must fall in this drug-like basin
    if n_rot <= 7 and n_heavy <= 45:
        return 1.25

    #~ [Friedrich et al., Flexible ligand conformer benchmarking, 2017] Flexible chain expansion ceiling (capped at 1.50 Å)
    return 1.50


def generate_reference_conformers(
    query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
    num_conformers: int = 100,  #! HYP could be bigger, like 200
    random_seed: int = 42,
) -> Path:
    target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    mol, is_3d = load_query_molecule(query)
    if is_3d:
        with Chem.SDWriter(str(target_path)) as writer:
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
        return target_path

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = 0.20  # since is single ligand its quite detailed (RDKit papers use 0.35-0.5, 0.1-0.2 is normal thermal vibrations)

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))  # based on crystal predictions
    if not conf_ids:
        params.useRandomCoords = True  # based on random initial + gradient optimization
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 embedding failed for: {query}")

    props = AllChem.MMFFGetMoleculeProperties(mol_h)  # tuned for organic, drug-like molecules (C, H, N, O, S, P, halogens)
    energies: list[tuple[float, int]] = []
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid)  # returns None if props is None or unparameterized
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)  # for all elements in periodic table
            
        if ff:
            ff.Minimize(maxIts=500)
            energies.append((float(ff.CalcEnergy()), cid))

    if not energies:
        raise RuntimeError(f"Could not calculate forcefield energies for: {query}")

    energies.sort(key=lambda x: x[0])
    min_e = energies[0][0]

    #~ [Hawkins, Physiological energy window, 2017] <= 8.0 kcal/mol window retains biologically accessible conformational basins
    accepted_cids = {cid for e, cid in energies if (e - min_e) <= 8.0}  # Whitelist: solves "None" issues
    for cid in conf_ids:
        if cid not in accepted_cids:
            mol_h.RemoveConformer(cid)

    remaining_cids = [conf.GetId() for conf in mol_h.GetConformers()]
    energy_map = {cid: float(e) for e, cid in energies}
    curated: list[int] = []

    #~ [Hawkins, Basin curation via Butina clustering, 2017] Adaptive cutoff isolates distinct conformational basins
    if len(remaining_cids) > 1:
        # [PACKAGE: rdkit.ML.Cluster.Butina] WHY: Cluster conformational basins by heavy-atom RMSD. WHAT: Unsupervised clustering.
        cutoff = get_adaptive_rmsd_cutoff(mol_h)
        # [PACKAGE: rdkit.Chem.AllChem.GetConformerRMSMatrix] Heavy-atom RMSD (Z > 1) prevents rotating Hs from splitting basins
        heavy_atom_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
        rms_matrix = AllChem.GetConformerRMSMatrix(mol_h, atomIds=heavy_atom_ids, prealigned=False)
        clusters = Butina.ClusterData(rms_matrix, len(remaining_cids), distThresh=cutoff, isDistData=True)
        for cluster in clusters:
            # Select the thermodynamic minimum within this conformational basin
            basin_cids = [remaining_cids[idx] for idx in cluster]
            best_cid = min(basin_cids, key=lambda cid: energy_map[cid])
            curated.append(best_cid)

        curated.sort(key=lambda cid: energy_map[cid])
    else:
        curated = remaining_cids

    with Chem.SDWriter(str(target_path)) as writer:
        for i, cid in enumerate(curated):
            mol_h.SetProp("_Name", f"Conformer_{i+1}")
            mol_h.SetDoubleProp("Energy_MMFF_kcal_mol", energy_map[cid])
            mol_h.SetDoubleProp("DeltaE_kcal_mol", energy_map[cid] - min_e)
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
        scaff_smi = Chem.MolFragmentToSmiles(mol_no_h, atomsToUse=list(core_atoms), isomericSmiles=isomeric)
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
            f.write(f"{scaff_smi}\t{str(query)}\t{scaffold_type}\t{n_heavy}\t{n_rings}\n")

    return scaff_smi


def create_rocs_scorer(references: str | Path | Chem.Mol | list[Chem.Mol] | list[str]) -> RDKitROCSScorer:
    ref_input = str(references) if isinstance(references, Path) else references
    conf_gen = RDKitConformerGenerator(
        max_conformers=22,
        max_heavy_atoms=55,
        max_rotatable_bonds=25,
        timeout=ROCS_TIMEOUT,
    )
    return RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        color_ff="oe_mills_dean",
        n_jobs=get_safe_parallel_jobs(),
    )


def create_environment(ref_sdf: str | Path) -> tuple[DrugExEnvironment, float]:
    rocs_scorer = create_rocs_scorer(ref_sdf)
    rocs_scorer.modifier = ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)

    sa_scorer = Property(
        prop="SA",
        modifier=ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0),
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    thresholds: list[float] = [0.25, 0.25]
    epsilon: float = 0.0

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=WeightedSum(),
    )
    return env, epsilon
