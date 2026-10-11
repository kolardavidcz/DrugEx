"""Environment construction and 3D reference conformer preparation."""
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
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from _david.code.utils import FallbackMonitor, get_safe_parallel_jobs

DEFAULT_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
ROCS_TIMEOUT: int = 30


def load_query_molecule(query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Load query molecule and detect whether 3D coordinates are present.
    
    query: str (SMILES) | Path (to .sdf .sd with H) or (text file with SMILES) | Chem.Mol (with 3d conformers)
    """
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


def generate_reference_conformers(
    query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
    num_conformers: int = 100,  #! HYP could be bigger, like 200
    random_seed: int = 42,
) -> Path:
    """Generate low-energy 3D conformer ensemble curated by Butina RMSD clustering."""
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
    params.pruneRmsThresh = 0.20    #?  since is single ligand its quite detailed (RDKit papers use 0.35-0.5, 0.1-0.2 is normal thermal vibrations)

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))  #?   based on crystal predictions
    if not conf_ids:
        params.useRandomCoords = True   #?  based on random initial + gradient optimization
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 embedding failed for: {query}")

    props = AllChem.MMFFGetMoleculeProperties(mol_h)    #?  tuned for organic, drug-like molecules (C, H, N, O, S, P, halogens)
    energies: list[tuple[float, int]] = []
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid) if props else None
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)  #?  for all elements in periodic table
            
        if ff:
            ff.Minimize(maxIts=500)
            energies.append((float(ff.CalcEnergy()), cid))

    energies.sort(key=lambda x: x[0])
    min_e = energies[0][0]
    
    # 1. Physiological energy window (Hawkins 2017: <= 8.0 kcal/mol)
    accepted_cids = {cid for e, cid in energies if (e - min_e) <= 8.0}
    for cid in conf_ids:
        if cid not in accepted_cids:
            mol_h.RemoveConformer(cid)

    remaining_cids = [conf.GetId() for conf in mol_h.GetConformers()]

    # 2. Geometric basin curation via Butina RMSD clustering (1.25 Å cutoff)
    if len(remaining_cids) > 1:
        # [PACKAGE: rdkit.ML.Cluster.Butina] WHY: Cluster conformational basins by heavy-atom RMSD. WHAT: Unsupervised clustering.
        rms_matrix = AllChem.GetConformerRMSMatrix(mol_h, prealigned=False)
        clusters = Butina.ClusterData(rms_matrix, len(remaining_cids), distThresh=1.25, isDistData=True)
        curated = [remaining_cids[c[0]] for c in clusters[:8]]
    else:
        curated = remaining_cids

    curated_set = set(curated)
    for cid in remaining_cids:
        if cid not in curated_set:
            mol_h.RemoveConformer(cid)

    # 3. Superimpose conformers onto central scaffold
    scaff = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(mol_h))
    match = mol_h.GetSubstructMatch(scaff) if scaff else ()
    atom_ids = list(match) if match else list(range(min(10, mol_h.GetNumAtoms())))
    try:
        AllChem.AlignMolConformers(mol_h, atomIds=atom_ids)
    except Exception:
        AllChem.AlignMolConformers(mol_h)

    with Chem.SDWriter(str(target_path)) as writer:
        for i, conf in enumerate(mol_h.GetConformers()):
            mol_h.SetProp("_Name", f"Conformer_{i+1}")
            writer.write(mol_h, confId=conf.GetId())

    return target_path


def extract_seed_core_scaffold(
    query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
) -> str:
    """Extract Murcko core ring framework of reference ligand as generative seed fragment."""
    mol, _ = load_query_molecule(query)
    mol_no_h = Chem.RemoveHs(mol)
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol_no_h)
        scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=False) if (scaff and scaff.GetNumHeavyAtoms() > 0) else Chem.MolToSmiles(mol_no_h, isomericSmiles=False)
    except Exception:
        scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)

    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("Scaffold\tQuery\n")
            f.write(f"{scaff_smi}\t{str(query)}\n")
    return scaff_smi


def create_rocs_scorer(references: str | Path | Chem.Mol | list[Chem.Mol] | list[str]) -> RDKitROCSScorer:
    """Construct native DrugEx RDKitROCSScorer with Mills-Dean color SMARTS."""
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
    """Constructs LBDD reinforcement learning environment with in-line level upgrades."""
    rocs_scorer = create_rocs_scorer(ref_sdf)
    rocs_scorer.modifier = ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)

    sa_scorer = Property(
        prop="SA",
        modifier=ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0),
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    thresholds: list[float] = [0.25, 0.25]
    epsilon: float = 0.0  # lvl1: 3D baseline (pure on-policy exploitation)
    # epsilon: float = 0.20  # lvl2: 20% prior mutation for scaffold hopping
    # qed_scorer = Property(prop="QED")  # lvl3: continuous drug-likeness positive feedback
    # scorers.append(qed_scorer)  # lvl3
    # thresholds.append(0.50)  # lvl3

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=WeightedSum(),
    )
    return env, epsilon
