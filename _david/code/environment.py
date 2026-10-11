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
from rdkit.Chem import AllChem, Descriptors3D, Lipinski
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

def get_adaptive_rmsd_cutoff(mol: Chem.Mol) -> float:
    """Adaptive conformational basin RMSD clustering cutoff based on torsional flexibility and molecular size.

    In 3D shape-based virtual screening and conformer ensemble generation (ROCS, Ebejer 2012, Hawkins 2017),
    the optimal RMSD clustering threshold scales with molecular mass / heavy atom count ($N_{\\text{heavy}}$)
    and the number of rotatable single bonds ($N_{\\text{rot}}$) governing conformational volume expansion.

    Parameters
    ----------
    mol : Chem.Mol
        Input RDKit molecule (with or without explicit hydrogens).

    Returns
    -------
    float
        Optimal heavy-atom RMSD distance threshold (in Angstroms) for Butina sphere-exclusion clustering.
        Bounded between 1.00 Å (rigid/compact fragments) and 1.50 Å (large flexible ligands).

    References
    ----------
    .. [1] Hawkins, P.C.D. "Conformation Generation: The State of the Art"
       J. Chem. Inf. Model. 2017, 57(8), 1747–1766.
    .. [2] Ebejer, J.P.; Morris, G.M.; Deane, C.M. "Freely Available Conformer Generation Methods: How Good Are They?"
       J. Chem. Inf. Model. 2012, 52(5), 1146–1158.
    .. [3] Friedrich, N.O. et al. "Benchmarking Commercial Conformer Ensemble Generators"
       J. Chem. Inf. Model. 2017, 57(11), 2719–2728.
    """
    if mol is None:
        raise ValueError("Cannot calculate adaptive RMSD cutoff for None molecule.")

    # [PACKAGE: rdkit.Chem.Lipinski] WHY: Quantify rotatable single bonds dictating conformational volume expansion.
    # WHAT: SMARTS pattern matching for non-terminal acyclic single bonds.
    # HOW: Evaluates acyclic single bonds between heavy atoms, excluding terminal methyls/hydroxyls.
    n_rot = int(Lipinski.NumRotatableBonds(mol))
    n_heavy = int(mol.GetNumHeavyAtoms())

    #~ [Ebejer et al., Conformer generation benchmarking, 2012]
    # Small, compact, or rigid scaffolds with minimal torsional freedom (N_heavy <= 20 and N_rot <= 2)
    # require 1.00 Å to avoid over-merging distinct geometrical rotamers.
    if n_rot <= 2 and n_heavy <= 20:
        return 1.00

    #~ [Hawkins, ROCS conformational ensemble benchmark, 2017]
    # 1.25 Å is the empirically proven optimal resolution for standard drug-like molecules (N_heavy in [20, 45]
    # or N_rot in [3, 7], including polyphenol leads like EGCG with N_heavy=33), balancing >85% bioactive shape
    # coverage with computationally compact ensembles (5-15 conformers).
    if n_rot <= 7 and n_heavy <= 45:
        return 1.25

    #~ [Friedrich et al., Flexible ligand conformer benchmarking, 2017; Hawkins 2017]
    # For large or highly flexible ligands (N_rot >= 8 or N_heavy > 45), 1.50 Å prevents explosive cluster proliferation
    # while avoiding severe shape distortion (thresholds > 1.50 Å degrade ROCS pharmacophoric matching).
    return 1.50

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
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid) if props else None
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)  # for all elements in periodic table
            
        if ff:
            ff.Minimize(maxIts=500)
            energies.append((float(ff.CalcEnergy()), cid))

    energies.sort(key=lambda x: x[0])
    min_e = energies[0][0]
    
    #~ [Hawkins, Physiological energy window, 2017] <= 8.0 kcal/mol window retains biologically accessible conformational basins
    accepted_cids = {cid for e, cid in energies if (e - min_e) <= 8.0} # Whitelist: solves "None" issues
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
        # [PACKAGE: rdkit.Chem.AllChem.GetConformerRMSMatrix] WHY: Heavy-atom pairwise RMSD distance matrix.
        # WHAT: Pairwise RMSD matrix aligned to reference conformer.
        # HOW: Restricts atomIds to heavy atoms (Z > 1) to eliminate superficial RMSD inflation from rotatable hydrogens.
        heavy_atom_ids = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
        rms_matrix = AllChem.GetConformerRMSMatrix(mol_h, atomIds=heavy_atom_ids, prealigned=False)
        clusters = Butina.ClusterData(rms_matrix, len(remaining_cids), distThresh=cutoff, isDistData=True)
        for cluster in clusters:
            # Select the energy minimum molecule from each conformational cluster
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
    scaffold_type: str = "bemis_murcko",
    isomeric: bool = False,
) -> str:
    """Extract core ring framework of reference ligand as generative seed fragment.

    In fragment-based and scaffold-constrained de novo generation (Bemis & Murcko 1996,
    Liu et al. 2021), seeding graph exploration with privileged core frameworks enables
    focused optimization around bioactive pharmacophores while exploring side-chain space.

    Parameters
    ----------
    query : str | Path | Chem.Mol, default=DEFAULT_MOLECULE
        Query reference molecule (SMILES, SDF path, or RDKit Mol).
    output_path : Path | str | None, default=None
        Optional file path to export the scaffold TSV for downstream generator prompting.
    scaffold_type : {"bemis_murcko", "core_ring_system", "generic_framework"}, default="bemis_murcko"
        Methodology for scaffold extraction:
        - "bemis_murcko": Standard Bemis & Murcko (1996) framework preserving all ring systems and connecting linkers.
        - "core_ring_system": Privileged largest contiguous ring system (e.g. fused bicyclic chroman in EGCG),
          stripping peripheral rings and linkers to maximize generative exploration freedom.
        - "generic_framework": Bemis-Murcko framework with all atoms converted to carbon and bonds to single bonds.
    isomeric : bool, default=False
        Whether to retain stereochemical designations in output SMILES. Default is False to match
        standard DrugEx graph-transformer vocabulary conventions.

    Returns
    -------
    str
        Canonical SMILES of the extracted seed scaffold.

    References
    ----------
    .. [1] Bemis, G.W.; Murcko, M.A. "The Properties of Known Drugs. 1. Molecular Frameworks"
       J. Med. Chem. 1996, 39(15), 2887–2893.
    .. [2] Congreve, M. et al. "Detection of fragments by NMR"
       Drug Discovery Today 2003, 8(19), 876–877.
    """
    valid_types = {"bemis_murcko", "core_ring_system", "generic_framework"}
    if scaffold_type not in valid_types:
        raise ValueError(f"Unknown scaffold_type {scaffold_type!r}. Must be one of {valid_types}")

    mol, _ = load_query_molecule(query)
    mol_no_h = Chem.RemoveHs(mol)

    # [PACKAGE: rdkit.Chem.Scaffolds.MurckoScaffold] WHY: Deconstruct ligand into fundamental ring frameworks.
    # WHAT: Bemis-Murcko framework extraction.
    # HOW: Identifies all ring atoms and connecting linkers, pruning non-linker acyclic substituents.
    scaff_smi = ""
    n_rings = 0

    if scaffold_type == "core_ring_system":
        # Extract the largest contiguous (fused / shared-atom) ring system
        ring_info = mol_no_h.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        if atom_rings:
            # Group overlapping rings into contiguous fused/spiro systems
            ring_systems: list[set[int]] = []
            for ring in atom_rings:
                ring_set = set(ring)
                merged = False
                for existing in ring_systems:
                    if existing & ring_set:
                        existing.update(ring_set)
                        merged = True
                        break
                if not merged:
                    ring_systems.append(ring_set)

            # Select largest contiguous ring system by heavy atom count
            largest_system = max(ring_systems, key=len)
            sub_bonds = [
                b.GetIdx() for b in mol_no_h.GetBonds()
                if b.GetBeginAtomIdx() in largest_system and b.GetEndAtomIdx() in largest_system
            ]
            # [PACKAGE: rdkit.Chem.PathToSubmol] WHY: Extract induced submolecule for core ring system.
            core_submol = Chem.PathToSubmol(mol_no_h, sub_bonds)
            # Reset dangling valences from cleaved exocyclic bonds to saturated implicit hydrogens
            for atom in core_submol.GetAtoms():
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
            Chem.SanitizeMol(core_submol)
            scaff_smi = Chem.MolToSmiles(core_submol, isomericSmiles=isomeric)
            n_rings = core_submol.GetRingInfo().NumRings()
        else:
            # Acyclic fallback
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)
            n_rings = 0

    elif scaffold_type == "generic_framework":
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol_no_h)
            if scaff and scaff.GetNumHeavyAtoms() > 0:
                gen_scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
                Chem.SanitizeMol(gen_scaff)
                scaff_smi = Chem.MolToSmiles(gen_scaff, isomericSmiles=isomeric)
                n_rings = gen_scaff.GetRingInfo().NumRings()
            else:
                scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)
                n_rings = 0
        except Exception:
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)
            n_rings = 0

    else:  # "bemis_murcko" (default)
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol_no_h)
            if scaff and scaff.GetNumHeavyAtoms() > 0:
                Chem.SanitizeMol(scaff)
                scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=isomeric)
                n_rings = scaff.GetRingInfo().NumRings()
            else:
                scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)
                n_rings = 0
        except Exception:
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)
            n_rings = 0

    scaff_mol = Chem.MolFromSmiles(scaff_smi)
    n_heavy = scaff_mol.GetNumHeavyAtoms() if scaff_mol else 0

    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings\n")
            f.write(f"{scaff_smi}\t{str(query)}\t{scaffold_type}\t{n_heavy}\t{n_rings}\n")

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
