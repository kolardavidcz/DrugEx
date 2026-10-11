"""Environment construction and 3D reference conformer preparation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def render_reference_ensemble(
    sdf_path: Path | str,
    output_png: Path | str | None = None,
) -> Path:
    """Render publication-quality 3-panel 3D superposition figure of reference conformers."""
    sdf_file = Path(sdf_path)
    target_png = Path(output_png) if output_png else sdf_file.parent / "reference_3d_superposition.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return target_png

    mol_combined = Chem.Mol(mols[0])
    mol_combined.RemoveAllConformers()
    for m in mols:
        mol_combined.AddConformer(m.GetConformer(), assignId=True)

    props = AllChem.MMFFGetMoleculeProperties(mol_combined)
    energies: list[float] = []
    for i in range(mol_combined.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol_combined, props, confId=i) if props else None
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(mol_combined, confId=i)
        energies.append(float(ff.CalcEnergy()) if ff else 0.0)

    min_e = min(energies)
    de_vals = [e - min_e for e in energies]
    min_idx = int(np.argmin(de_vals))
    max_idx = int(np.argmax(de_vals))

    scaff = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(mols[0]))
    match = mols[0].GetSubstructMatch(scaff) if scaff else ()
    core_atoms = set(match) if match else set(range(min(10, mols[0].GetNumAtoms())))

    fig = plt.figure(figsize=(18, 5.5), dpi=200)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(de_vals) if max(de_vals) > 0 else 1.0)

    def _draw_mol(ax, mol, bond_c, atom_c, alpha=1.0, lw=2.0, pt_size=30, label_core=False):
        conf = mol.GetConformer()
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetSymbol() == "H" or a2.GetSymbol() == "H":
                continue
            p1, p2 = conf.GetAtomPosition(a1.GetIdx()), conf.GetAtomPosition(a2.GetIdx())
            is_core = (a1.GetIdx() in core_atoms) and (a2.GetIdx() in core_atoms)
            c = "#1e293b" if (label_core and is_core) else bond_c
            w = lw * 1.3 if (label_core and is_core) else lw
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color=c, alpha=alpha, linewidth=w)

        heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        for a in heavy:
            p = conf.GetAtomPosition(a.GetIdx())
            is_core = a.GetIdx() in core_atoms
            c = "#0f172a" if (label_core and is_core) else atom_c
            s = pt_size * 1.4 if (label_core and is_core) else pt_size
            ax.scatter([p.x], [p.y], [p.z], color=c, s=s, alpha=alpha, edgecolors="none")

    # Panel A: Ground state
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_mol(ax1, mols[min_idx], "#0284c7", "#0284c7", lw=2.4, pt_size=40, label_core=True)
    ax1.set_title(f"(A) Ground State Basin (ΔE = {de_vals[min_idx]:.2f} kcal/mol)\nCompact Folded Conformation", fontsize=11, fontweight="bold", pad=12)
    ax1.view_init(elev=20, azim=40)
    ax1.set_xlabel("X (Å)", fontsize=8)
    ax1.set_ylabel("Y (Å)", fontsize=8)
    ax1.set_zlabel("Z (Å)", fontsize=8)

    # Panel B: Most extended state
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_mol(ax2, mols[max_idx], "#f59e0b", "#f59e0b", lw=2.4, pt_size=40, label_core=True)
    ax2.set_title(f"(B) Pocket-Adapted Basin (ΔE = {de_vals[max_idx]:.2f} kcal/mol)\nExtended Conformation", fontsize=11, fontweight="bold", pad=12)
    ax2.view_init(elev=20, azim=40)
    ax2.set_xlabel("X (Å)", fontsize=8)
    ax2.set_ylabel("Y (Å)", fontsize=8)
    ax2.set_zlabel("Z (Å)", fontsize=8)

    # Panel C: Core-aligned superposition
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for i, m in enumerate(mols):
        c = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, c, c, alpha=0.75, lw=1.6, pt_size=20)
    _draw_mol(ax3, mols[min_idx], "#0f172a", "#0f172a", alpha=0.9, lw=2.6, pt_size=35, label_core=True)
    ax3.set_title(f"(C) Core-Aligned Ensemble ({len(mols)} Centroids)\nPharmacophore Envelope & Rotameric Fanning", fontsize=11, fontweight="bold", pad=12)
    ax3.view_init(elev=20, azim=40)
    ax3.set_xlabel("X (Å)", fontsize=8)
    ax3.set_ylabel("Y (Å)", fontsize=8)
    ax3.set_zlabel("Z (Å)", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label("Relative Energy ΔE (kcal/mol)", fontsize=8, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close()
    return target_png


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
