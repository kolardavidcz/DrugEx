from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import (
    ParetoCrowdingDistance,
    ParetoTanimotoDistance,
    WeightedSum,
)
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.interfaces import ScoreModifier, Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from _david.code.utils import FallbackMonitor, HardwareManager

DEFAULT_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)


class PrepareMolecule:
    def __init__(
        self,
        num_conformers: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.num_conformers = num_conformers
        self.random_seed = random_seed

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

    def load_query_molecule(
        self, query: str | Path | Chem.Mol
    ) -> tuple[Chem.Mol, bool]:
        # 1. Direct RDKit Mol instance
        if isinstance(query, Chem.Mol):
            return query, bool(query.GetNumConformers() and query.GetConformer().Is3D())

        # 2. Fast Path: SMILES string (avoids OS filesystem syscalls & filename length limits)
        if isinstance(query, str) and not query.endswith(
            (".sdf", ".sd", ".txt", ".smi")
        ):
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
    def _calculate_adaptive_rmsd_cutoff(
        mol: Chem.Mol | None,
    ) -> tuple[float, int]:
        """Calculate heavy-atom RMSD cutoff and maximum reference ensemble size (Hawkins 2017).

        WHY:
        Rigid molecules have minimal conformational entropy; an RMSD cutoff of 1.00 Å
        and a single reference conformer (K = 1) avoids multi-modal gradient noise in RL.
        Standard drug-like molecules (e.g. EGCG) have 3-7 rotatable bonds; an RMSD cutoff of
        1.25 Å and K = 3 conformers captures the primary bioactive energy wells without
        causing gradient conflicts during policy gradient optimization.
        Highly flexible ligands (> 7 rotatable bonds) require an RMSD cutoff of 1.50 Å
        and up to K = 5 conformers to span their wider conformational volume.

        WHAT:
        Returns (rmsd_cutoff: float, max_ensemble_size: int):
        - Rigid/compact (<= 2 rotatable bonds, <= 20 heavy atoms): (1.00 Å, 1)
        - Drug-like (<= 7 rotatable bonds, <= 45 heavy atoms): (1.25 Å, 3)
        - Flexible (> 7 rotatable bonds or > 45 heavy atoms): (1.50 Å, 5)
        - Empty/null molecule: (1.00 Å, 1)

        HOW:
        Removes explicit hydrogens before counting rotatable bonds and heavy atoms to ensure
        topological invariance regardless of whether hydrogens are implicit or explicit.
        """
        if mol is None:
            return 1.00, 1
        # Enforce topological invariance against explicit hydrogens
        mol_no_h = (
            Chem.RemoveHs(mol) if mol.GetNumAtoms() > mol.GetNumHeavyAtoms() else mol
        )
        n_heavy = int(mol_no_h.GetNumHeavyAtoms())
        if n_heavy == 0:
            return 1.00, 1
        n_rot = int(Lipinski.NumRotatableBonds(mol_no_h))
        flexible_rotatable_bonds = 7
        if n_rot <= 2 and n_heavy <= 20:
            return 1.00, 1
        if n_rot <= flexible_rotatable_bonds and n_heavy <= 45:
            return 1.25, 1  # NOTE just testing 1, true value sould be: 3
        return 1.50, 1  # NOTE just testing 1, true value sould be: 5

    def generate_reference_conformers(
        self,
        query: str | Path | Chem.Mol,
        output_path: Path | str | None = None,
        num_conformers: int = 100,  # ^ HYP could be bigger, like 200
        random_seed: int = 42,
    ) -> Path:
        output_path_real = (
            Path(output_path) if output_path else Path("reference_3d.sdf")
        )
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
        params.pruneRmsThresh = 0.35  # since is single ligand its quite detailed (RDKit papers use 0.35-0.5, 0.1-0.2 is normal thermal vibrations)

        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        )  # based on crystal predictions
        if not conf_ids:
            params.useRandomCoords = (
                True  # based on random initial + gradient optimization
            )
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol_h, numConfs=num_conformers, params=params
                )
            )
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 embedding failed for: {query}")

        # 3. Force Field Minimization & Energy Profiling (MMFF94 with UFF fallback)
        # [PACKAGE: rdkit.Chem.AllChem] WHY: Multi-threaded batch force field minimization with UFF fallback for non-MMFF elements.
        results = []
        fallback_reason: str | None = None
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=0
            )
            # If all MMFF conformers suffered setup failure (status == -1), attempt UFF fallback
            if results and all(status == -1 for status, _ in results):
                results = []
                fallback_reason = (
                    "MMFF94 setup failed for all conformers; falling back to UFF."
                )
        else:
            fallback_reason = (
                "MMFF94 lacks parameters for query molecule; falling back to UFF."
            )

        if not results and AllChem.UFFHasAllMoleculeParams(mol_h):
            if fallback_reason:
                FallbackMonitor.record_fallback(
                    "FORCE_FIELD",
                    fallback_reason,
                    {"query": str(query)},
                )
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=0
            )
        if not results:
            raise RuntimeError(f"Force field optimization failed for: {query}")

        # Filter out failed setup (status == -1) and non-finite energies
        energies: dict[int, float] = {}
        for cid, (status, e) in zip(conf_ids, results):
            e_val = float(e)
            if status == -1 or not math.isfinite(e_val):
                mol_h.RemoveConformer(cid)
            else:
                energies[cid] = e_val

        if not energies:
            raise RuntimeError(
                f"Force field optimization produced zero valid conformers for: {query}"
            )

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
        cutoff, max_ensemble_size = self._calculate_adaptive_rmsd_cutoff(mol_h)
        if len(remaining_cids) > 1:
            # [PACKAGE: rdkit.Chem.AllChem.GetBestRMS & rdkit.ML.Cluster.Butina]
            # WHY: Fixed atom-index RMSD treats 180° rotations of symmetric moieties (phenyl, catechol, gallate)
            #      as artificial new conformers (>1.8 Å). GetBestRMS evaluates topological automorphisms.
            # WHAT: Lower-triangular heavy-atom distance matrix matching true geometric symmetry.
            # HOW: Chem.RemoveHs isolates heavy atoms (Hawkins 2017) and GetBestRMS finds minimal symmetry RMSD.
            mol_no_h = Chem.RemoveHs(mol_h)
            n_confs = len(remaining_cids)
            rms_matrix: list[float] = [
                float(
                    AllChem.GetBestRMS(
                        mol_no_h, mol_no_h, remaining_cids[i], remaining_cids[j]
                    )
                )
                for i in range(n_confs)
                for j in range(i)
            ]
            clusters = Butina.ClusterData(
                rms_matrix, n_confs, distThresh=cutoff, isDistData=True
            )

            # Select lowest-energy conformer within each conformational basin
            curated = []
            for cluster in clusters:
                basin_cids = [remaining_cids[i] for i in cluster]
                curated.append(min(basin_cids, key=energies.get))
            curated.sort(key=energies.get)
            curated = curated[:max_ensemble_size]
        else:
            curated = remaining_cids[:max_ensemble_size]

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
        query: str | Path | Chem.Mol,
        output_path: Path | str | None = None,
        scaffold_type: str = "BemisMurcko",
        isomeric: bool = False,
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
        max_conformers=30,  # ^ HYP could be bigger, like 50
        max_heavy_atoms=55,
        max_rotatable_bonds=25,
        timeout=60,  # ^ HYP could be bigger, like 60
    )

    # 3. Multi-Threaded ROCS Alignment Construction
    return RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        color_ff="oe_mills_dean",  # standard
        n_jobs=HardwareManager.get_safe_parallel_jobs(),
    )


class SafeSAModifier(ScoreModifier):
    """Piecewise-linear SA feasibility floor that explicitly guards uncalculable (<= 0.0) fallback.

    WHY:
    Raw SA score is defined on [1.0, 10.0]. When DrugEx Property.getScores encounters an
    unhandled exception or uncalculable molecular graph, it defaults to raw score 0.0.
    Under standard ClippedScore(lower_x=6.0, upper_x=5.5), any score <= 5.5 evaluates to
    maximum desirability (1.0), rewarding crashed molecules with perfect synthesizability.

    WHAT:
    Maps raw SA in [5.5, 6.0] to [1.0, 0.0], but maps any raw score <= 0.0 (crash/uncalculated)
    strictly to 0.0.

    HOW:
    Applies ClippedScore and masks out values <= 0.0 with 0.0 reward.
    """

    def __init__(
        self,
        lower_x: float = 6.0,
        upper_x: float = 5.5,
        low_score: float = 0.0,
        high_score: float = 1.0,
    ) -> None:
        self.clip = ClippedScore(
            lower_x=lower_x, upper_x=upper_x, low_score=low_score, high_score=high_score
        )

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        out = self.clip(arr)
        out = np.where(
            arr <= 0.0, 0.0, out
        )  # Any chemically invalid or uncalculated fallback (SA <= 0.0) receives 0.0 desirability
        return float(out.item()) if np.ndim(x) == 0 else out


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
        modifier=SafeSAModifier(
            lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0
        ),
    )

    # 3. Pareto Multi-Objective DrugEx Environment
    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    thresholds: list[float] = [0.25, 0.25]

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=WeightedSum(),
    )
    return env, epsilon
