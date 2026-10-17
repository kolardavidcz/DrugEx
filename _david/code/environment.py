from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import WeightedSum
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.interfaces import ScoreModifier, Scorer
from drugex.training.scorers.modifiers import ClippedScore, SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from _david.code.config import DEFAULT_MOLECULE
from _david.code.schema import std
from _david.code.utils import FallbackMonitor, HardwareManager


class PrepareMolecule:
    """Standardizes inputs, prepares 3D reference conformers, and extracts seed core scaffolds."""

    def __init__(
        self,
        num_conformers: int = 100,
        max_reference_conformers: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.num_conformers = num_conformers
        self.max_reference_conformers = max_reference_conformers
        self.random_seed = random_seed
        self.standardizer = DefaultStandardizer()

    def standardize(self, smiles: str) -> str:
        """Standardize, desalt, uncharge, and normalize input chemical structure using DrugEx standards."""
        try:
            return self.standardizer(smiles)
        except Exception:
            return Chem.CanonSmiles(smiles)

    def load_molecule(self, query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
        """Load and standardize molecule across RDKit Mols, SMILES strings, and structure files."""
        if isinstance(query, Chem.Mol):
            if query.GetNumConformers() > 0 and query.GetConformer().Is3D():
                return query, True
            return Chem.RemoveHs(query), False

        p = Path(query) if isinstance(query, (Path, str)) and (
            isinstance(query, Path) or query.endswith((".sdf", ".sd", ".txt", ".smi"))
        ) else None

        if p and p.is_file():
            if p.suffix.lower() in {".sdf", ".sd"}:
                mols = [m for m in Chem.SDMolSupplier(str(p), removeHs=False) if m is not None]
                if not mols:
                    raise ValueError(f"No valid molecule in SDF: {p}")
                base = mols[0]
                for extra in mols[1:]:
                    base.AddConformer(extra.GetConformer(), assignId=True)
                if base.GetNumConformers() > 0 and base.GetConformer().Is3D():
                    return base, True
                return Chem.RemoveHs(base), False

            text = p.read_text(encoding="utf-8").strip()
            tokens = text.split()
            if not tokens:
                raise ValueError(f"Empty molecule file: {p}")
            query = tokens[0]

        if isinstance(query, str):
            std_smi = self.standardize(query)
            mol = Chem.MolFromSmiles(std_smi)
            if mol is None:
                raise ValueError(f"Could not parse molecule query: {query!r}")
            return Chem.RemoveHs(mol), False

        raise ValueError(f"Could not parse molecule query: {query!r}")

    # Canonical aliases for backward compatibility
    load_query_molecule = load_molecule

    def load_sdf_ensemble(self, path: Path | str) -> tuple[Chem.Mol, bool]:
        """Backward-compatible alias delegating to load_molecule."""
        return self.load_molecule(path)

    @classmethod
    def get_adaptive_rmsd_cutoff(
        cls,
        mol: Chem.Mol | None,
        max_ensemble_size: int = 1,
    ) -> tuple[float, int]:
        # ~ [Hawkins, Conformer Generation with RDKit, J. Chem. Inf. Model. 2017]
        if mol is None:
            return 1.00, 1
        mol_no_h = Chem.RemoveHs(mol) if mol.GetNumAtoms() > mol.GetNumHeavyAtoms() else mol
        n_heavy = int(mol_no_h.GetNumHeavyAtoms())
        if n_heavy == 0:
            return 1.00, 1
        n_rot = int(Lipinski.NumRotatableBonds(mol_no_h))

        if n_rot <= 2 and n_heavy <= 20:
            return 1.00, min(1, max_ensemble_size)
        if n_rot <= 7 and n_heavy <= 45:
            return 1.25, min(3, max_ensemble_size)
        return 1.50, min(5, max_ensemble_size)

    _calculate_adaptive_rmsd_cutoff = get_adaptive_rmsd_cutoff

    def generate_reference_conformers(
        self,
        query: str | Path | Chem.Mol,
        output_path: Path | str | None = None,
        num_conformers: int = 100,
        random_seed: int = 42,
    ) -> Path:
        """Embed, optimize, filter energy window, and cluster 3D reference conformers."""
        target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        mol, is_3d = self.load_molecule(query)
        if is_3d:
            with Chem.SDWriter(str(target_path)) as writer:
                for conf in mol.GetConformers():
                    writer.write(mol, confId=conf.GetId())
            return target_path

        # ~ [Hawkins, Conformer Generation with RDKit, J. Chem. Inf. Model. 2017] ETKDGv3 embedding
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = 0.35

        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if not conf_ids:
            params.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params))
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {query}")

        # Force field minimization with MMFF94 and UFF fallback
        results = []
        fallback_reason: str | None = None
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=1000, numThreads=0)
            if results and all(status == -1 for status, _ in results):
                results = []
                fallback_reason = "MMFF94 setup failed for all conformers; falling back to UFF."
        else:
            fallback_reason = "MMFF94 lacks parameters for query molecule; falling back to UFF."

        if not results and AllChem.UFFHasAllMoleculeParams(mol_h):
            if fallback_reason:
                FallbackMonitor.record_fallback("FORCE_FIELD", fallback_reason, {"query": str(query)})
            results = AllChem.UFFOptimizeMoleculeConfs(mol_h, maxIters=1000, numThreads=0)
        if not results:
            raise RuntimeError(f"Force field optimization failed for: {query}")

        energies: dict[int, float] = {}
        for cid, (status, e) in zip(conf_ids, results):
            e_val = float(e)
            if status == -1 or not math.isfinite(e_val):
                mol_h.RemoveConformer(cid)
            else:
                energies[cid] = e_val

        if not energies:
            raise RuntimeError(f"Force field optimization produced zero valid conformers for: {query}")

        # ~ [Hawkins, Physiological energy window, 2017] Delta E <= 8.0 kcal/mol window
        min_e = min(energies.values())
        for cid, e in list(energies.items()):
            if (e - min_e) > 8.0:
                mol_h.RemoveConformer(cid)
                del energies[cid]

        # ~ [Hawkins, Basin curation via Butina clustering, 2017] Sphere-exclusion RMSD clustering
        remaining_cids = list(energies.keys())
        cutoff, max_ensemble_size = self.get_adaptive_rmsd_cutoff(mol_h, self.max_reference_conformers)
        if len(remaining_cids) > 1:
            mol_no_h = Chem.RemoveHs(mol_h)
            n_confs = len(remaining_cids)
            rms_matrix: list[float] = [
                float(AllChem.GetBestRMS(mol_no_h, mol_no_h, remaining_cids[i], remaining_cids[j]))
                for i in range(n_confs)
                for j in range(i)
            ]
            clusters = Butina.ClusterData(rms_matrix, n_confs, distThresh=cutoff, isDistData=True)
            curated = [min([remaining_cids[idx] for idx in cl], key=energies.get) for cl in clusters]
            curated.sort(key=energies.get)
            curated = curated[:max_ensemble_size]
        else:
            curated = remaining_cids[:max_ensemble_size]

        with Chem.SDWriter(str(target_path)) as writer:
            for i, cid in enumerate(curated):
                mol_h.SetProp("_Name", f"Conformer_{i + 1}")
                mol_h.SetDoubleProp("Energy_MMFF_kcal_mol", energies[cid])
                mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
                writer.write(mol_h, confId=cid)

        return target_path

    def extract_seed_core_scaffold(
        self,
        query: str | Path | Chem.Mol,
        output_path: Path | str | None = None,
        scaffold_type: str = "BemisMurcko",
        isomeric: bool = False,
    ) -> str:
        # ~ [Bemis & Murcko, The Properties of Known Drugs. 1. Molecular Frameworks, J. Med. Chem. 1996]
        mol, _ = self.load_molecule(query)
        mol_no_h = Chem.RemoveHs(mol)

        scaff_mol = MurckoScaffold.GetScaffoldForMol(mol_no_h)
        if scaff_mol and scaff_mol.GetNumHeavyAtoms() > 0:
            Chem.SanitizeMol(scaff_mol)
            scaff_smi = Chem.MolToSmiles(scaff_mol, isomericSmiles=isomeric)
        else:
            scaff_smi = Chem.MolToSmiles(mol_no_h, isomericSmiles=isomeric)

        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            res_mol = Chem.MolFromSmiles(scaff_smi)
            n_heavy = res_mol.GetNumHeavyAtoms() if res_mol else 0
            n_rings = res_mol.GetRingInfo().NumRings() if res_mol else 0
            with open(target, "w", encoding="utf-8") as f:
                f.write("Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings\n")
                f.write(f"{scaff_smi}\t{str(query)}\t{scaffold_type}\t{n_heavy}\t{n_rings}\n")

        return scaff_smi


# Top-level module convenience dispatchers
def load_query_molecule(query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Top-level convenience helper loading molecule via default PrepareMolecule."""
    return PrepareMolecule().load_molecule(query)


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    max_conformers: int = 30,
    timeout: int = 60,
) -> RDKitROCSScorer:
    """Build multi-threaded RDKitROCSScorer configured to produce canonical std.ROCS key."""
    ref_input = str(references) if isinstance(references, Path) else references
    conf_gen = RDKitConformerGenerator(
        max_conformers=max_conformers,
        max_heavy_atoms=55,
        max_rotatable_bonds=25,
        timeout=timeout,
    )
    scorer = RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        color_ff="oe_mills_dean",
        n_jobs=HardwareManager.get_safe_parallel_jobs(),
    )
    scorer.getKey = lambda: [std.ROCS]
    return scorer


class SafeSAModifier(ScoreModifier):
    # ~ [Ertl & Schuffenhauer, Estimation of Synthetic Accessibility Score, J. Cheminform. 2009]
    def __init__(
        self,
        upper_x: float = 3.5,
        lower_x: float = 5.5,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        self.smooth = SmoothClippedScore(
            upper_x=upper_x, lower_x=lower_x, low_score=low_score, high_score=high_score
        )

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """Map raw SAscore [1, 10] into desirability [0, 1], strictly penalizing invalid molecules (<= 0.0)."""
        arr = np.asarray(x, dtype=np.float64)
        out = self.smooth(arr)
        out = np.where(arr <= 0.0, 0.0, out)
        return float(out.item()) if np.ndim(x) == 0 else out


def create_environment(
    ref_sdf: str | Path,
    thresholds: tuple[float, float] = (0.5, 0.5),
    epsilon: float = 0.2,
    max_conformers: int = 30,
    timeout: int = 60,
) -> tuple[DrugExEnvironment, float]:
    """Construct multi-objective DrugEx reinforcement learning environment."""
    rocs_scorer = create_rocs_scorer(ref_sdf, max_conformers=max_conformers, timeout=timeout)
    rocs_scorer.modifier = ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)

    sa_scorer = Property(
        prop="SA",
        modifier=SafeSAModifier(upper_x=3.5, lower_x=5.5, low_score=0.0, high_score=1.0),
    )
    scorers: list[Scorer] = [rocs_scorer, sa_scorer]

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=list(thresholds),
        reward_scheme=WeightedSum(),
    )
    return env, epsilon
