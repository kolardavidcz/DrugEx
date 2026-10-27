from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, RewardScheme, WeightedSum
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore, SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from rdkit.ML.Cluster import Butina

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig
from _david.code.schema import std
from _david.code.utils import HardwareManager


class PrepareMolecule:
    """Standardizes input chemical queries and generates 3D reference conformer ensembles."""

    def __init__(
        self,
        num_conformers: int = 100,
        max_reference_conformers: int = 3,
        random_seed: int = 42,
    ) -> None:
        self.num_conformers = num_conformers
        self.max_reference_conformers = max_reference_conformers
        self.random_seed = random_seed
        self.standardizer: DefaultStandardizer = DefaultStandardizer(
            preserve_stereochemistry=True
        )

    def standardize(self, smiles: str) -> str:
        """Standardize input chemical structure into canonical SMILES representation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse molecule query: {smiles!r}")
        try:
            return self.standardizer(smiles)
        except Exception:
            return Chem.MolToSmiles(mol)

    def load_molecule(self, query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
        """Load and standardize molecule across RDKit Mols, SMILES strings, and structure files."""
        if isinstance(query, Chem.Mol):
            if query.GetNumConformers() > 0 and query.GetConformer().Is3D():
                return query, True
            if query.GetNumAtoms() > query.GetNumHeavyAtoms():
                return Chem.RemoveHs(query), False
            return query, False

        p = (
            Path(query)
            if isinstance(query, (Path, str))
            and (
                isinstance(query, Path)
                or query.endswith((".sdf", ".sd", ".txt", ".smi"))
            )
            else None
        )

        if p and p.is_file():
            if p.suffix.lower() in {".sdf", ".sd"}:
                mols = [
                    m
                    for m in Chem.SDMolSupplier(str(p), removeHs=False)
                    if m is not None
                ]
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

    @classmethod
    def determine_ensemble_parameters(
        cls,
        mol: Chem.Mol | None,
        max_ensemble_size: int = 3,
    ) -> tuple[float, int]:
        """Calculate Butina RMSD clustering cutoff and maximum reference ensemble size based on molecular flexibility."""
        if mol is None:
            return 1.00, 1
        mol_no_h = (
            Chem.RemoveHs(mol) if mol.GetNumAtoms() > mol.GetNumHeavyAtoms() else mol
        )
        n_heavy = int(mol_no_h.GetNumHeavyAtoms())
        if n_heavy == 0:
            return 1.00, 1
        n_rot = int(Lipinski.NumRotatableBonds(mol_no_h))

        if n_rot <= 2 and n_heavy <= 18:
            return 1.00, min(1, max_ensemble_size)
        if n_rot <= 7:
            return 1.25, min(3, max_ensemble_size)
        return 1.50, min(3, max_ensemble_size)

    @classmethod
    def calculate_conformer_energies(
        cls,
        mol_h: Chem.Mol,
        query_desc: str = "",
    ) -> dict[int, float]:
        """Minimize conformers using MMFF94 with UFF fallback and return valid conformer energies in kcal/mol."""
        results = []
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=0
            )
            if results and all(status == -1 for status, _ in results):
                results = []

        if not results and AllChem.UFFHasAllMoleculeParams(mol_h):
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=0
            )

        if not results:
            raise RuntimeError(f"Force field optimization failed for: {query_desc}")

        energies: dict[int, float] = {}
        for conf, (status, e) in zip(mol_h.GetConformers(), results):
            cid = conf.GetId()
            e_val = float(e)
            if status != -1 and math.isfinite(e_val):
                energies[cid] = e_val
            else:
                mol_h.RemoveConformer(cid)

        if not energies:
            raise RuntimeError(
                f"Force field optimization produced zero valid conformers for: {query_desc}"
            )
        return energies

    def generate_reference_conformers(
        self,
        query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
        output_path: Path | str | None = None,
        num_conformers: int = 100,
        random_seed: int = 42,
    ) -> Path:
        """Embed, minimize, energy-filter, and cluster stereochemically specified references."""
        target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        mol, is_3d = self.load_molecule(query)
        if is_3d:
            with Chem.SDWriter(str(target_path)) as writer:
                for conf in mol.GetConformers():
                    writer.write(mol, confId=conf.GetId())
            return target_path

        # ETKDG respects the stereochemistry retained by reference standardization.
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.pruneRmsThresh = 0.35

        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        )
        if not conf_ids:
            params.useRandomCoords = True
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol_h, numConfs=num_conformers, params=params
                )
            )
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {query}")

        energies = self.calculate_conformer_energies(mol_h, query_desc=str(query))

        # This project energy window is not a physiological population model.
        min_e = min(energies.values())
        for cid, e in list(energies.items()):
            if (e - min_e) > 8.0:
                mol_h.RemoveConformer(cid)
                del energies[cid]

        # Retain the lowest-energy representative of each heavy-atom RMSD basin.
        remaining_cids = list(energies.keys())
        cutoff, max_ensemble_size = self.determine_ensemble_parameters(
            mol_h, self.max_reference_conformers
        )
        if len(remaining_cids) > 1:
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
            curated = [
                min([remaining_cids[idx] for idx in cl], key=energies.get)
                for cl in clusters
            ]
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


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    max_conformers: int = 30,
    timeout: int = 0,
    score_variant: str = "tanimoto_combo",
    conformer_threads: int = 0,
    n_jobs: int | None = None,
) -> RDKitROCSScorer:
    """Build multi-threaded RDKitROCSScorer configured to produce canonical std.ROCS key."""
    ref_input = str(references) if isinstance(references, Path) else references
    conf_gen = RDKitConformerGenerator(
        max_conformers=max_conformers,
        max_isomers=4,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=conformer_threads,
        show_progress=False,
    )
    safe_jobs = (
        n_jobs
        if n_jobs is not None
        else HardwareManager.get_safe_parallel_jobs(ratio=0.90)
    )
    variant = (
        "tanimoto" if score_variant in ("tanimoto", "tanimoto_combo") else score_variant
    )
    scorer = RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        score_variant=variant,
        use_colors=True,
        show_progress=False,
        n_jobs=safe_jobs,
    )
    scorer.getKey = lambda: [std.ROCS]
    return scorer


class HarringtonDesirabilityReward(RewardScheme):
    """Continuous Multi-Attribute Desirability (Harrington 1965) for Molecular RL.

    Maps each objective score s_j through a two-parameter Gompertz desirability function:
        d_j(s_j) = exp(-exp(-(b_{0,j} + b_{1,j} * s_j)))
    The composite desirability D(x) is the weighted geometric mean across all M objectives:
        D(x) = exp(sum_j w_j * ln(d_j(s_j) + eps) / sum_j w_j)
    """

    def __init__(
        self,
        weights: Sequence[float] = (0.50, 0.25, 0.25),
        rocs_anchors: tuple[float, float, float, float] = (0.50, 0.05, 0.70, 0.70),
        sa_anchors: tuple[float, float, float, float] = (0.20, 0.05, 0.80, 0.90),
        qed_anchors: tuple[float, float, float, float] = (0.30, 0.05, 0.70, 0.90),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = np.array(weights, dtype=np.float64)
        self.eps = float(eps)
        self.params: list[tuple[float, float]] = [
            self._fit_anchors(*rocs_anchors),
            self._fit_anchors(*sa_anchors),
            self._fit_anchors(*qed_anchors),
        ]

    @staticmethod
    def _fit_anchors(y0: float, d0: float, y1: float, d1: float) -> tuple[float, float]:
        """Fit Gompertz parameters (b0, b1) from two anchor points (y0, d0) and (y1, d1)."""
        y0_prime = -np.log(-np.log(d0))
        y1_prime = -np.log(-np.log(d1))
        b1 = (y1_prime - y0_prime) / (y1 - y0)
        b0 = y0_prime - b1 * y0
        return float(b0), float(b1)

    @staticmethod
    def _eval_desirability(s: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Evaluate Gompertz desirability function on score array s."""
        z = b0 + b1 * s
        z_clipped = np.clip(z, -30.0, 30.0)
        return np.exp(-np.exp(-z_clipped))

    def __call__(
        self,
        smiles: Sequence[str] | np.ndarray,
        scores: np.ndarray | None = None,
        thresholds: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate continuous composite Harrington desirability rewards."""
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            scores_arr = np.asarray(smiles, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = (
                scores_arr.reshape(1, -1)
                if len(scores_arr) > 0
                else scores_arr.reshape(0, 1)
            )

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        scores_clean = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)
        scores_clean = np.clip(scores_clean, 0.0, 1.0)
        n_objs = scores_clean.shape[1]

        w = (
            self.weights[:n_objs]
            if len(self.weights) >= n_objs
            else np.ones(n_objs, dtype=np.float64)
        )
        w_norm = w / np.sum(w)

        log_d_sum = np.zeros(n_samples, dtype=np.float64)
        for j in range(n_objs):
            b0, b1 = self.params[j] if j < len(self.params) else (0.0, 5.0)
            d_j = self._eval_desirability(scores_clean[:, j], b0, b1)
            log_d_sum += w_norm[j] * np.log(d_j + self.eps)

        rewards = np.exp(log_d_sum)
        rewards = np.clip(rewards, 0.0, 1.0)
        return rewards.reshape(-1, 1)


def resolve_reward_scheme(
    scheme: str | RewardScheme | None = None,
    weights: tuple[float, ...] = (0.50, 0.25, 0.25),
) -> RewardScheme:
    """Resolve RewardScheme instance from canonical name or pass through existing instance."""
    if isinstance(scheme, RewardScheme):
        return scheme
    name = (scheme or "weighted_sum").lower().strip()
    if name in ("weighted_sum", "ws"):
        return WeightedSum()
    elif name in ("pareto", "pareto_crowding"):
        return ParetoCrowdingDistance()
    elif name in ("harrington", "harrington_desirability", "desirability"):
        return HarringtonDesirabilityReward(weights=weights)
    else:
        raise ValueError(
            f"Unknown reward scheme: '{scheme}'. Valid options: ['weighted_sum', 'pareto_crowding', 'harrington']"
        )


def create_environment(
    ref_sdf: str | Path,
    config: PipelineConfig | None = None,
    thresholds: tuple[float, ...] | None = None,
    max_conformers: int = 30,
    timeout: int = 0,
    reward_scheme: str | RewardScheme | None = None,
    score_variant: str = "tanimoto_combo",
    conformer_threads: int = 0,
) -> DrugExEnvironment:
    """Construct multi-objective DrugEx reinforcement learning environment for ROCS + SA + QED."""
    cfg = config or PipelineConfig()
    active_thresholds = thresholds if thresholds is not None else cfg.thresholds
    active_scheme_str = (
        reward_scheme if reward_scheme is not None else cfg.reward_scheme
    )
    active_variant = score_variant or cfg.score_variant
    active_max_confs = max_conformers or cfg.max_conformers
    active_threads = (
        conformer_threads if conformer_threads is not None else cfg.conformer_threads
    )

    rocs_scorer = create_rocs_scorer(
        ref_sdf,
        max_conformers=active_max_confs,
        timeout=timeout,
        score_variant=active_variant,
        conformer_threads=active_threads,
    )
    # Map raw TanimotoCombo [0, 2] linearly into [0, 1] desirability
    rocs_scorer.setModifier(
        ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)
    )

    # SA scorer: Smooth clipped score (raw 3.0 -> 0.88, raw 5.0 -> 0.12)
    sa_scorer = Property("SA")
    sa_scorer.setModifier(
        SmoothClippedScore(lower_x=5.0, upper_x=3.0, low_score=0.0, high_score=1.0)
    )

    # QED scorer: Drug-likeness score [0, 1]
    qed_scorer = Property("QED")
    qed_scorer.setModifier(
        ClippedScore(lower_x=0.0, upper_x=1.0, low_score=0.0, high_score=1.0)
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer, qed_scorer]
    active_reward_scheme = resolve_reward_scheme(active_scheme_str, weights=cfg.weights)

    return DrugExEnvironment(
        scorers=scorers,
        thresholds=list(active_thresholds),
        reward_scheme=active_reward_scheme,
    )
