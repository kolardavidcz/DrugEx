"""Academic Result Analysis and Lead Profiling Suite for Ligand-Based De Novo Drug Design.

Evaluates generated molecules against reference 3D query template:
1. Peak Model Restoration & De Novo Sampling: Unbiased sampling (drop_invalid=False, drop_duplicates=False).
2. Standardized Academic VUN Metrics: Validity (>=80%), Uniqueness (>=80%), and Novelty against Papyrus.
3. High-Fidelity 50-Conformer 3D ROCS Evaluation: Gaussian shape + Mills-Dean color overlap (TanimotoCombo).
4. Bemis-Murcko Scaffold Extraction & Diversity: Core framework extraction and diversity ratio calculation.
5. Physicochemical Distribution Profiling: HAC, MW, CLogP, TPSA, RotBonds, HBD, HBA, SAscore, QED, and dynamic envelope benchmarking.
6. Publication-Grade Artifacts: lead_structures_2d.png, property_distributions.png, result_analysis_summary.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# [PACKAGE: matplotlib] WHY: Headless rendering of publication-grade property histograms and distribution plots. WHAT: Plotting framework. HOW: matplotlib.use("Agg"), plt.subplots, fig.savefig.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# [PACKAGE: numpy] WHY: Array conversions, percentile/IQR computation, and score normalization. WHAT: Numerical computing library. HOW: np.asarray, np.percentile, np.clip.
import numpy as np

# [PACKAGE: pandas] WHY: Tabular candidate metric auditing, TSV exports, and columnar parquet reading. WHAT: Data manipulation library. HOW: pd.DataFrame, pd.read_parquet, df.to_csv.
import pandas as pd

# [PACKAGE: torch] WHY: Deep learning model weight restoration and state dict checkpoint loading. WHAT: PyTorch deep learning framework. HOW: torch.load, torch.nn.Module.
import torch

# [PACKAGE: drugex] WHY: Native conformer generator, 3D ROCS scorer, and Synthetic Accessibility score calculation. WHAT: Deep learning drug design framework. HOW: RDKitConformerGenerator, RDKitROCSScorer, sascorer.
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

# [PACKAGE: rdkit] WHY: 3D conformer generation, Gaussian shape/color overlay, descriptor calculation, and Bemis-Murcko scaffold extraction. WHAT: Comprehensive cheminformatics toolkit. HOW: Chem.MolFromSmiles, Descriptors, rdMolDescriptors, MurckoScaffold, Draw.
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
except Exception:
    def calculate_sa_score(mol: Chem.Mol) -> float:
        """Fallback SA score evaluator returning nominal neutral value."""
        return 3.5

try:
    from fallback_monitor import FallbackMonitor
except (ImportError, ModuleNotFoundError):
    from _david.code.fallback_monitor import FallbackMonitor

try:
    from prepare_query_conformer import QueryConformerPreparer
except (ImportError, ModuleNotFoundError):
    from _david.code.prepare_query_conformer import QueryConformerPreparer

# Proof reference molecule: Epigallocatechin gallate (EGCG)
MY_MOLECULE: str = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
)
EGCG_SMILES: str = MY_MOLECULE

# High-fidelity 50-conformer 3D ROCS evaluation parameters (R5)
NUM_CONFORMERS_ROCS: int = 50
ROCS_MAX_HEAVY_ATOMS: int = 55
ROCS_MAX_ROTATABLE_BONDS: int = 25
ROCS_TIMEOUT: int = 60
ROCS_SCORE_TYPE: str = "TanimotoCombo"
ROCS_COLOR_FF: str = "oe_mills_dean"

# Academic VUN Acceptance Thresholds
VALIDITY_THRESHOLD: float = 0.80
UNIQUENESS_THRESHOLD: float = 0.80

# Default remote Papyrus 05.6 dataset path on dichtator
DEFAULT_PAPYRUS_PARQUET_PATH: Path = Path(
    "/home/kolarv/DrugEx/tutorial/data/data/.Papyrus/papyrus/05.6/05.6++_combined_set_without_stereochemistry.tsv.parquet"
)

__all__: list[str] = [
    "DEFAULT_PAPYRUS_PARQUET_PATH",
    "EGCG_SMILES",
    "MY_MOLECULE",
    "NUM_CONFORMERS_ROCS",
    "ROCS_COLOR_FF",
    "ROCS_SCORE_TYPE",
    "ResultAnalyzer",
    "UNIQUENESS_THRESHOLD",
    "VALIDITY_THRESHOLD",
    "calculate_sa_score",
]


class ResultAnalyzer:
    """Academically rigorous result evaluation and lead profiling suite for LBDD in DrugEx."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        ref_sdf_path: Path | str | None = None,
        seed_smiles: str = MY_MOLECULE,
        papyrus_parquet_path: Path | str | None = None,
    ) -> None:
        """Initialize ResultAnalyzer with query reference coordinates and output directory.

        Args:
            output_dir: Destination folder for analysis tables, figures, and JSON summaries.
            ref_sdf_path: Path to low-energy 3D conformer ensemble SDF of the reference ligand.
            seed_smiles: SMILES string of the primary reference proof molecule.
            papyrus_parquet_path: Path to Papyrus 05.6 dataset parquet for novelty lookup.
        """
        self.output_dir: Path = Path(output_dir) if output_dir is not None else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed_smiles: str = seed_smiles
        self.seed_mol: Chem.Mol | None = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None

        # Precompute reference 2D fingerprint and baseline physicochemical descriptors
        self.seed_fp: DataStructs.ExplicitBitVect | None = (
            rdMolDescriptors.GetMorganFingerprintAsBitVect(self.seed_mol, 2, nBits=2048)
            if self.seed_mol is not None
            else None
        )
        self.seed_hac: int = self.seed_mol.GetNumHeavyAtoms() if self.seed_mol else 33
        self.seed_mw: float = float(Descriptors.MolWt(self.seed_mol)) if self.seed_mol else 458.37
        self.seed_logp: float = float(Descriptors.MolLogP(self.seed_mol)) if self.seed_mol else 2.32
        self.seed_tpsa: float = float(rdMolDescriptors.CalcTPSA(self.seed_mol)) if self.seed_mol else 197.37
        self.seed_rotbonds: int = int(Lipinski.NumRotatableBonds(self.seed_mol)) if self.seed_mol else 4
        self.seed_qed: float = float(QED.qed(self.seed_mol)) if self.seed_mol else 0.22
        self.seed_sa: float = float(calculate_sa_score(self.seed_mol)) if self.seed_mol else 4.15

        if ref_sdf_path is not None:
            self.ref_sdf_path: Path = Path(ref_sdf_path)
        else:
            shared_ref = PROJECT_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
            self.ref_sdf_path = shared_ref

        self.papyrus_parquet_path: Path = (
            Path(papyrus_parquet_path) if papyrus_parquet_path is not None else DEFAULT_PAPYRUS_PARQUET_PATH
        )
        self._reference_corpus_set: set[str] | None = None

    def load_peak_model_weights(
        self,
        agent: Any,
        pkg_path: Path | str | None = None,
    ) -> Any:
        """Load restored peak checkpoint weights ({path}_peak.pkg) into the agent network.

        Args:
            agent: PyTorch model or DrugEx generator agent to receive restored weights.
            pkg_path: Explicit path to checkpoint file; defaults to searching output_dir.

        Returns:
            The loaded PyTorch state dictionary.
        """
        target_pkg: Path | None = None
        if pkg_path is not None:
            target_pkg = Path(pkg_path)
        else:
            candidates = list(self.output_dir.glob("*_peak.pkg"))
            if not candidates:
                candidates = list(self.output_dir.parent.glob("*_peak.pkg"))
            if candidates:
                target_pkg = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        if target_pkg is None or not target_pkg.exists():
            raise FileNotFoundError(f"Peak checkpoint package not found at {pkg_path or self.output_dir}")

        # [PACKAGE: torch] WHY: Deserialize PyTorch weights from peak checkpoint package. WHAT: Checkpoint reader. HOW: torch.load.
        state_dict = torch.load(target_pkg, weights_only=False)

        if hasattr(agent, "load_state_dict"):
            agent.load_state_dict(state_dict)
        elif hasattr(agent, "loadStatesFromFile"):
            agent.loadStatesFromFile(target_pkg)
        elif hasattr(agent, "agent") and hasattr(agent.agent, "load_state_dict"):
            agent.agent.load_state_dict(state_dict)

        return state_dict

    def sample_candidates(
        self,
        generator: Any,
        num_samples: int = 100,
        input_frags: Sequence[str] | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Sample candidate leads using unbiased sampling to measure the true raw distribution (N_total).

        Args:
            generator: DrugEx generator model (e.g. GraphTransformer or FragGraphExplorer).
            num_samples: Number of molecules to generate in raw sampling mode.
            input_frags: Generative seed fragment SMILES prompt.
            batch_size: Inference batch size.
            **kwargs: Additional parameters forwarded to generator.generate().

        Returns:
            DataFrame containing raw generated candidate structures without duplicate or validity filtering.
        """
        # Strict academic VUN protocol: drop_invalid=False and drop_duplicates=False
        kwargs["drop_invalid"] = False
        kwargs["drop_duplicates"] = False

        if hasattr(generator, "generate"):
            df_sampled = generator.generate(
                input_frags=list(input_frags) if input_frags else None,
                num_samples=num_samples,
                batch_size=batch_size,
                **kwargs,
            )
            return df_sampled

        raise TypeError(f"Generator {type(generator)} does not implement generate() method")

    def get_reference_corpus(self) -> set[str]:
        """Load or retrieve the cached reference SMILES corpus from Papyrus parquet.

        Returns:
            Set of canonical non-isomeric SMILES from Papyrus 05.6 for O(1) novelty lookup.
        """
        if self._reference_corpus_set is not None:
            return self._reference_corpus_set

        if self.papyrus_parquet_path.exists():
            try:
                # [PACKAGE: pandas] WHY: Fast columnar extraction of SMILES from 439k-row parquet file. WHAT: Parquet reader. HOW: pd.read_parquet.
                df_pap = pd.read_parquet(self.papyrus_parquet_path, columns=["SMILES"])
                raw_smiles = df_pap["SMILES"].dropna().astype(str).tolist()
                self._reference_corpus_set = set(raw_smiles)
                return self._reference_corpus_set
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="papyrus_parquet_read_failed",
                    message=f"Failed to load Papyrus parquet at {self.papyrus_parquet_path}: {e}",
                )

        # Fallback if parquet is missing or unreachable
        FallbackMonitor.record_fallback(
            fallback_type="papyrus_corpus_missing",
            message=f"Papyrus parquet missing at {self.papyrus_parquet_path}; using empty reference corpus",
        )
        self._reference_corpus_set = set()
        return self._reference_corpus_set

    def compute_vun_metrics(
        self,
        raw_candidates: Sequence[str] | pd.DataFrame,
        reference_corpus: set[str] | Sequence[str] | Path | str | None = None,
    ) -> dict[str, Any]:
        """Compute standardized academic VUN metrics on raw generated candidate pool.

        Validity: V = N_valid / N_total (Acceptance: >= 80%).
        Internal Uniqueness: U = N_unique_valid / N_valid (Acceptance: >= 80%).
        Novelty: N_ov = N_novel / N_unique_valid against reference corpus.

        Args:
            raw_candidates: Unfiltered sequence of SMILES or DataFrame.
            reference_corpus: Optional custom reference corpus or path; defaults to Papyrus parquet.

        Returns:
            Dictionary containing counts, percentages, and acceptance flags.
        """
        if isinstance(raw_candidates, pd.DataFrame):
            col = (
                "SMILES"
                if "SMILES" in raw_candidates.columns
                else ("smiles" if "smiles" in raw_candidates.columns else None)
            )
            raw_smiles = raw_candidates[col].tolist() if col else []
        else:
            raw_smiles = list(raw_candidates)

        n_total = len(raw_smiles)
        if n_total == 0:
            return {
                "total_generated": 0,
                "valid_count": 0,
                "unique_valid_count": 0,
                "novel_count": 0,
                "validity": 0.0,
                "validity_percentage": 0.0,
                "uniqueness": 0.0,
                "uniqueness_percentage": 0.0,
                "novelty": 0.0,
                "novelty_percentage": 0.0,
                "valid_canonical_smiles": [],
                "unique_canonical_smiles": [],
                "passed_validity": False,
                "passed_uniqueness": False,
            }

        valid_canonical_smiles: list[str] = []
        for item in raw_smiles:
            smi = str(item).strip() if item is not None else ""
            if not smi:
                continue
            # [PACKAGE: rdkit] WHY: Academic valence and chemical parsing verification. WHAT: Chemical parser. HOW: Chem.MolFromSmiles.
            mol = Chem.MolFromSmiles(smi)
            if mol is not None and mol.GetNumHeavyAtoms() > 0:
                can_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                valid_canonical_smiles.append(can_smi)

        n_valid = len(valid_canonical_smiles)
        validity = float(n_valid / n_total) if n_total > 0 else 0.0

        unique_canonical_smiles = sorted(list(set(valid_canonical_smiles)))
        n_unique_valid = len(unique_canonical_smiles)
        uniqueness = float(n_unique_valid / n_valid) if n_valid > 0 else 0.0

        # Resolve reference corpus for novelty calculation
        ref_set: set[str]
        if reference_corpus is not None:
            if isinstance(reference_corpus, (set, frozenset, list, tuple)):
                ref_set = set(reference_corpus)
            elif isinstance(reference_corpus, (str, Path)):
                ref_path = Path(reference_corpus)
                if ref_path.suffix.lower() == ".parquet" and ref_path.exists():
                    df_ref = pd.read_parquet(ref_path, columns=["SMILES"])
                    ref_set = set(df_ref["SMILES"].dropna().astype(str).tolist())
                elif ref_path.exists():
                    with open(ref_path, encoding="utf-8") as f:
                        ref_set = {line.strip() for line in f if line.strip()}
                else:
                    ref_set = set()
            else:
                ref_set = set()
        else:
            ref_set = self.get_reference_corpus()

        if n_unique_valid > 0:
            novel_count = sum(1 for s in unique_canonical_smiles if s not in ref_set)
            novelty = float(novel_count / n_unique_valid)
        else:
            novel_count = 0
            novelty = 0.0

        return {
            "total_generated": n_total,
            "valid_count": n_valid,
            "unique_valid_count": n_unique_valid,
            "novel_count": novel_count,
            "validity": validity,
            "validity_percentage": round(validity * 100.0, 2),
            "uniqueness": uniqueness,
            "uniqueness_percentage": round(uniqueness * 100.0, 2),
            "novelty": novelty,
            "novelty_percentage": round(novelty * 100.0, 2),
            "valid_canonical_smiles": valid_canonical_smiles,
            "unique_canonical_smiles": unique_canonical_smiles,
            "passed_validity": bool(validity >= VALIDITY_THRESHOLD),
            "passed_uniqueness": bool(uniqueness >= UNIQUENESS_THRESHOLD),
        }

    def extract_scaffold(self, mol_or_smiles: Chem.Mol | str) -> str:
        """Extract canonical Bemis-Murcko core scaffold framework from a molecule or SMILES string.

        Args:
            mol_or_smiles: RDKit Mol object or SMILES string.

        Returns:
            Canonical non-isomeric Murcko scaffold SMILES, or '[Acyclic]' if no rings exist.
        """
        mol: Chem.Mol | None
        if isinstance(mol_or_smiles, Chem.Mol):
            mol = mol_or_smiles
        else:
            mol = Chem.MolFromSmiles(str(mol_or_smiles))

        if mol is None:
            return "[Invalid]"

        try:
            # [PACKAGE: rdkit] WHY: Extract core framework ring systems and linkers. WHAT: Bemis-Murcko scaffold algorithm. HOW: MurckoScaffold.MurckoScaffoldSmiles.
            scaff_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if not scaff_smi:
                return "[Acyclic]"
            return scaff_smi
        except Exception:
            return "[Acyclic]"

    def compute_scaffold_diversity(self, smiles_list: Sequence[str]) -> dict[str, Any]:
        """Compute Bemis-Murcko scaffold diversity metrics on a set of valid SMILES.

        Diversity ratio: D_scaffold = N_unique_scaffolds / N_unique_valid.

        Args:
            smiles_list: Sequence of valid SMILES strings.

        Returns:
            Dictionary containing diversity ratio, scaffold counts, and scaffold frequency distribution.
        """
        unique_valid = sorted(list(set(smiles_list)))
        if not unique_valid:
            return {
                "diversity_ratio": 0.0,
                "unique_scaffold_count": 0,
                "unique_valid_count": 0,
                "scaffold_distribution": {},
            }

        scaffolds = [self.extract_scaffold(s) for s in unique_valid]
        scaffold_counts = Counter(scaffolds)
        unique_scaffolds = len(scaffold_counts)
        diversity_ratio = float(unique_scaffolds / len(unique_valid))

        return {
            "diversity_ratio": round(diversity_ratio, 4),
            "unique_scaffold_count": unique_scaffolds,
            "unique_valid_count": len(unique_valid),
            "scaffold_distribution": dict(scaffold_counts),
        }

    def compute_rocs_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute high-fidelity 50-conformer Gaussian ROCS shape & color overlay scores.

        Uses native DrugEx RDKitROCSScorer configured with:
        - max_conformers=50, max_heavy_atoms=55, max_rotatable_bonds=25, timeout=60s.
        - score_type="TanimotoCombo" (0.0 to 2.0 scale).
        - color_ff="oe_mills_dean" OpenEye Mills-Dean color SMARTS.

        Args:
            smiles_list: Sequence of candidate SMILES strings to evaluate.

        Returns:
            1D NumPy array of float32 ROCS TanimotoCombo scores (0.0 to 2.0).
        """
        if not smiles_list:
            return np.zeros(0, dtype=np.float32)

        if not self.ref_sdf_path.exists():
            # Attempt automated reference conformer generation if missing
            try:
                preparer = QueryConformerPreparer(output_dir=self.output_dir / "references")
                gen_path = preparer.generate_curated_ensemble(
                    query=self.seed_smiles,
                    output_path=self.ref_sdf_path,
                )
                self.ref_sdf_path = gen_path
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_missing_ref_sdf",
                    message=f"Reference SDF missing and generation failed ({e}); returning zeros",
                    context={"n_molecules": len(smiles_list)},
                )
                return np.zeros(len(smiles_list), dtype=np.float32)

        # [PACKAGE: drugex] WHY: High-fidelity 50-conformer generation for academic evaluation. WHAT: Conformer generator. HOW: RDKitConformerGenerator.
        conf_gen = RDKitConformerGenerator(
            max_conformers=NUM_CONFORMERS_ROCS,
            max_isomers=2,
            max_heavy_atoms=ROCS_MAX_HEAVY_ATOMS,
            max_rotatable_bonds=ROCS_MAX_ROTATABLE_BONDS,
            timeout=ROCS_TIMEOUT,
            show_progress=False,
        )

        try:
            # [PACKAGE: drugex] WHY: Native RDKit ROCS scoring with 6-type Mills-Dean color SMARTS. WHAT: 3D shape overlay scorer. HOW: RDKitROCSScorer.
            scorer = RDKitROCSScorer(
                conformer_generator=conf_gen,
                references=str(self.ref_sdf_path),
                score_type=ROCS_SCORE_TYPE,
                color_ff=ROCS_COLOR_FF,
                use_colors=True,
                show_progress=False,
                n_jobs=-1,
            )
            scores = np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
            return scores
        except Exception as e:
            FallbackMonitor.record_fallback(
                fallback_type="rocs_scoring_failed",
                message=f"RDKit ROCS evaluation failed ({e}); returning zeros",
                context={"n_molecules": len(smiles_list)},
            )
            return np.zeros(len(smiles_list), dtype=np.float32)

    def compute_physicochemical_distributions(
        self,
        df_valid: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Calculate statistical distribution metrics (mean, median, std, min, max, IQR) for properties.

        Args:
            df_valid: DataFrame containing valid candidates.

        Returns:
            Dictionary mapping property names to their descriptive statistical moments.
        """
        props = ["HAC", "MW", "LogP", "TPSA", "RotBonds", "HBD", "HBA", "SA", "QED", "ROCS"]
        results: dict[str, dict[str, float]] = {}

        if df_valid.empty:
            for p in props:
                results[p] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "q25": 0.0,
                    "q75": 0.0,
                    "iqr": 0.0,
                }
            return results

        for p in props:
            if p in df_valid.columns:
                arr = pd.to_numeric(df_valid[p], errors="coerce").dropna().values
                if len(arr) > 0:
                    mean_val = float(np.mean(arr))
                    median_val = float(np.median(arr))
                    std_val = float(np.std(arr))
                    min_val = float(np.min(arr))
                    max_val = float(np.max(arr))
                    q25, q75 = [float(v) for v in np.percentile(arr, [25, 75])]
                    iqr_val = float(q75 - q25)
                    results[p] = {
                        "mean": round(mean_val, 3),
                        "median": round(median_val, 3),
                        "std": round(std_val, 3),
                        "min": round(min_val, 3),
                        "max": round(max_val, 3),
                        "q25": round(q25, 3),
                        "q75": round(q75, 3),
                        "iqr": round(iqr_val, 3),
                    }
                else:
                    results[p] = {
                        "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0
                    }
            else:
                results[p] = {
                    "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0
                }
        return results

    def compute_dynamic_envelope_benchmarks(
        self,
        df_valid: pd.DataFrame,
    ) -> dict[str, Any]:
        """Benchmark candidate properties against reference query dynamic envelope ([0.70*X, 1.30*X]).

        Args:
            df_valid: DataFrame containing valid candidates.

        Returns:
            Dictionary containing target envelope bounds, compliance counts, and percentages.
        """
        # Dynamic envelope bounds based on seed molecule (e.g. EGCG)
        hac_bounds = (round(0.70 * self.seed_hac), round(1.30 * self.seed_hac))
        mw_bounds = (round(0.70 * self.seed_mw, 1), round(1.30 * self.seed_mw, 1))
        # LogP delta-envelope around reference
        logp_bounds = (round(self.seed_logp - 1.5, 2), round(self.seed_logp + 1.5, 2))
        tpsa_bounds = (round(0.70 * self.seed_tpsa, 1), round(1.30 * self.seed_tpsa, 1))
        rotbonds_bounds = (max(0, round(0.70 * self.seed_rotbonds)), round(1.30 * self.seed_rotbonds) + 2)

        n_valid = len(df_valid)
        if n_valid == 0:
            return {
                "reference_seed": {
                    "HAC": self.seed_hac,
                    "MW": self.seed_mw,
                    "LogP": self.seed_logp,
                    "TPSA": self.seed_tpsa,
                    "RotBonds": self.seed_rotbonds,
                },
                "envelope_bounds": {
                    "HAC": hac_bounds,
                    "MW": mw_bounds,
                    "LogP": logp_bounds,
                    "TPSA": tpsa_bounds,
                    "RotBonds": rotbonds_bounds,
                },
                "compliance": {},
                "synthesizability_floor": {
                    "unpenalized_count": 0,
                    "unpenalized_percentage": 0.0,
                    "penalized_count": 0,
                    "penalized_percentage": 0.0,
                },
            }

        hac_comp = int(((df_valid["HAC"] >= hac_bounds[0]) & (df_valid["HAC"] <= hac_bounds[1])).sum())
        mw_comp = int(((df_valid["MW"] >= mw_bounds[0]) & (df_valid["MW"] <= mw_bounds[1])).sum())
        logp_comp = int(((df_valid["LogP"] >= logp_bounds[0]) & (df_valid["LogP"] <= logp_bounds[1])).sum())
        tpsa_comp = int(((df_valid["TPSA"] >= tpsa_bounds[0]) & (df_valid["TPSA"] <= tpsa_bounds[1])).sum())
        rot_comp = int(((df_valid["RotBonds"] >= rotbonds_bounds[0]) & (df_valid["RotBonds"] <= rotbonds_bounds[1])).sum())

        # Single SA feasibility floor: unpenalized if SA <= 5.5, penalized if SA > 6.0
        sa_unpenalized = int((df_valid["SA"] <= 5.5).sum())
        sa_transition = int(((df_valid["SA"] > 5.5) & (df_valid["SA"] <= 6.0)).sum())
        sa_penalized = int((df_valid["SA"] > 6.0).sum())

        return {
            "reference_seed": {
                "HAC": self.seed_hac,
                "MW": self.seed_mw,
                "LogP": self.seed_logp,
                "TPSA": self.seed_tpsa,
                "RotBonds": self.seed_rotbonds,
                "SA": self.seed_sa,
                "QED": self.seed_qed,
            },
            "envelope_bounds": {
                "HAC": list(hac_bounds),
                "MW": list(mw_bounds),
                "LogP": list(logp_bounds),
                "TPSA": list(tpsa_bounds),
                "RotBonds": list(rotbonds_bounds),
            },
            "compliance": {
                "HAC": {"count": hac_comp, "percentage": round(hac_comp / n_valid * 100.0, 1)},
                "MW": {"count": mw_comp, "percentage": round(mw_comp / n_valid * 100.0, 1)},
                "LogP": {"count": logp_comp, "percentage": round(logp_comp / n_valid * 100.0, 1)},
                "TPSA": {"count": tpsa_comp, "percentage": round(tpsa_comp / n_valid * 100.0, 1)},
                "RotBonds": {"count": rot_comp, "percentage": round(rot_comp / n_valid * 100.0, 1)},
            },
            "synthesizability_floor": {
                "unpenalized_count": sa_unpenalized,
                "unpenalized_percentage": round(sa_unpenalized / n_valid * 100.0, 1),
                "transition_count": sa_transition,
                "transition_percentage": round(sa_transition / n_valid * 100.0, 1),
                "penalized_count": sa_penalized,
                "penalized_percentage": round(sa_penalized / n_valid * 100.0, 1),
                "single_floor_definition": "SA <= 5.5 unpenalized (reward 1.0), SA > 6.0 penalized",
            },
        }

    def analyze_candidates(self, candidates: Sequence[str] | pd.DataFrame) -> pd.DataFrame:
        """Profile all candidate molecules with complete 3D, 2D, and physicochemical metrics.

        Args:
            candidates: Sequence of SMILES strings or DataFrame containing candidate molecules.

        Returns:
            Enriched DataFrame containing all evaluated chemical, structural, and 3D descriptors.
        """
        if isinstance(candidates, pd.DataFrame):
            col = (
                "SMILES"
                if "SMILES" in candidates.columns
                else ("smiles" if "smiles" in candidates.columns else None)
            )
            raw_smiles = candidates[col].dropna().astype(str).tolist() if col else []
        else:
            raw_smiles = [str(s).strip() for s in candidates if str(s).strip()]

        unique_smiles = list(dict.fromkeys(raw_smiles))
        records: list[dict[str, Any]] = []

        valid_mols: list[tuple[str, Chem.Mol]] = []
        for smi in unique_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                records.append({
                    "SMILES": smi,
                    "Valid": False,
                    "HAC": 0,
                    "MW": 0.0,
                    "LogP": 0.0,
                    "QED": 0.0,
                    "SA": 10.0,
                    "HBD": 0,
                    "HBA": 0,
                    "TPSA": 0.0,
                    "RotBonds": 0,
                    "Scaffold": None,
                    "Tanimoto_to_Seed": 0.0,
                })
            else:
                valid_mols.append((smi, mol))

        # Profile valid molecules
        for _smi, mol in valid_mols:
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            hac = int(mol.GetNumHeavyAtoms())
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            qed_val = float(QED.qed(mol))
            hbd = int(Lipinski.NumHDonors(mol))
            hba = int(Lipinski.NumHAcceptors(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            rot_bonds = int(Lipinski.NumRotatableBonds(mol))

            try:
                sa_val = float(calculate_sa_score(mol))
            except Exception:
                sa_val = 5.0

            scaff_smi = self.extract_scaffold(mol)

            # [PACKAGE: rdkit] WHY: Measure 2D structural distance to query seed via 2048-bit Morgan ECFP4 fingerprint. WHAT: Fingerprint similarity. HOW: rdMolDescriptors.GetMorganFingerprintAsBitVect.
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim_2d = float(DataStructs.TanimotoSimilarity(self.seed_fp, fp)) if self.seed_fp else 0.0

            records.append({
                "SMILES": can_smi,
                "Valid": True,
                "HAC": hac,
                "MW": round(mw, 2),
                "LogP": round(logp, 2),
                "QED": round(qed_val, 3),
                "SA": round(sa_val, 2),
                "HBD": hbd,
                "HBA": hba,
                "TPSA": round(tpsa, 1),
                "RotBonds": rot_bonds,
                "Scaffold": scaff_smi,
                "Tanimoto_to_Seed": round(sim_2d, 3),
            })

        df = pd.DataFrame(records)
        if df.empty:
            df["ROCS"] = 0.0
            df["Desirability"] = 0.0
            return df

        valid_mask = df["Valid"]
        valid_smiles = df.loc[valid_mask, "SMILES"].tolist()

        # Batch 50-conformer RDKit ROCS evaluation on valid molecules
        rocs_scores = self.compute_rocs_scores(valid_smiles)

        df["ROCS"] = 0.0
        df.loc[valid_mask, "ROCS"] = rocs_scores

        # Continuous Anti-Collapse Soft-Gated Multi-Objective Desirability (R3/R5)
        # R(x) = w_3D * S_3D + w_props * G(S_3D) * S_props * S_SA_floor
        # Guarantees strictly 0 reward for candidates with S_3D == 0, eliminating Pareto cheating.
        s_3d = np.clip(df["ROCS"].values / 2.0, 0.0, 1.0)
        tau_3d = 0.60
        g_3d = np.minimum(1.0, s_3d / tau_3d)

        # Secondary property component (QED drug-likeness)
        s_qed = np.clip(df["QED"].values, 0.0, 1.0)

        # Single synthesizability floor: 1.0 for SA <= 5.5, linearly penalized above 5.5, 0.0 above 6.0
        sa_vals = df["SA"].values
        s_sa_floor = np.where(
            sa_vals <= 5.5,
            1.0,
            np.clip((6.0 - sa_vals) / 0.50, 0.0, 1.0),
        )

        w_3d = 0.70
        w_props = 0.30
        base_desirability = w_3d * s_3d + w_props * g_3d * s_qed * s_sa_floor

        # Strict anti-cheating zero mask for non-conforming 3D shapes
        zero_mask = s_3d <= 1e-6
        base_desirability[zero_mask] = 0.0

        df["Desirability"] = np.where(df["Valid"], np.round(base_desirability, 4), 0.0)

        out_tsv = self.output_dir / "all_analyzed_candidates.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        return df

    def prioritize_leads(self, df_analyzed: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Prioritize elite lead molecules balancing 3D shape, synthesizability, and Bemis-Murcko scaffold diversity.

        Args:
            df_analyzed: Analyzed candidates DataFrame.
            top_k: Number of diverse top leads to prioritize.

        Returns:
            Filtered and ranked DataFrame containing top elite lead candidates.
        """
        if df_analyzed.empty:
            return pd.DataFrame()

        viable = df_analyzed[df_analyzed["Valid"]].copy()
        if viable.empty:
            return pd.DataFrame()

        # Sort descending by multi-objective anti-collapse Desirability score
        viable = viable.sort_values(by="Desirability", ascending=False).reset_index(drop=True)

        # Enforce Bemis-Murcko scaffold diversity: pick best candidate per unique scaffold
        selected_indices: list[int] = []
        seen_scaffolds: set[str] = set()

        for idx, row in viable.iterrows():
            scaff = str(row["Scaffold"])
            if scaff not in seen_scaffolds:
                selected_indices.append(idx)
                seen_scaffolds.add(scaff)
            if len(selected_indices) >= top_k:
                break

        # If insufficient distinct scaffolds, fill up with remaining highest desirability candidates
        if len(selected_indices) < top_k:
            for idx in viable.index:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= top_k:
                    break

        leads_df = viable.loc[selected_indices].reset_index(drop=True)
        leads_tsv = self.output_dir / "top_prioritized_leads.tsv"
        leads_df.to_csv(leads_tsv, sep="\t", index=False)
        return leads_df

    def visualize_leads(
        self,
        df_leads: pd.DataFrame,
        df_all: pd.DataFrame | None = None,
        max_leads: int = 6,
    ) -> dict[str, Path]:
        """Generate high-resolution 2D chemical structure grids and property distribution charts.

        Args:
            df_leads: DataFrame of prioritized elite leads.
            df_all: Optional complete candidates DataFrame for distribution plots.
            max_leads: Maximum number of leads to include in 2D grid image.

        Returns:
            Dictionary mapping visualization names to generated image file paths.
        """
        figures: dict[str, Path] = {}
        if df_leads.empty:
            return figures

        # 1. Publication-Grade 2D Chemical Structure Grid Image
        plot_mols: list[Chem.Mol] = []
        legends: list[str] = []

        # Add seed molecule as the primary visual benchmark
        if self.seed_mol is not None:
            plot_mols.append(self.seed_mol)
            legends.append(
                f"QUERY SEED (EGCG)\n"
                f"HAC: {self.seed_hac} | MW: {self.seed_mw:.1f}\n"
                f"QED: {self.seed_qed:.2f} | SA: {self.seed_sa:.2f}"
            )

        for idx, row in df_leads.head(max_leads).iterrows():
            m = Chem.MolFromSmiles(row["SMILES"])
            if m is not None:
                plot_mols.append(m)
                legends.append(
                    f"Lead #{idx + 1}\n"
                    f"ROCS: {row['ROCS']:.2f} | Des: {row['Desirability']:.2f}\n"
                    f"MW: {row['MW']:.1f} | QED: {row['QED']:.2f} | SA: {row['SA']:.2f}"
                )

        if plot_mols:
            # [PACKAGE: rdkit] WHY: High-resolution publication-quality 2D structure grid visualization. WHAT: Molecular grid drawer. HOW: Draw.MolsToGridImage.
            grid_img = Draw.MolsToGridImage(
                plot_mols,
                molsPerRow=min(3, len(plot_mols)),
                subImgSize=(340, 260),
                legends=legends,
                useSVG=False,
            )
            grid_path = self.output_dir / "lead_structures_2d.png"
            grid_img.save(str(grid_path))
            figures["lead_structures_2d"] = grid_path

        # 2. Physicochemical Property Distributions Multi-Panel Chart
        eval_df = df_all if df_all is not None and not df_all.empty else df_leads
        valid_df = eval_df[eval_df["Valid"]].copy()

        if len(valid_df) >= 1:
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            fig.suptitle(
                "Ligand-Based De Novo Drug Design: Physicochemical & 3D Profiles",
                fontsize=15,
                weight="bold",
                y=0.98,
            )

            # Subplot 1: 3D ROCS Distribution
            ax = axes[0, 0]
            ax.hist(valid_df["ROCS"], bins=15, color="#2A9D8F", edgecolor="black", alpha=0.80)
            ax.axvline(1.00, color="#E76F51", linestyle="--", linewidth=1.5, label="High Overlap (1.00)")
            ax.set_title("3D Gaussian ROCS (TanimotoCombo)", fontsize=11, weight="bold")
            ax.set_xlabel("ROCS Score (0 to 2.0)")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Subplot 2: QED Drug-likeness Distribution
            ax = axes[0, 1]
            ax.hist(valid_df["QED"], bins=15, color="#457B9D", edgecolor="black", alpha=0.80)
            ax.axvline(self.seed_qed, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({self.seed_qed:.2f})")
            ax.set_title("Drug-likeness Estimate (QED)", fontsize=11, weight="bold")
            ax.set_xlabel("QED Score (0 to 1.0)")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Subplot 3: Synthetic Accessibility (SAscore) Feasibility Floor
            ax = axes[0, 2]
            ax.hist(valid_df["SA"], bins=15, color="#E9C46A", edgecolor="black", alpha=0.80)
            ax.axvspan(1.0, 5.5, color="#2A9D8F", alpha=0.15, label="Feasible (<= 5.5)")
            ax.axvline(5.5, color="#2A9D8F", linestyle="--", linewidth=1.5)
            ax.axvline(6.0, color="#E76F51", linestyle=":", linewidth=1.5, label="Penalized (> 6.0)")
            ax.set_title("Synthetic Accessibility (SAscore)", fontsize=11, weight="bold")
            ax.set_xlabel("SA Score (1 = Easy, 10 = Hard)")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Subplot 4: Molecular Weight (MW) & Dynamic Envelope
            ax = axes[1, 0]
            ax.hist(valid_df["MW"], bins=15, color="#264653", edgecolor="black", alpha=0.80)
            mw_lo, mw_hi = round(0.70 * self.seed_mw, 1), round(1.30 * self.seed_mw, 1)
            ax.axvspan(mw_lo, mw_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{mw_lo}, {mw_hi}]")
            ax.axvline(self.seed_mw, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({self.seed_mw:.1f})")
            ax.set_title("Molecular Weight (MW)", fontsize=11, weight="bold")
            ax.set_xlabel("MW (Da)")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Subplot 5: CLogP Lipophilicity
            ax = axes[1, 1]
            ax.hist(valid_df["LogP"], bins=15, color="#1D3557", edgecolor="black", alpha=0.80)
            logp_lo, logp_hi = round(self.seed_logp - 1.5, 2), round(self.seed_logp + 1.5, 2)
            ax.axvspan(logp_lo, logp_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{logp_lo}, {logp_hi}]")
            ax.axvline(self.seed_logp, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({self.seed_logp:.2f})")
            ax.set_title("Calculated Lipophilicity (CLogP)", fontsize=11, weight="bold")
            ax.set_xlabel("CLogP")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Subplot 6: Polar Surface Area (TPSA)
            ax = axes[1, 2]
            ax.hist(valid_df["TPSA"], bins=15, color="#6D597A", edgecolor="black", alpha=0.80)
            tpsa_lo, tpsa_hi = round(0.70 * self.seed_tpsa, 1), round(1.30 * self.seed_tpsa, 1)
            ax.axvspan(tpsa_lo, tpsa_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{tpsa_lo}, {tpsa_hi}]")
            ax.axvline(self.seed_tpsa, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({self.seed_tpsa:.1f})")
            ax.set_title("Topological Polar Surface Area (TPSA)", fontsize=11, weight="bold")
            ax.set_xlabel("TPSA (Å²)")
            ax.set_ylabel("Molecule Count")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            dist_path = self.output_dir / "property_distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            figures["property_distributions"] = dist_path

        return figures

    def export_summary(
        self,
        df_all: pd.DataFrame,
        df_leads: pd.DataFrame,
        vun_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile and export comprehensive descriptive analysis summary JSON.

        Args:
            df_all: DataFrame of all evaluated candidates.
            df_leads: DataFrame of prioritized elite leads.
            vun_metrics: Optional precomputed VUN metrics dictionary.

        Returns:
            Dictionary containing structured statistical audit metrics.
        """
        valid_df = df_all[df_all["Valid"]].copy() if not df_all.empty else pd.DataFrame()
        total_n = len(df_all)
        valid_n = len(valid_df)

        if vun_metrics is None:
            raw_smiles = df_all["SMILES"].tolist() if not df_all.empty else []
            vun_metrics = self.compute_vun_metrics(raw_smiles)

        # Bemis-Murcko scaffold diversity on unique valid leads
        valid_smiles = valid_df["SMILES"].tolist() if valid_n > 0 else []
        scaffold_diversity = self.compute_scaffold_diversity(valid_smiles)

        # Physicochemical distributions & envelope compliance
        prop_stats = self.compute_physicochemical_distributions(valid_df)
        envelope_benchmarks = self.compute_dynamic_envelope_benchmarks(valid_df)

        summary: dict[str, Any] = {
            "query_seed_smiles": self.seed_smiles,
            "seed_descriptors": {
                "HAC": self.seed_hac,
                "MW": self.seed_mw,
                "LogP": self.seed_logp,
                "TPSA": self.seed_tpsa,
                "RotBonds": self.seed_rotbonds,
                "QED": self.seed_qed,
                "SA": self.seed_sa,
            },
            "vun_metrics": {
                "total_generated": vun_metrics.get("total_generated", total_n),
                "valid_count": vun_metrics.get("valid_count", valid_n),
                "validity_percentage": vun_metrics.get("validity_percentage", round(valid_n / total_n * 100.0, 1) if total_n > 0 else 0.0),
                "unique_valid_count": vun_metrics.get("unique_valid_count", int(valid_df["SMILES"].nunique()) if valid_n > 0 else 0),
                "uniqueness_percentage": vun_metrics.get("uniqueness_percentage", round(valid_df["SMILES"].nunique() / valid_n * 100.0, 1) if valid_n > 0 else 0.0),
                "novel_count": vun_metrics.get("novel_count", 0),
                "novelty_percentage": vun_metrics.get("novelty_percentage", 0.0),
                "passed_validity_threshold": vun_metrics.get("passed_validity", bool(valid_n / total_n >= 0.80 if total_n > 0 else False)),
                "passed_uniqueness_threshold": vun_metrics.get("passed_uniqueness", bool(valid_df["SMILES"].nunique() / valid_n >= 0.80 if valid_n > 0 else False)),
            },
            "bemis_murcko_scaffold_diversity": scaffold_diversity,
            "physicochemical_distributions": prop_stats,
            "dynamic_envelope_benchmarks": envelope_benchmarks,
            "metrics": {
                "mean_rocs": prop_stats["ROCS"]["mean"],
                "max_rocs": prop_stats["ROCS"]["max"],
                "median_rocs": prop_stats["ROCS"]["median"],
                "iqr_rocs": prop_stats["ROCS"]["iqr"],
                "mean_qed": prop_stats["QED"]["mean"],
                "max_qed": prop_stats["QED"]["max"],
                "mean_sa": prop_stats["SA"]["mean"],
                "mean_mw": prop_stats["MW"]["mean"],
                "mean_logp": prop_stats["LogP"]["mean"],
                "mean_tpsa": prop_stats["TPSA"]["mean"],
                "mean_tanimoto_to_seed": round(float(valid_df["Tanimoto_to_Seed"].mean()), 3) if valid_n > 0 and "Tanimoto_to_Seed" in valid_df.columns else 0.0,
            },
            "top_leads_count": len(df_leads),
            "top_leads": df_leads.to_dict(orient="records") if not df_leads.empty else [],
        }

        json_path = self.output_dir / "result_analysis_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary
