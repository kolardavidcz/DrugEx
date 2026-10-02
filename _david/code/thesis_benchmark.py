"""Academic evaluation and benchmark suite for Ligand-Based De Novo Drug Design."""
from __future__ import annotations

import argparse
import sys
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Descriptors, Draw, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
    CDPKIT_AVAILABLE: bool = True
except (ImportError, Exception):
    CDPKIT_AVAILABLE = False
try:
    from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
    RDKIT_ROCS_AVAILABLE: bool = True
except (ImportError, Exception):
    RDKIT_ROCS_AVAILABLE = False

from _david.code.fallback_monitor import FallbackMonitor

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    FallbackMonitor.record_fallback(
        fallback_type="sklearn_pca_fallback",
        message="sklearn.decomposition.PCA not found; falling back to lightweight SVD PCA",
    )
    class PCA:
        """Lightweight SVD-based PCA fallback when scikit-learn is not installed in the environment."""
        def __init__(self, n_components: int = 2, random_state: int = 42) -> None:
            self.n_components: int = n_components
            self.explained_variance_ratio_: list[float] = [0.5, 0.3]

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            X_centered = X - np.mean(X, axis=0)
            u, s, _ = np.linalg.svd(X_centered, full_matrices=False)
            explained_variance = (s ** 2) / max(1, (len(X) - 1))
            total_var = explained_variance.sum()
            if total_var > 0:
                self.explained_variance_ratio_ = (explained_variance / total_var)[:self.n_components].tolist()
            return u[:, :self.n_components] * s[:self.n_components]


# Load SAscore directly from DrugEx
try:
    from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
except Exception as e:
    FallbackMonitor.record_fallback(
        fallback_type="sascorer_import_fallback",
        message=f"DrugEx calculateScore import failed ({e}); falling back to default 3.5",
    )

    def calculate_sa_score(mol: Any) -> float:
        return 3.5

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
ROCS_THRESHOLD: float = 0.50
USRCAT_THRESHOLD: float = 0.22
RA_THRESHOLD: float = 0.50
QED_THRESHOLD: float = 0.40
SA_THRESHOLD: float = 4.0
RANDOM_STATE: int = 42

COLOR_REF: str = "#E63946"
COLOR_RNN: str = "#457B9D"
COLOR_GRAPH: str = "#E76F51"
COLOR_FUNNEL: str = "#2A9D8F"


class ThesisBenchmark:
    """Academic evaluation and benchmark suite for Ligand-Based De Novo Drug Design."""

    def __init__(
        self,
        output_dir: Path | None = None,
        ref_sdf: Path | str | None = None,
        seed_smiles: str = MY_MOLECULE,
    ) -> None:
        self.ROOT_PATH: Path = PROJECT_ROOT
        self.BASE_OUTPUT_DIR: Path = output_dir or (self.ROOT_PATH / "_david" / "output" / "benchmark")
        self.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.seed_smiles: str = seed_smiles
        #? [RDKit] Construct seed molecule graph from canonical SMILES string.
        self.SEED_MOL: Chem.Mol | None = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None
        #? [RDKit Morgan Fingerprint] Pre-compute 2048-bit ECFP4 bit vector of seed for Tanimoto distance auditing.
        self.SEED_FP: DataStructs.ExplicitBitVect | None = (
            AllChem.GetMorganFingerprintAsBitVect(self.SEED_MOL, 2, nBits=2048) if self.SEED_MOL else None
        )
        # Dynamic heavy-atom bounds scaled relative to query active
        self.seed_hac: int = int(self.SEED_MOL.GetNumHeavyAtoms()) if self.SEED_MOL else 33
        self.min_heavy_atoms: int = max(6, int(self.seed_hac * 0.50))
        self.max_heavy_atoms: int = int(self.seed_hac * 1.50)

        if ref_sdf is not None:
            self.REF_SDF: Path = Path(ref_sdf)
        else:
            seed_hash = None
            if self.seed_smiles:
                try:
                    from _david.code.prepare_query_conformer import get_seed_hash
                    seed_hash = get_seed_hash(self.seed_smiles)
                except Exception:
                    pass
            candidates: list[Path] = []
            if seed_hash:
                candidates.extend([
                    self.BASE_OUTPUT_DIR.parent / "references" / f"seed_{seed_hash}_3d.sdf",
                    self.ROOT_PATH / "_david" / "references" / f"seed_{seed_hash}_3d.sdf",
                ])
            candidates.extend([
                self.BASE_OUTPUT_DIR.parent / "references" / "query_molecule_3d.sdf",
                self.ROOT_PATH / "_david" / "references" / "query_molecule_3d.sdf",
                self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf",
            ])
            self.REF_SDF = next((p for p in candidates if p.exists()), candidates[-1])

    def compute_molecular_properties(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        """Compute real physicochemical and cheminformatics descriptors for candidate SMILES."""
        records: list[dict[str, Any]] = []
        for smi in smiles_list:
            smi_str = str(smi).strip()
            mol = Chem.MolFromSmiles(smi_str)
            if mol is None:
                records.append({
                    "SMILES": smi_str,
                    "Valid": 0,
                    "QED": 0.0,
                    "SA": 10.0,
                    "MW": 0.0,
                    "HeavyAtoms": 0,
                    "LogP": 0.0,
                    "Tanimoto_to_Seed": 0.0,
                    "Murcko_Scaffold": None,
                })
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = float(DataStructs.TanimotoSimilarity(self.SEED_FP, fp)) if self.SEED_FP else 0.0

            try:
                scaff = MurckoScaffold.GetScaffoldForMol(mol)
                scaff_smi = Chem.MolToSmiles(scaff) if scaff.GetNumHeavyAtoms() > 0 else None
            except Exception:
                scaff_smi = None

            try:
                sa = float(calculate_sa_score(mol))
            except Exception:
                FallbackMonitor.record_fallback(
                    fallback_type="sascorer_calc_fallback",
                    message="calculate_sa_score failed on molecule; falling back to 5.0",
                    context={"smiles": smi_str},
                )
                sa = 5.0

            records.append({
                "SMILES": smi_str,
                "Valid": 1,
                "QED": float(QED.qed(mol)),
                "SA": sa,
                "MW": float(Descriptors.MolWt(mol)),
                "HeavyAtoms": int(Lipinski.HeavyAtomCount(mol)),
                "LogP": float(Descriptors.MolLogP(mol)),
                "Tanimoto_to_Seed": sim,
                "Murcko_Scaffold": scaff_smi,
            })

        return pd.DataFrame(records)

    def load_real_datasets(
        self,
        rnn_path: Path | None = None,
        graph_path: Path | None = None,
        funnel_path: Path | None = None,
        leads_path: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load real empirical candidate datasets without any synthetic simulation."""
        data: dict[str, pd.DataFrame] = {}
        target_leads_path = leads_path or funnel_path

        # 1. Sequence-RNN (Regime 1: Unconstrained SMILES RL)
        df_rnn = None
        if rnn_path and Path(rnn_path).exists():
            df_rnn = self.prepare_regime_dataframe(pd.read_csv(rnn_path, sep="\t"))
        else:
            found = self._find_latest_run_file("stage1_sequence_rnn", "stage1_hits.tsv")
            if found is None:
                found = self._find_latest_run_file("stage1_sequence_rnn", "rl_generated_candidates_scored.tsv")
            if found is None:
                found = self._find_latest_pipeline_stage("stage1_sequence_rnn")
            if found is not None:
                df_rnn = self.prepare_regime_dataframe(found)

        if df_rnn is not None and len(df_rnn) > 0:
            data["Sequence-RNN (Pure RL)"] = df_rnn

        # 2. Graph-Transformer (Regime 2: Direct Seed Scaffold RL)
        df_graph = None
        if graph_path and Path(graph_path).exists():
            df_graph = self.prepare_regime_dataframe(pd.read_csv(graph_path, sep="\t"))
        else:
            found = self._find_latest_run_file("stage2_graph_transformer", "stage2_leads.tsv")
            if found is None:
                found = self._find_latest_run_file("stage2_graph_transformer", "rl_generated_candidates_scored.tsv")
            if found is None:
                found = self._find_latest_run_file("stage3_graph_transformer", "stage3_leads.tsv")
            if found is None:
                found = self._find_latest_pipeline_stage("stage2_graph_transformer")
            if found is None:
                found = self._find_latest_pipeline_stage("stage3_graph_transformer")
            if found is not None:
                df_graph = self.prepare_regime_dataframe(found)

        if df_graph is not None and len(df_graph) > 0:
            data["Graph-Transformer (Scaffold RL)"] = df_graph

        # 3. Prioritized Leads (Post-Generation Lead Selection)
        df_leads = None
        if target_leads_path and Path(target_leads_path).exists():
            df_leads = self.prepare_regime_dataframe(pd.read_csv(target_leads_path, sep="\t"))
        else:
            found = self._find_latest_run_file("stage3_lead_selection", "selected_leads.tsv")
            if found is None:
                found = self._find_latest_run_file("stage2_bemis_murcko", "funnel_scaffolds.tsv")
            if found is None:
                found = self._find_latest_pipeline_stage("stage3_lead_selection")
            if found is None:
                found = self._find_latest_pipeline_stage("stage2_bemis_murcko")
            if found is not None:
                df_leads = self.prepare_regime_dataframe(found)

        if df_leads is not None and len(df_leads) > 0:
            is_duplicate = False
            for _, existing_df in list(data.items()):
                if (
                    len(df_leads) == len(existing_df)
                    and "SMILES" in df_leads.columns
                    and "SMILES" in existing_df.columns
                    and df_leads["SMILES"].equals(existing_df["SMILES"])
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                data["Prioritized Leads (Top Desirability)"] = df_leads

        if not data:
            raise FileNotFoundError(
                "No real candidate datasets found to benchmark! Please provide valid TSV paths (--rnn-file, --graph-file, or --leads-file), "
                "or run the pipeline to generate real candidate files. Synthetic mock simulation has been eliminated."
            )

        return data

    def _find_latest_run_file(self, folder_name: str, file_name: str) -> pd.DataFrame | None:
        """Search directory hierarchy for the most recent matching TSV file."""
        search_roots = [
            self.ROOT_PATH / "_david" / "output",
            self.ROOT_PATH / "_david" / "outputs",
            self.ROOT_PATH / "_david" / folder_name,
        ]
        all_candidates: list[Path] = []
        for root in search_roots:
            if root.exists():
                all_candidates.extend(list(root.rglob(file_name)))

        if not all_candidates:
            return None
        all_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in all_candidates:
            try:
                df = pd.read_csv(cand, sep="\t")
                if len(df) > 0 and ("SMILES" in df.columns or "smiles" in df.columns):
                    return df
            except Exception:
                continue
        return None

    def _find_latest_pipeline_stage(self, stage_folder: str) -> pd.DataFrame | None:
        """Search output/ and outputs/pipeline_runs/ for the most recent stage output file."""
        roots = [
            self.ROOT_PATH / "_david" / "output",
            self.ROOT_PATH / "_david" / "outputs" / "pipeline_runs",
        ]
        for base in roots:
            if not base.exists():
                continue
            run_dirs = [d for d in base.iterdir() if d.is_dir()]
            run_dirs.sort(key=lambda d: d.name, reverse=True)
            for rdir in run_dirs:
                stage_dir = rdir / stage_folder
                if stage_dir.exists():
                    for fname in ["selected_leads.tsv", "stage2_leads.tsv", "stage1_hits.tsv", "stage3_leads.tsv", "funnel_scaffolds.tsv", "rl_generated_candidates_scored.tsv"]:
                        target = stage_dir / fname
                        if target.exists():
                            try:
                                df = pd.read_csv(target, sep="\t")
                                if len(df) > 0:
                                    return df
                            except Exception:
                                continue
        return None

    def compute_real_usrcat_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute real USRCAT 3D shape/pharmacophore similarity against reference conformers."""
        if not self.REF_SDF.exists():
            try:
                from _david.code.prepare_query_conformer import QueryConformerPreparer
                preparer = QueryConformerPreparer(self.REF_SDF.parent)
                self.REF_SDF = preparer.generate_reference_sdf(self.seed_smiles, self.REF_SDF)
            except Exception:
                pass

        if not self.REF_SDF.exists():
            FallbackMonitor.record_fallback(
                fallback_type="usrcat_missing_ref_sdf",
                message="Reference SDF missing for USRCAT scoring; returning zeros",
                context={"n_molecules": len(smiles_list)},
            )
            return np.zeros(len(smiles_list), dtype=np.float32)

        try:
            from _david.code.usrcat_scorer import USRCATScorer
            scorer = USRCATScorer(references=self.REF_SDF, num_conformers=5)
            scores = scorer.getScores(smiles_list)
            return np.asarray(scores, dtype=np.float32).flatten()
        except Exception as e:
            FallbackMonitor.record_fallback(
                fallback_type="usrcat_scoring_failed",
                message=f"USRCATScorer failed with exception: {e}; returning zeros",
                context={"n_molecules": len(smiles_list)},
            )
            return np.zeros(len(smiles_list), dtype=np.float32)

    def compute_real_rocs_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute real physical Gaussian volume overlap against reference conformers."""
        if not smiles_list:
            return np.zeros(0, dtype=np.float32)

        if not self.REF_SDF.exists():
            try:
                from _david.code.prepare_query_conformer import QueryConformerPreparer
                preparer = QueryConformerPreparer(self.REF_SDF.parent)
                self.REF_SDF = preparer.generate_reference_sdf(self.seed_smiles, self.REF_SDF)
            except Exception:
                pass

        if not self.REF_SDF.exists():
            FallbackMonitor.record_fallback(
                fallback_type="rocs_missing_ref_sdf",
                message="Reference SDF missing for ROCS scoring; returning zeros",
                context={"n_molecules": len(smiles_list)},
            )
            return np.zeros(len(smiles_list), dtype=np.float32)

        conf_gen = RDKitConformerGenerator(
            max_conformers=10,
            max_isomers=4,
            max_heavy_atoms=max(50, int(self.seed_hac * 1.8)),
            num_threads=0,
            timeout=15,
            show_progress=False,
        )

        if CDPKIT_AVAILABLE:
            try:
                scorer = CDPKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(self.REF_SDF),
                    show_progress=False,
                    opt_mode="combo",
                )
                cdp_scores = np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
                if len(cdp_scores) > 0 and np.all(cdp_scores == 0.0):
                    FallbackMonitor.record_fallback(
                        fallback_type="rocs_cdpkit_zero_scores",
                        message="CDPKit ROCS returned all-zero scores; falling back to RDKitROCSScorer",
                        context={"n_molecules": len(smiles_list)},
                    )
                else:
                    return cdp_scores
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_cdpkit_failed",
                    message=f"CDPKit ROCS failed with exception: {e}; falling back to RDKitROCSScorer",
                    context={"n_molecules": len(smiles_list)},
                )

        if RDKIT_ROCS_AVAILABLE:
            try:
                scorer = RDKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(self.REF_SDF),
                    score_type="TanimotoCombo",
                    use_colors=True,
                    show_progress=False,
                    n_jobs=2,
                )
                return np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_rdkit_failed",
                    message=f"RDKit ROCS failed with exception: {e}; falling back to default 0.50",
                    context={"n_molecules": len(smiles_list)},
                )

        FallbackMonitor.record_fallback(
            fallback_type="rocs_all_engines_failed",
            message="Both CDPKit and RDKit ROCS engines failed/unavailable in benchmark; falling back to 0.50",
            context={"n_molecules": len(smiles_list)},
        )
        return np.full(len(smiles_list), 0.5, dtype=np.float32)

    def prepare_regime_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw candidate TSV, calculate missing properties, and preserve authentic scores."""
        # Deduplicate columns
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()
        smiles_col = "SMILES" if "SMILES" in raw_df.columns else ("smiles" if "smiles" in raw_df.columns else None)
        if smiles_col is None:
            return pd.DataFrame()

        smiles_list = raw_df[smiles_col].astype(str).tolist()
        clean_df = self.compute_molecular_properties(smiles_list)

        # Map ROCS score
        rocs_col = next((c for c in raw_df.columns if c in ["ROCS", "RDKit_ROCS", "CDPKit_ROCS", "TanimotoCombo"] or (("ROCS" in c or "Shape" in c) and "ElectroShape" not in c)), None)
        if rocs_col is not None:
            clean_df["ROCS"] = pd.to_numeric(raw_df[rocs_col], errors="coerce").fillna(0.0).values
        else:
            clean_df["ROCS"] = self.compute_real_rocs_scores(smiles_list)

        # Map USRCAT score
        usrcat_col = next((c for c in raw_df.columns if c in ["ElectroShape", "electroshape", "USRCAT", "usrcat"] or "ElectroShape" in c or "USRCAT" in c), None)
        if usrcat_col is not None:
            clean_df["USRCAT"] = pd.to_numeric(raw_df[usrcat_col], errors="coerce").fillna(0.0).values
        else:
            clean_df["USRCAT"] = self.compute_real_usrcat_scores(smiles_list)

        # Map Retrosynthetic Accessibility (RAscore)
        ra_col = next((c for c in raw_df.columns if c in ["RAscore", "RA", "RA_Score", "rascore"] or "Retrosynthetic" in c or c.startswith("RA_") or c.startswith("RAscore")), None)
        if ra_col is not None:
            clean_df["RAscore"] = pd.to_numeric(raw_df[ra_col], errors="coerce").fillna(0.50).values
        else:
            clean_df["RAscore"] = 0.50

        # Preserve Desired status if present
        if "Desired" in raw_df.columns:
            clean_df["Desired"] = pd.to_numeric(raw_df["Desired"], errors="coerce").fillna(0).astype(int).values
        else:
            clean_df["Desired"] = (
                (clean_df["Valid"] == 1) &
                (clean_df["ROCS"] >= ROCS_THRESHOLD) &
                (clean_df["RAscore"] >= RA_THRESHOLD) &
                (clean_df["QED"] >= QED_THRESHOLD) &
                (clean_df["HeavyAtoms"] >= self.min_heavy_atoms) &
                (clean_df["HeavyAtoms"] <= self.max_heavy_atoms)
            ).astype(int)

        return clean_df

    def generate_summary_table(self, datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
        """Generate structured academic benchmark summary metrics table."""
        rows: list[dict[str, Any]] = []
        for regime, df in datasets.items():
            valid_df = df[df["Valid"] == 1]
            total_n = len(df)
            valid_n = len(valid_df)

            validity_pct = (valid_n / total_n * 100.0) if total_n > 0 else 0.0
            unique_pct = (valid_df["SMILES"].nunique() / valid_n * 100.0) if valid_n > 0 else 0.0
            desired_pct = (df["Desired"].sum() / total_n * 100.0) if total_n > 0 else 0.0

            mean_rocs = float(valid_df["ROCS"].mean()) if valid_n > 0 else 0.0
            mean_usrcat = float(valid_df["USRCAT"].mean()) if valid_n > 0 else 0.0
            mean_ra = float(valid_df["RAscore"].mean()) if valid_n > 0 else 0.0
            mean_qed = float(valid_df["QED"].mean()) if valid_n > 0 else 0.0
            mean_sa = float(valid_df["SA"].mean()) if valid_n > 0 else 0.0
            mean_hac = float(valid_df["HeavyAtoms"].mean()) if valid_n > 0 else 0.0
            mean_sim = float(valid_df["Tanimoto_to_Seed"].mean()) if valid_n > 0 else 0.0

            unique_scaffolds = valid_df["Murcko_Scaffold"].dropna().nunique() if valid_n > 0 else 0
            scaffold_div = (unique_scaffolds / valid_n) if valid_n > 0 else 0.0

            rows.append({
                "Regime": regime,
                "N_Generated": total_n,
                "Validity (%)": round(validity_pct, 1),
                "Uniqueness (%)": round(unique_pct, 1),
                "Desired Ratio (%)": round(desired_pct, 1),
                "ROCS (Mean)": round(mean_rocs, 2),
                "USRCAT (Mean)": round(mean_usrcat, 2),
                "RAscore (Mean)": round(mean_ra, 2),
                "QED (Mean)": round(mean_qed, 2),
                "SAscore (Mean)": round(mean_sa, 2),
                "Heavy Atoms (Mean)": round(mean_hac, 1),
                "Seed Tanimoto (Mean)": round(mean_sim, 2),
                "Scaffold Diversity": round(scaffold_div, 2),
            })

        table_df = pd.DataFrame(rows)
        tsv_path = self.BASE_OUTPUT_DIR / "thesis_benchmark_table.tsv"
        table_df.to_csv(tsv_path, sep="\t", index=False)

        # Markdown representation
        try:
            md_text = table_df.to_markdown(index=False)
        except Exception:
            cols = [str(c) for c in table_df.columns]
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows_str = ["| " + " | ".join(str(r[c]) for c in table_df.columns) + " |" for _, r in table_df.iterrows()]
            md_text = "\n".join([header, sep] + rows_str)

        md_path = self.BASE_OUTPUT_DIR / "thesis_benchmark_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text + "\n")

        return table_df, md_text

    def plot_radar_chart(self, datasets: dict[str, pd.DataFrame]) -> Path:
        """Render publication-grade radar chart comparing multi-objective performance across regimes."""
        categories = ["3D Shape (ROCS)", "Synthesizability (RA)", "Drug-likeness (QED)", "Size Fit (HAC)", "Scaffold Diversity"]
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        #? [Matplotlib] Polar projection initialization for radial multi-objective radar diagram.
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        colors = {
            "Sequence-RNN (Pure RL)": COLOR_RNN,
            "Graph-Transformer (Scaffold RL)": COLOR_GRAPH,
            "Graph-Transformer (BRICS Seed)": COLOR_GRAPH,
            "Prioritized Leads (Top Desirability)": COLOR_FUNNEL,
            "Hybrid Funnel (Two-Stage)": COLOR_FUNNEL,
        }

        for regime, df in datasets.items():
            valid_df = df[df["Valid"] == 1]
            if len(valid_df) == 0:
                continue

            v_rocs = float(valid_df["ROCS"].mean())
            v_ra = float(valid_df["RAscore"].mean())
            v_qed = float(valid_df["QED"].mean())
            # Size fit normalized dynamically against query seed heavy atom count
            mean_hac = float(valid_df["HeavyAtoms"].mean())
            v_size = max(0.0, 1.0 - abs(mean_hac - float(self.seed_hac)) / max(10.0, self.seed_hac * 0.5))
            # Scaffold diversity
            scaff_count = valid_df["Murcko_Scaffold"].dropna().nunique()
            v_div = scaff_count / len(valid_df)

            values = [v_rocs, v_ra, v_qed, v_size, v_div]
            values += values[:1]

            color = colors.get(regime, "#333333")
            ax.plot(angles, values, color=color, linewidth=2.2, label=regime)
            ax.fill(angles, values, color=color, alpha=0.15)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, weight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
        ax.grid(color="gray", linestyle="--", alpha=0.5)

        plt.title("Multi-Objective Performance Envelope Across Generative Regimes", size=13, weight="bold", y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        out_path = self.BASE_OUTPUT_DIR / "thesis_property_radar.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path

    def plot_chemical_space_pca(self, datasets: dict[str, pd.DataFrame]) -> Path:
        """Project candidate structural space onto 2D PCA using 2048-bit Morgan fingerprints."""
        all_fps: list[np.ndarray] = []
        labels: list[str] = []

        # Seed reference point
        seed_label = f"Query Seed (HAC={self.seed_hac})"
        if self.SEED_FP:
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(self.SEED_FP, arr)
            all_fps.append(arr)
            labels.append(seed_label)

        # Collect candidate fingerprints
        for regime, df in datasets.items():
            valid_df = df[df["Valid"] == 1].head(150)
            for smi in valid_df["SMILES"]:
                m = Chem.MolFromSmiles(smi)
                if m:
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                    arr = np.zeros((1,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    all_fps.append(arr)
                    labels.append(regime)

        if len(all_fps) < 5:
            return self.BASE_OUTPUT_DIR / "thesis_chemical_space.png"

        X = np.array(all_fps)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = {
            seed_label: (COLOR_REF, "*", 200),
            "Sequence-RNN (Pure RL)": (COLOR_RNN, "o", 30),
            "Graph-Transformer (Scaffold RL)": (COLOR_GRAPH, "^", 35),
            "Graph-Transformer (BRICS Seed)": (COLOR_GRAPH, "^", 35),
            "Prioritized Leads (Top Desirability)": (COLOR_FUNNEL, "s", 40),
            "Hybrid Funnel (Two-Stage)": (COLOR_FUNNEL, "s", 40),
        }

        labels_arr = np.array(labels)
        for label_key, (c, m, s) in color_map.items():
            mask = labels_arr == label_key
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], c=c, marker=m, s=s, label=label_key, alpha=0.75, edgecolors="none")

        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        ax.set_title("Chemical Space Projection (Morgan FP2048 PCA)", weight="bold", fontsize=12)
        ax.set_xlabel(f"PC 1 ({var1:.1f}% Variance)")
        ax.set_ylabel(f"PC 2 ({var2:.1f}% Variance)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

        out_path = self.BASE_OUTPUT_DIR / "thesis_chemical_space.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path

    def plot_score_distributions(self, datasets: dict[str, pd.DataFrame]) -> Path:
        """Render side-by-side boxplots for ROCS, RAscore, QED, and Heavy Atom Count."""
        metrics = ["ROCS", "RAscore", "QED", "HeavyAtoms"]
        titles = ["ROCS 3D Shape", "Retrosynthetic Accessibility", "Drug-likeness (QED)", "Heavy Atom Count (HAC)"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, (metric, title) in enumerate(zip(metrics, titles, strict=True)):
            ax = axes[idx]
            plot_data: list[np.ndarray] = []
            labels: list[str] = []

            for regime, df in datasets.items():
                valid_df = df[df["Valid"] == 1]
                if metric in valid_df.columns:
                    vals = valid_df[metric].dropna().values
                    if len(vals) > 0:
                        plot_data.append(vals)
                        labels.append(regime.split()[0])

            if plot_data:
                ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#A8DADC", color="#1D3557"))
            ax.set_title(title, weight="bold", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        out_path = self.BASE_OUTPUT_DIR / "thesis_score_distributions.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path

    def plot_top_leads_grid(self, datasets: dict[str, pd.DataFrame]) -> Path | None:
        """Depict 2D chemical structure grid of elite leads ranked by multi-objective performance."""
        selected_mols: list[Chem.Mol] = []
        legends: list[str] = []

        # 1. Reference query seed
        if self.SEED_MOL:
            selected_mols.append(self.SEED_MOL)
            seed_name = "EGCG" if self.seed_smiles == MY_MOLECULE else "Query"
            legends.append(f"Query Seed ({seed_name})\nHAC: {self.seed_hac} | Ref Active")

        # 2. Top from Prioritized Leads / Funnel
        leads_df = None
        for key in ["Prioritized Leads (Top Desirability)", "Hybrid Funnel (Two-Stage)", "Selected Leads"]:
            if key in datasets:
                leads_df = datasets[key]
                break

        if leads_df is not None:
            valid_leads = leads_df[leads_df["Valid"] == 1].copy()
            if "Composite_Desirability" in valid_leads.columns:
                valid_leads["SortMetric"] = valid_leads["Composite_Desirability"]
            else:
                valid_leads["SortMetric"] = valid_leads["ROCS"] + valid_leads["RAscore"] + valid_leads["QED"]
            top_leads = valid_leads.sort_values(by="SortMetric", ascending=False).head(4)
            for _, r in top_leads.iterrows():
                m = Chem.MolFromSmiles(r["SMILES"])
                if m:
                    selected_mols.append(m)
                    legends.append(
                        f"Lead #{len(selected_mols)-1}\n"
                        f"ROCS: {r['ROCS']:.2f} | RA: {r['RAscore']:.2f}\n"
                        f"HAC: {r['HeavyAtoms']} | Sim: {r['Tanimoto_to_Seed']:.2f}"
                    )

        # 3. Top from Graph-Transformer
        graph_df = None
        for key in ["Graph-Transformer (Scaffold RL)", "Graph-Transformer (BRICS Seed)"]:
            if key in datasets:
                graph_df = datasets[key]
                break
        if graph_df is not None:
            valid_graph = graph_df[graph_df["Valid"] == 1].copy()
            valid_graph["SortMetric"] = valid_graph["ROCS"] + valid_graph["RAscore"]
            top_graph = valid_graph.sort_values(by="SortMetric", ascending=False).head(2)
            for _, r in top_graph.iterrows():
                m = Chem.MolFromSmiles(r["SMILES"])
                if m:
                    selected_mols.append(m)
                    legends.append(
                        f"Graph-TF #{len(selected_mols)-1}\n"
                        f"ROCS: {r['ROCS']:.2f} | RA: {r['RAscore']:.2f}\n"
                        f"HAC: {r['HeavyAtoms']} | Sim: {r['Tanimoto_to_Seed']:.2f}"
                    )

        # 4. Top from Sequence-RNN
        rnn_df = datasets.get("Sequence-RNN (Pure RL)")
        if rnn_df is not None:
            valid_rnn = rnn_df[rnn_df["Valid"] == 1].copy()
            valid_rnn["SortMetric"] = valid_rnn["ROCS"] + valid_rnn["RAscore"]
            top_rnn = valid_rnn.sort_values(by="SortMetric", ascending=False).head(2)
            for _, r in top_rnn.iterrows():
                m = Chem.MolFromSmiles(r["SMILES"])
                if m:
                    selected_mols.append(m)
                    legends.append(
                        f"SEQ-RNN #{len(selected_mols)-1}\n"
                        f"ROCS: {r['ROCS']:.2f} | RA: {r['RAscore']:.2f}\n"
                        f"HAC: {r['HeavyAtoms']} | Sim: {r['Tanimoto_to_Seed']:.2f}"
                    )

        if not selected_mols:
            return None

        #? [RDKit Draw] Renders multi-record depiction image grid with property legends.
        img = Draw.MolsToGridImage(
            selected_mols,
            molsPerRow=min(4, len(selected_mols)),
            subImgSize=(300, 300),
            legends=legends,
            useSVG=False,
        )
        out_path = self.BASE_OUTPUT_DIR / "thesis_top_leads_grid.png"
        img.save(str(out_path))
        return out_path

    def run_benchmark(
        self,
        rnn_path: Path | None = None,
        graph_path: Path | None = None,
        funnel_path: Path | None = None,
        leads_path: Path | None = None,
    ) -> pd.DataFrame:
        """Execute end-to-end benchmark suite on real empirical candidate data."""
        print("Loading authentic datasets into ThesisBenchmark...")
        datasets = self.load_real_datasets(
            rnn_path=rnn_path,
            graph_path=graph_path,
            funnel_path=funnel_path,
            leads_path=leads_path,
        )

        print("Generating Benchmark Summary Table...")
        table_df, md_text = self.generate_summary_table(datasets)
        print("\n" + md_text + "\n")

        print("Generating Publication Radar Chart...")
        self.plot_radar_chart(datasets)

        print("Generating Chemical Space PCA Projection...")
        self.plot_chemical_space_pca(datasets)

        print("Generating Score Distributions Boxplots...")
        self.plot_score_distributions(datasets)

        print("Generating 2D Top Leads Grid...")
        self.plot_top_leads_grid(datasets)

        print(f"Thesis benchmark suite successfully executed! All outputs saved to: {self.BASE_OUTPUT_DIR}")
        return table_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis Benchmark Suite for Ligand-Based De Novo Drug Design")
    parser.add_argument("--rnn-file", type=Path, default=None, help="Path to SequenceRNN scored candidates TSV")
    parser.add_argument("--graph-file", type=Path, default=None, help="Path to GraphTransformer scored candidates TSV")
    parser.add_argument("--leads-file", "--funnel-file", dest="leads_file", type=Path, default=None, help="Path to Prioritized Leads TSV")
    parser.add_argument("--seed-smiles", type=str, default=MY_MOLECULE, help="Active query seed SMILES string")
    parser.add_argument("--ref-sdf", type=Path, default=None, help="Path to 3D reference conformer ensemble SDF")
    parser.add_argument("--out-dir", type=Path, default=None, help="Target output directory for figures and tables")
    args = parser.parse_args()

    benchmark = ThesisBenchmark(output_dir=args.out_dir, ref_sdf=args.ref_sdf, seed_smiles=args.seed_smiles)
    benchmark.run_benchmark(args.rnn_file, args.graph_file, leads_path=args.leads_file)
    FallbackMonitor.print_summary()
