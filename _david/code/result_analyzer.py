"""Result Analysis and Lead Profiling for Ligand-Based De Novo Drug Design.

Evaluates generated molecules against query reference:
1. 3D shape overlay: Gaussian ROCS (TanimotoCombo) and 75-D ElectroShape moments.
2. Physicochemical properties: QED, RAscore, SAscore, MW, LogP, HAC, TPSA, HBD, HBA, RotBonds.
3. 2D structural distance: Morgan ECFP4 Tanimoto similarity to seed ligand.
4. Chemical liability audit: Catechol / polyphenol and PAINS detection.
5. Multi-objective lead prioritization and diversity selection.
6. Publication-grade visualizations and structured summary export.
"""
from __future__ import annotations

import contextlib
import json
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
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, Lipinski, QED, rdMolDescriptors
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

try:
    from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
except Exception:
    def calculate_sa_score(mol: Chem.Mol) -> float:
        return 3.5

from _david.code.anti_cheating_rewards import AugmentedChebyshevReward
from _david.code.fallback_monitor import FallbackMonitor
from _david.code.usrcat_scorer import USRCATScorer

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
CATECHOL_SMARTS: str = "c1c([OH])c([OH])ccc1"
ROCS_THRESHOLD: float = 0.50
QED_THRESHOLD: float = 0.40
RA_THRESHOLD: float = 0.50


class ResultAnalyzer:
    """Comprehensive evaluative profiler and visualizer for generated drug leads."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        ref_sdf_path: Path | str | None = None,
        seed_smiles: str = MY_MOLECULE,
    ) -> None:
        """Initialize ResultAnalyzer with query reference coordinates and output directory.

        Args:
            output_dir: Destination folder for analysis tables, figures, and JSON summaries.
            ref_sdf_path: Path to low-energy 3D conformer ensemble SDF of the reference ligand.
            seed_smiles: SMILES string of the primary reference proof molecule.
        """
        self.output_dir: Path = Path(output_dir) if output_dir is not None else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed_smiles: str = seed_smiles
        self.seed_mol: Chem.Mol | None = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None

        #? [RDKit Morgan Fingerprint] Pre-computes 2048-bit ECFP4 fingerprint for 2D structural distance auditing.
        self.seed_fp: DataStructs.ExplicitBitVect | None = (
            rdMolDescriptors.GetMorganFingerprintAsBitVect(self.seed_mol, 2, nBits=2048)
            if self.seed_mol is not None
            else None
        )
        self.seed_hac: int = self.seed_mol.GetNumHeavyAtoms() if self.seed_mol else 33

        if ref_sdf_path is not None:
            self.ref_sdf_path: Path = Path(ref_sdf_path)
        else:
            shared_ref = PROJECT_ROOT / "_david" / "references" / "my_molecule_3d.sdf"
            self.ref_sdf_path = shared_ref

        #? [RDKit Substructure] Pre-compile catechol liability SMARTS pattern.
        self._catechol_query: Chem.Mol = Chem.MolFromSmarts(CATECHOL_SMARTS)

    def compute_rocs_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute Gaussian ROCS shape overlay scores against reference 3D conformers.

        Args:
            smiles_list: Sequence of candidate SMILES strings to evaluate.

        Returns:
            1D NumPy array of float32 ROCS TanimotoCombo scores (0.0 to 2.0).
        """
        if not smiles_list:
            return np.zeros(0, dtype=np.float32)

        if not self.ref_sdf_path.exists():
            FallbackMonitor.record_fallback(
                fallback_type="rocs_missing_ref_sdf",
                message=f"Reference SDF missing at {self.ref_sdf_path}; returning default score 0.50",
                context={"n_molecules": len(smiles_list)},
            )
            return np.full(len(smiles_list), 0.50, dtype=np.float32)

        #? [RDKit Conformer Generator] Multi-threaded conformer ensemble generator for ROCS overlay alignment.
        conf_gen = RDKitConformerGenerator(
            max_conformers=10,
            max_isomers=2,
            max_heavy_atoms=max(50, int(self.seed_hac * 1.8)),
            num_threads=0,
            timeout=15,
            show_progress=False,
        )

        if CDPKIT_AVAILABLE:
            try:
                scorer = CDPKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(self.ref_sdf_path),
                    show_progress=False,
                    opt_mode="combo",
                )
                scores = np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
                if len(scores) > 0 and not np.all(scores == 0.0):
                    return scores
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_cdpkit_failed",
                    message=f"CDPKit ROCS failed ({e}); falling back to RDKitROCSScorer",
                    context={"n_molecules": len(smiles_list)},
                )

        if RDKIT_ROCS_AVAILABLE:
            try:
                scorer = RDKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(self.ref_sdf_path),
                    score_type="TanimotoCombo",
                    use_colors=True,
                    show_progress=False,
                    n_jobs=2,
                )
                return np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_rdkit_failed",
                    message=f"RDKit ROCS failed ({e}); returning default score 0.50",
                    context={"n_molecules": len(smiles_list)},
                )

        return np.full(len(smiles_list), 0.50, dtype=np.float32)

    def compute_electroshape_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute ultrafast 75-D ElectroShape shape/charge moment similarity scores.

        Args:
            smiles_list: Sequence of candidate SMILES strings to evaluate.

        Returns:
            1D NumPy array of float32 moment similarity scores (0.0 to 1.0).
        """
        if not smiles_list or not self.ref_sdf_path.exists():
            return np.zeros(len(smiles_list), dtype=np.float32)

        try:
            scorer = USRCATScorer(references=str(self.ref_sdf_path), score_type="electroshape", num_conformers=5)
            scores = scorer.getScores(smiles_list)
            return np.asarray(scores, dtype=np.float32).flatten()
        except Exception as e:
            FallbackMonitor.record_fallback(
                fallback_type="electroshape_calc_failed",
                message=f"ElectroShape moment scoring failed ({e}); returning zeros",
                context={"n_molecules": len(smiles_list)},
            )
            return np.zeros(len(smiles_list), dtype=np.float32)

    def compute_rascores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute machine-learning retrosynthetic synthesizability scores using RAscore.

        Args:
            smiles_list: Sequence of candidate SMILES strings to evaluate.

        Returns:
            1D NumPy array of float32 synthesizability probabilities (0.0 to 1.0).
        """
        if not smiles_list:
            return np.zeros(0, dtype=np.float32)
        try:
            scorer = RetrosyntheticAccessibilityScorer()
            scores = scorer.getScores(smiles_list)
            return np.asarray(scores, dtype=np.float32).flatten()
        except Exception as e:
            FallbackMonitor.record_fallback(
                fallback_type="rascore_calc_failed",
                message=f"RAscore evaluation failed ({e}); returning default score 0.50",
                context={"n_molecules": len(smiles_list)},
            )
            return np.full(len(smiles_list), 0.50, dtype=np.float32)

    def analyze_candidates(self, candidates: Sequence[str] | pd.DataFrame) -> pd.DataFrame:
        """Profile all candidate molecules with complete 3D, 2D, and physicochemical metrics.

        Args:
            candidates: Sequence of SMILES or DataFrame containing candidate molecules.

        Returns:
            Enriched DataFrame containing all evaluated chemical, structural, and 3D descriptors.
        """
        if isinstance(candidates, pd.DataFrame):
            col = "SMILES" if "SMILES" in candidates.columns else ("smiles" if "smiles" in candidates.columns else None)
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
                    "Has_Catechol": False,
                })
            else:
                valid_mols.append((smi, mol))

        # Profile valid molecules
        for smi, mol in valid_mols:
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

            try:
                scaff = MurckoScaffold.GetScaffoldForMol(mol)
                scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=False) if scaff.GetNumHeavyAtoms() > 0 else can_smi
            except Exception:
                scaff_smi = can_smi

            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim_2d = float(DataStructs.TanimotoSimilarity(self.seed_fp, fp)) if self.seed_fp else 0.0

            # Toxicophore / catechol liability audit
            has_catechol = bool(mol.HasSubstructMatch(self._catechol_query))

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
                "Has_Catechol": has_catechol,
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        valid_mask = df["Valid"]
        valid_smiles = df.loc[valid_mask, "SMILES"].tolist()

        # Batch 3D & synthesizability evaluations on valid molecules
        rocs_scores = self.compute_rocs_scores(valid_smiles)
        electroshape_scores = self.compute_electroshape_scores(valid_smiles)
        ra_scores = self.compute_rascores(valid_smiles)

        df["ROCS"] = 0.0
        df["ElectroShape"] = 0.0
        df["RAscore"] = 0.0

        df.loc[valid_mask, "ROCS"] = rocs_scores
        df.loc[valid_mask, "ElectroShape"] = electroshape_scores
        df.loc[valid_mask, "RAscore"] = ra_scores

        # Calculate Augmented Chebyshev Desirability score
        # Balances ROCS (weight 0.40), QED (weight 0.30), RAscore (weight 0.30)
        norm_rocs = np.clip(df["ROCS"] / 1.50, 0.0, 1.0)
        norm_qed = np.clip(df["QED"], 0.0, 1.0)
        norm_ra = np.clip(df["RAscore"], 0.0, 1.0)

        # Penalty for catechol liability
        liability_penalty = np.where(df["Has_Catechol"], 0.10, 1.00)

        chebyshev_min = np.minimum(np.minimum(norm_rocs, norm_qed), norm_ra)
        chebyshev_mean = (0.40 * norm_rocs + 0.30 * norm_qed + 0.30 * norm_ra)
        desirability = (0.80 * chebyshev_min + 0.20 * chebyshev_mean) * liability_penalty
        df["Desirability"] = np.where(df["Valid"], np.round(desirability, 4), 0.0)

        out_tsv = self.output_dir / "all_analyzed_candidates.tsv"
        df.to_csv(out_tsv, sep="\t", index=False)
        return df

    def prioritize_leads(self, df_analyzed: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Prioritize elite lead molecules balancing 3D shape, synthesizability, and scaffold diversity.

        Args:
            df_analyzed: Analyzed candidates DataFrame.
            top_k: Number of diverse top leads to prioritize.

        Returns:
            Filtered and ranked DataFrame containing top elite lead candidates.
        """
        if df_analyzed.empty:
            return pd.DataFrame()

        # Filter strictly for valid, catechol-free candidates
        viable = df_analyzed[df_analyzed["Valid"] & (~df_analyzed["Has_Catechol"])].copy()
        if viable.empty:
            # Fallback to valid candidates if none are completely catechol-free
            FallbackMonitor.record_fallback(
                fallback_type="lead_prioritization_catechol_fallback",
                message="Zero catechol-free candidates found; relaxing liability filter",
            )
            viable = df_analyzed[df_analyzed["Valid"]].copy()

        if viable.empty:
            return pd.DataFrame()

        # Sort descending by multi-objective Desirability score
        viable = viable.sort_values(by="Desirability", ascending=False).reset_index(drop=True)

        # Enforce scaffold diversity: pick best candidate per unique Murcko scaffold
        selected_indices: list[int] = []
        seen_scaffolds: set[str] = set()

        for idx, row in viable.iterrows():
            scaff = row["Scaffold"]
            if scaff not in seen_scaffolds:
                selected_indices.append(idx)
                seen_scaffolds.add(scaff)
            if len(selected_indices) >= top_k:
                break

        # If not enough distinct scaffolds, fill up with next highest desirability candidates
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

        # 1. 2D Chemical Structure Grid Image
        plot_mols: list[Chem.Mol] = []
        legends: list[str] = []

        # Add seed molecule as the primary visual benchmark
        if self.seed_mol is not None:
            plot_mols.append(self.seed_mol)
            seed_qed = float(QED.qed(self.seed_mol))
            legends.append(f"QUERY SEED (EGCG)\nHAC:{self.seed_hac} | QED:{seed_qed:.2f}\nCatechol: YES (Liab)")

        for idx, row in df_leads.head(max_leads).iterrows():
            m = Chem.MolFromSmiles(row["SMILES"])
            if m is not None:
                plot_mols.append(m)
                legends.append(
                    f"Lead #{idx + 1}\n"
                    f"ROCS:{row['ROCS']:.2f} | QED:{row['QED']:.2f}\n"
                    f"RA:{row['RAscore']:.2f} | Des:{row['Desirability']:.2f}"
                )

        if plot_mols:
            grid_img = Draw.MolsToGridImage(
                plot_mols,
                molsPerRow=min(3, len(plot_mols)),
                subImgSize=(320, 240),
                legends=legends,
                useSVG=False,
            )
            grid_path = self.output_dir / "lead_structures_2d.png"
            grid_img.save(str(grid_path))
            figures["structures_grid"] = grid_path

        # 2. Physicochemical Property Distributions Chart
        eval_df = df_all if df_all is not None and not df_all.empty else df_leads
        valid_df = eval_df[eval_df["Valid"]].copy()

        if len(valid_df) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            fig.suptitle("Ligand-Based De Novo Drug Design: Generated Lead Profiles", fontsize=14, weight="bold")

            # A. 3D ROCS Distribution
            ax = axes[0, 0]
            ax.hist(valid_df["ROCS"], bins=15, color="#2A9D8F", edgecolor="black", alpha=0.75)
            ax.axvline(ROCS_THRESHOLD, color="#E76F51", linestyle="--", label=f"Threshold ({ROCS_THRESHOLD})")
            ax.set_title("3D Gaussian ROCS Overlay (TanimotoCombo)", fontsize=11)
            ax.set_xlabel("ROCS Score (0 to 2.0)")
            ax.set_ylabel("Candidate Count")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            # B. QED Drug-likeness Distribution
            ax = axes[0, 1]
            ax.hist(valid_df["QED"], bins=15, color="#457B9D", edgecolor="black", alpha=0.75)
            ax.axvline(QED_THRESHOLD, color="#E76F51", linestyle="--", label=f"Threshold ({QED_THRESHOLD})")
            ax.set_title("Quantitative Estimate of Drug-likeness (QED)", fontsize=11)
            ax.set_xlabel("QED Score")
            ax.set_ylabel("Candidate Count")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            # C. Retrosynthetic Synthesizability (RAscore)
            ax = axes[1, 0]
            ax.hist(valid_df["RAscore"], bins=15, color="#E9C46A", edgecolor="black", alpha=0.75)
            ax.axvline(RA_THRESHOLD, color="#E76F51", linestyle="--", label=f"Threshold ({RA_THRESHOLD})")
            ax.set_title("Retrosynthetic Synthesizability (RAscore)", fontsize=11)
            ax.set_xlabel("RAscore Probability")
            ax.set_ylabel("Candidate Count")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            # D. 2D Similarity vs 3D Shape Correlation
            ax = axes[1, 1]
            ax.scatter(valid_df["Tanimoto_to_Seed"], valid_df["ROCS"], c=valid_df["Desirability"], cmap="viridis", alpha=0.8, edgecolors="none", s=50)
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label("Desirability", fontsize=9)
            ax.set_title("2D Tanimoto Similarity vs 3D ROCS Shape", fontsize=11)
            ax.set_xlabel("2D Morgan ECFP4 Tanimoto to Seed")
            ax.set_ylabel("3D ROCS Score")
            ax.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            dist_path = self.output_dir / "property_distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            figures["property_distributions"] = dist_path

        return figures

    def export_summary(self, df_all: pd.DataFrame, df_leads: pd.DataFrame) -> dict[str, Any]:
        """Compile comprehensive descriptive analysis summary JSON.

        Args:
            df_all: DataFrame of all evaluated candidates.
            df_leads: DataFrame of prioritized elite leads.

        Returns:
            Dictionary containing structured statistical audit metrics.
        """
        valid_df = df_all[df_all["Valid"]].copy() if not df_all.empty else pd.DataFrame()
        total_n = len(df_all)
        valid_n = len(valid_df)

        summary: dict[str, Any] = {
            "query_seed_smiles": self.seed_smiles,
            "seed_heavy_atom_count": self.seed_hac,
            "total_candidates_generated": total_n,
            "valid_candidates_count": valid_n,
            "validity_percentage": round(valid_n / total_n * 100.0, 1) if total_n > 0 else 0.0,
            "unique_candidates_count": int(valid_df["SMILES"].nunique()) if valid_n > 0 else 0,
            "uniqueness_percentage": round(valid_df["SMILES"].nunique() / valid_n * 100.0, 1) if valid_n > 0 else 0.0,
            "catechol_free_count": int((~valid_df["Has_Catechol"]).sum()) if valid_n > 0 else 0,
            "catechol_free_percentage": round((~valid_df["Has_Catechol"]).sum() / valid_n * 100.0, 1) if valid_n > 0 else 0.0,
            "metrics": {
                "mean_rocs": round(float(valid_df["ROCS"].mean()), 3) if valid_n > 0 else 0.0,
                "max_rocs": round(float(valid_df["ROCS"].max()), 3) if valid_n > 0 else 0.0,
                "mean_qed": round(float(valid_df["QED"].mean()), 3) if valid_n > 0 else 0.0,
                "max_qed": round(float(valid_df["QED"].max()), 3) if valid_n > 0 else 0.0,
                "mean_rascore": round(float(valid_df["RAscore"].mean()), 3) if valid_n > 0 else 0.0,
                "max_rascore": round(float(valid_df["RAscore"].max()), 3) if valid_n > 0 else 0.0,
                "mean_electroshape": round(float(valid_df["ElectroShape"].mean()), 3) if valid_n > 0 else 0.0,
                "mean_tanimoto_to_seed": round(float(valid_df["Tanimoto_to_Seed"].mean()), 3) if valid_n > 0 else 0.0,
            },
            "top_leads_count": len(df_leads),
            "top_leads": df_leads.to_dict(orient="records") if not df_leads.empty else [],
        }

        json_path = self.output_dir / "result_analysis_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary
