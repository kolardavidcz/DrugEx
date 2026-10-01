"""Molecular chemical space visualization using Scaffviz t-SNE and Plotly dashboards."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeAlign

try:
    from papyrus_scripts.reader import read_papyrus
except ImportError:
    read_papyrus = None
try:
    from qsprpred.data import MoleculeTable
    from qsprpred.data.descriptors.fingerprints import MorganFP
except ImportError:
    MoleculeTable = None
    MorganFP = None
try:
    from scaffviz.clustering.manifold import TSNE
    from scaffviz.data.manifold_table import ManifoldTable
    from scaffviz.depiction.plot import Plot
except ImportError:
    TSNE = None
    ManifoldTable = None
    Plot = None

try:
    from _david.code import receptor_similar
except ImportError:
    try:
        from _david import receptor_similar
    except ImportError:
        receptor_similar = None

from _david.code.fallback_monitor import FallbackMonitor

SEED_SMILES: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
N_BACKGROUND: int = 1000
N_CANDIDATES: int = 1000
FP_RADIUS: int = 3
FP_NBITS: int = 2048
TSNE_PERPLEXITY: float = 35.0
TSNE_N_ITER: int = 1000
RANDOM_STATE: int = 42
N_JOBS: int = 8
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "_david" / "output" / "molecular_space_visualization"

VIBRANT_BLUE_CMAP = [
    (0.00, "#FFFFFF"),
    (0.25, "#E1EDFE"),
    (0.50, "#85B6FF"),
    (0.75, "#1B64F2"),
    (1.00, "#001A66"),
]


@dataclass(frozen=True)
class MolecularSpaceConfig:
    query_smiles: str = SEED_SMILES
    n_background: int = N_BACKGROUND
    n_candidates: int = N_CANDIDATES
    fp_radius: int = FP_RADIUS
    fp_nbits: int = FP_NBITS
    tsne_perplexity: float = TSNE_PERPLEXITY
    tsne_n_iter: int = TSNE_N_ITER
    random_state: int = RANDOM_STATE
    n_jobs: int = N_JOBS


class ConformerSimilarityEngine:
    @staticmethod
    def generate_3d_conformer(smiles: str, random_seed: int = RANDOM_STATE) -> Optional[Chem.Mol]:
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            code = AllChem.EmbedMolecule(mol, params)
            if code == -1:
                FallbackMonitor.record_fallback(
                    fallback_type="visualize_embed_random_coords",
                    message="Standard ETKDGv3 embedding failed; relaxing with useRandomCoords=True",
                    context={"smiles": smiles},
                )
                code = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=random_seed)
            if code == -1:
                return None
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        except Exception:
            return None

    @classmethod
    def compute_3d_similarities(
        cls, ref_smiles: str, candidate_smiles: list[str], n_jobs: int = N_JOBS
    ) -> Tuple[List[float], List[float], List[float]]:
        ref_mol = cls.generate_3d_conformer(ref_smiles)
        if ref_mol is None:
            return [0.0] * len(candidate_smiles), [0.0] * len(candidate_smiles), [0.0] * len(candidate_smiles)

        def _score_single(smi: str) -> Tuple[float, float, float]:
            if smi == ref_smiles:
                return 1.0, 1.0, 1.0
            try:
                probe = cls.generate_3d_conformer(smi)
                if probe is None:
                    return 0.0, 0.0, 0.0
                shape_sim, color_sim = rdShapeAlign.AlignMol(ref_mol, probe)
                combo = (shape_sim + color_sim) / 2.0
                return float(shape_sim), float(color_sim), float(combo)
            except Exception:
                return 0.0, 0.0, 0.0

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            scores = list(executor.map(_score_single, candidate_smiles))

        shape_scores = [round(s[0], 4) for s in scores]
        color_scores = [round(s[1], 4) for s in scores]
        combo_scores = [round(s[2], 4) for s in scores]
        return shape_scores, color_scores, combo_scores


class ChemicalSpaceDataset:
    def __init__(self, papyrus_path: str, config: MolecularSpaceConfig) -> None:
        self.papyrus_path: str = papyrus_path
        self.config: MolecularSpaceConfig = config

    def sample_background_molecules(
        self, n_samples: int, exclude_smiles: Set[str], random_seed: int = RANDOM_STATE
    ) -> pd.DataFrame:
        exclude = exclude_smiles or set()
        raw_papyrus = read_papyrus(
            version="latest",
            source_path=self.papyrus_path,
            plusplus=True,
            chunksize=1,
        )
        lf = raw_papyrus.lazy() if isinstance(raw_papyrus, pl.DataFrame) else raw_papyrus

        collected = (
            lf.select(["SMILES"])
            .filter(pl.col("SMILES").is_not_null())
            .limit(n_samples * 25)
            .collect()
            .to_pandas()
        )

        valid_smiles = []
        for smi in collected["SMILES"].dropna().unique():
            if smi not in exclude and len(smi) > 4:
                valid_smiles.append(smi)
            if len(valid_smiles) >= n_samples * 3:
                break

        rng = np.random.default_rng(random_seed)
        selected_smiles = rng.choice(valid_smiles, size=min(n_samples, len(valid_smiles)), replace=False)

        bg_df = pd.DataFrame({
            "SMILES": selected_smiles,
            "Group": "Background Space (1,000 Random Molecules)",
            "property_similarity": 0.0,
            "sim_3d_shape": 0.0,
            "sim_3d_color": 0.0,
            "sim_3d_combo": 0.0,
            "Source": "Papyrus (Random Chemical Universe)",
            "Target": "None (Background)",
            "pAffinity": np.nan,
        })
        return bg_df

    def format_candidate_molecules(
        self, candidates_raw: Any, group_label: str = "Bioactivity Candidates"
    ) -> pd.DataFrame:
        if isinstance(candidates_raw, pd.DataFrame):
            df = candidates_raw.copy()
        elif isinstance(candidates_raw, pl.DataFrame):
            df = candidates_raw.to_pandas()
        elif isinstance(candidates_raw, list):
            df = pd.DataFrame(candidates_raw)
        else:
            raise ValueError(f"Unsupported candidate container type: {type(candidates_raw)}")

        if "SMILES" not in df.columns and "smiles" in df.columns:
            df["SMILES"] = df["smiles"]

        df["Group"] = group_label

        if "property_similarity" not in df.columns:
            df["property_similarity"] = 0.50

        if "source" in df.columns:
            df["Source"] = df["source"]
        else:
            df["Source"] = "Papyrus Bioactivity"

        if "target_accession" in df.columns:
            df["Target"] = df["target_accession"]
        else:
            df["Target"] = "Shared Receptors"

        keep_cols = ["SMILES", "Group", "property_similarity", "Source", "Target"]
        for extra in ["pAffinity", "MW", "logP", "TPSA", "QED"]:
            if extra in df.columns:
                keep_cols.append(extra)

        df = df[keep_cols].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

        shape_sim, color_sim, combo_sim = ConformerSimilarityEngine.compute_3d_similarities(
            SEED_SMILES, df["SMILES"].tolist(), n_jobs=self.config.n_jobs
        )
        df["sim_3d_shape"] = shape_sim
        df["sim_3d_color"] = color_sim
        df["sim_3d_combo"] = combo_sim

        return df

    def format_query_molecule(self, query_smiles: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "SMILES": query_smiles,
            "Group": "Query Molecule (my_molecule / EGCG)",
            "property_similarity": 1.0,
            "sim_3d_shape": 1.0,
            "sim_3d_color": 1.0,
            "sim_3d_combo": 1.0,
            "Source": "Input Reference",
            "Target": "Query Reference Ligand",
            "pAffinity": np.nan,
        }])

    def assemble_unified_table(
        self,
        background_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        query_df: pd.DataFrame,
        table_name: str,
        store_dir: Path,
    ) -> Tuple[Any, pd.DataFrame]:
        combined = pd.concat([background_df, candidates_df, query_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["SMILES"], keep="last").reset_index(drop=True)

        store_dir.mkdir(parents=True, exist_ok=True)
        table = MoleculeTable(
            name=table_name,
            df=combined,
            smiles_col="SMILES",
            store_dir=str(store_dir),
            overwrite=True,
            n_jobs=self.config.n_jobs,
        )

        table.addDescriptors([
            MorganFP(radius=self.config.fp_radius, nBits=self.config.fp_nbits)
        ])

        return table, combined


class ManifoldEmbeddingEngine:
    @staticmethod
    def compute_tsne_coordinates(table: Any, config: MolecularSpaceConfig) -> Tuple[str, str, pd.DataFrame]:
        manifold_tab = ManifoldTable.fromMolTable(table)
        tsne_manifold = TSNE(
            perplexity=config.tsne_perplexity,
            n_iter=config.tsne_n_iter,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )

        manifold_tab.addManifoldData(tsne_manifold)
        df_coords = manifold_tab.getDF()

        coord_cols = [c for c in df_coords.columns if c.startswith("TSNE_") or c.startswith("tSNE_")]
        if len(coord_cols) >= 2:
            x_col, y_col = coord_cols[0], coord_cols[1]
        else:
            num_cols = df_coords.select_dtypes(include=[np.number]).columns.tolist()
            x_col, y_col = num_cols[-2], num_cols[-1]

        return x_col, y_col, df_coords


class MolecularSpaceRenderer:
    @classmethod
    def render_matplotlib(
        cls, df: pd.DataFrame, x_col: str, y_col: str, output_png: Path, title: str, subtitle: str
    ) -> Path:
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        df_cand = df[is_candidate].sort_values(by="property_similarity", ascending=True)
        df_query = df[is_query]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("VibrantBlue", VIBRANT_BLUE_CMAP, N=256)

        fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
        fig.patch.set_facecolor("#FCFDFF")
        ax.set_facecolor("#FCFDFF")

        ax.scatter(
            df_bg[x_col],
            df_bg[y_col],
            c="#B8C2CC",
            s=18,
            alpha=0.45,
            edgecolors="none",
            label="Background Universe (1,000 Random Molecules)",
            zorder=1,
        )

        sim_2d = df_cand["property_similarity"].fillna(0.0).values
        sim_3d = df_cand["sim_3d_shape"].fillna(0.0).values
        marker_sizes = 20.0 + 200.0 * (sim_3d ** 1.5)

        scatter_cand = ax.scatter(
            df_cand[x_col],
            df_cand[y_col],
            c=sim_2d,
            cmap=cmap,
            vmin=float(sim_2d.min()) if len(sim_2d) > 0 else 0.0,
            vmax=1.0,
            s=marker_sizes,
            alpha=0.88,
            edgecolors="#001A66",
            linewidths=0.55,
            label=f"Candidates ({len(df_cand):,} Mols | Size propto 3D Shape Sim)",
            zorder=3,
        )

        if not df_query.empty:
            qx, qy = df_query[x_col].iloc[0], df_query[y_col].iloc[0]
            ax.scatter(
                [qx],
                [qy],
                c="#E63946",
                s=420,
                marker="*",
                edgecolors="#1A1A1A",
                linewidths=2.2,
                label="Reference: EGCG (my_molecule)",
                zorder=10,
            )

        full_title = f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=12.5, fontweight="bold", pad=16, color="#1A202C", loc="left", linespacing=1.35)
        ax.set_xlabel(f"{x_col} (Morgan ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)
        ax.set_ylabel(f"{y_col} (Morgan ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)

        cbar = fig.colorbar(scatter_cand, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("2D Bioactivity Profile Similarity", fontsize=9.5, fontweight="bold", color="#1A202C", labelpad=10)

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return output_png


class MolecularSpacePipeline:
    def __init__(self, config: MolecularSpaceConfig | None = None, output_dir: Path | None = None) -> None:
        self.config: MolecularSpaceConfig = config or MolecularSpaceConfig()
        self.output_dir: Path = output_dir or DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Molecular space visualization module loaded successfully.")
