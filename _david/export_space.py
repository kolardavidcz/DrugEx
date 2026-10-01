import sys
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david import receptor_similar
from _david.visualize_molecular_space import (
    SEED_SMILES,
    MolecularSpaceConfig,
    ChemicalSpaceDataset,
    ManifoldEmbeddingEngine,
    MolecularSpacePipeline
)

def run_export():
    config = MolecularSpaceConfig()
    pipeline = MolecularSpacePipeline(config=config)
    
    # 1. Version 1: Papyrus-only
    print("Exporting V1 Papyrus dataset...")
    pipeline_v1 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=False)
    result_v1 = pipeline_v1.run(
        config.query_smiles,
        min_paffinity=6.0,
        max_molecules=config.n_candidates,
        quality_filter=None,
        allow_chembl_fallback=False,
        show_progress=False
    )
    df_cand_v1 = pd.DataFrame(result_v1.molecules)
    if "smiles" in df_cand_v1.columns and "SMILES" not in df_cand_v1.columns:
        df_cand_v1["SMILES"] = df_cand_v1["smiles"]
    df_cand_v1["Group"] = "Papyrus Candidates (High-Affinity, pAffinity >= 6.0)"
    if "property_similarity" not in df_cand_v1.columns:
        df_cand_v1["property_similarity"] = 0.5
    if "source" in df_cand_v1.columns:
        df_cand_v1["Source"] = df_cand_v1["source"]
    else:
        df_cand_v1["Source"] = "Papyrus Bioactivity"
    if "target_accession" in df_cand_v1.columns:
        df_cand_v1["Target"] = df_cand_v1["target_accession"]
    else:
        df_cand_v1["Target"] = "Shared Receptors"
    
    keep_cols = ["SMILES", "Group", "property_similarity", "Source", "Target"]
    for c in ["pAffinity", "MW", "logP", "TPSA", "QED"]:
        if c in df_cand_v1.columns:
            keep_cols.append(c)
    df_cand_v1 = df_cand_v1[keep_cols].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    
    query_df_v1 = pipeline.dataset_manager.format_query_molecule(config.query_smiles)
    exclude_v1 = set(df_cand_v1["SMILES"]).union(set(query_df_v1["SMILES"]))
    bg_df_v1 = pipeline.dataset_manager.sample_background_molecules(
        n_samples=config.n_background,
        exclude_smiles=exclude_v1,
        random_seed=42
    )
    
    table_dir_v1 = SCRIPT_DIR / "cache_export" / "v1"
    table_v1, combined_df_v1 = pipeline.dataset_manager.assemble_unified_table(
        bg_df_v1, df_cand_v1, query_df_v1, table_name="v1_export", store_dir=table_dir_v1
    )
    x_col_v1, y_col_v1, enriched_df_v1 = ManifoldEmbeddingEngine.compute_tsne_coordinates(table_v1, config)
    enriched_df_v1 = pipeline._enrich_coords_with_metadata(enriched_df_v1, combined_df_v1)
    
    out_v1 = SCRIPT_DIR / "v1_space_raw.csv"
    enriched_df_v1.to_csv(out_v1, index=False)
    print(f"Saved V1 space ({len(enriched_df_v1)} rows, x_col={x_col_v1}, y_col={y_col_v1}) to {out_v1}")

    # 2. Version 2: Papyrus + ChEMBL
    print("Exporting V2 Papyrus + ChEMBL dataset...")
    pipeline_v2 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=True)
    result_v2 = pipeline_v2.run(
        config.query_smiles,
        min_paffinity=5.0,
        max_molecules=config.n_candidates,
        quality_filter=None,
        allow_chembl_fallback=True,
        show_progress=False
    )
    df_cand_v2 = pd.DataFrame(result_v2.molecules)
    if "smiles" in df_cand_v2.columns and "SMILES" not in df_cand_v2.columns:
        df_cand_v2["SMILES"] = df_cand_v2["smiles"]
    df_cand_v2["Group"] = "Papyrus + ChEMBL Candidates (Broad Space, pIC50 >= 5.0)"
    if "property_similarity" not in df_cand_v2.columns:
        df_cand_v2["property_similarity"] = 0.5
    if "source" in df_cand_v2.columns:
        df_cand_v2["Source"] = df_cand_v2["source"]
    else:
        df_cand_v2["Source"] = "Papyrus + ChEMBL"
    if "target_accession" in df_cand_v2.columns:
        df_cand_v2["Target"] = df_cand_v2["target_accession"]
    else:
        df_cand_v2["Target"] = "Shared Receptors"
        
    df_cand_v2 = df_cand_v2[keep_cols].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    
    query_df_v2 = pipeline.dataset_manager.format_query_molecule(config.query_smiles)
    exclude_v2 = set(df_cand_v2["SMILES"]).union(set(query_df_v2["SMILES"]))
    bg_df_v2 = pipeline.dataset_manager.sample_background_molecules(
        n_samples=config.n_background,
        exclude_smiles=exclude_v2,
        random_seed=101
    )
    
    table_dir_v2 = SCRIPT_DIR / "cache_export" / "v2"
    table_v2, combined_df_v2 = pipeline.dataset_manager.assemble_unified_table(
        bg_df_v2, df_cand_v2, query_df_v2, table_name="v2_export", store_dir=table_dir_v2
    )
    x_col_v2, y_col_v2, enriched_df_v2 = ManifoldEmbeddingEngine.compute_tsne_coordinates(table_v2, config)
    enriched_df_v2 = pipeline._enrich_coords_with_metadata(enriched_df_v2, combined_df_v2)
    
    out_v2 = SCRIPT_DIR / "v2_space_raw.csv"
    enriched_df_v2.to_csv(out_v2, index=False)
    print(f"Saved V2 space ({len(enriched_df_v2)} rows, x_col={x_col_v2}, y_col={y_col_v2}) to {out_v2}")

if __name__ == "__main__":
    run_export()
