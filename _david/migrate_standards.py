"""Automated Meta-Script for Column Standardization and Historical Result Re-generation.

WHY:
Historical DrugEx runs and intermediate artifacts across `_david/output/` recorded
inconsistent column names (e.g., dynamic ROCS scorer variants like
'RDKit_Supermol_TanimotoCombo_oe', 'RDKit_Aggregate_8refs_TanimotoCombo_oe',
'Desirability', duplicate 'SA'/'SAscore' columns, 'Frags', and 'logP').
Rather than littering runtime code with fuzzy fallback functions, this script
migrates all tabular data to the single source of truth (`std`) and re-generates
all downstream visualizations so that historical figures display true metrics
(ROCS, Desired, MW, QED, SA) instead of zero-fallbacks.

WHAT:
1. Traverses all TSV/CSV files across `_david/output/`.
2. Normalizes column headers according to `std` schema.
3. Automatically prunes duplicate columns (e.g. dropping redundant 'SAscore' when 'SA' is present).
4. Re-renders `lead_structures_2d.png`, `pareto_frontier.png`, and `pareto_epoch_evolution.png`
   for all historical runs using canonical data.
5. Rebuilds the unified `lead_structures_gallery.html` and `lead_structures_comparison.md`.

HOW:
Direct in-place pandas-based header and column rewriting followed by calls into
`_david.code.visualization` rendering pipelines.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from _david.code.environment import DEFAULT_MOLECULE
from _david.code.schema import std
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_lead_structures_2d,
    render_pareto_frontier,
)


class StandardsMigrator:
    """Orchestrates recursive schema standardization and figure re-generation."""

    def __init__(self, target_dir: Path | str) -> None:
        self.target_dir = Path(target_dir).resolve()
        self.migrated_files: list[Path] = []
        self.regenerated_images: list[Path] = []

    def standardize_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Normalize DataFrame columns and remove duplicates. Returns (df, changed)."""
        if df.empty:
            return df, False

        orig_cols = list(df.columns)
        df_out = std.standardize(df)
        new_cols = list(df_out.columns)
        changed = orig_cols != new_cols
        return df_out, changed

    def migrate_file(self, file_path: Path) -> bool:
        """Standardize a single TSV/CSV file in place."""
        if not file_path.is_file() or file_path.suffix.lower() not in {".tsv", ".csv"}:
            return False

        sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
        try:
            # Read first line to inspect header before full parse
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
            if not first_line or "\t" not in first_line and sep == "\t":
                return False

            df = pd.read_csv(file_path, sep=sep, low_memory=False)
            df_std, changed = self.standardize_dataframe(df)

            if changed:
                df_std.to_csv(file_path, sep=sep, index=False)
                self.migrated_files.append(file_path)
                return True
        except Exception as e:
            print(f"[WARN] Failed to migrate {file_path.name}: {e}")
            return False
        return False

    def migrate_all_tabular(self) -> int:
        """Scan target directory and migrate all tabular files."""
        print(f"\n[1/3] Scanning and migrating tabular files in {self.target_dir}...")
        count = 0
        for p in self.target_dir.rglob("*.tsv"):
            if self.migrate_file(p):
                count += 1
        for p in self.target_dir.rglob("*.csv"):
            if self.migrate_file(p):
                count += 1
        print(f"      Migrated {count} tabular files to standard schema.")
        return count

    def regenerate_run_visualizations(self) -> int:
        """Re-render lead structures, Pareto frontiers, and epoch evolution for all runs."""
        print("\n[2/3] Re-generating visual artifacts from standardized data...")
        img_count = 0

        # Find all stage2 lead tables
        for leads_tsv in self.target_dir.rglob("top_prioritized_leads.tsv"):
            analysis_dir = leads_tsv.parent
            run_dir = analysis_dir.parent
            if run_dir.name == "stage1_graph_transformer":
                run_dir = run_dir.parent

            try:
                leads_df = pd.read_csv(leads_tsv, sep="\t")
                if not leads_df.empty and std.SMILES in leads_df.columns:
                    # 1. Re-render lead structures 2D
                    out_png = analysis_dir / "lead_structures_2d.png"
                    res = render_lead_structures_2d(
                        leads_df,
                        output_png=out_png,
                        reference=DEFAULT_MOLECULE,
                        max_leads=10,
                    )
                    if res and res.exists():
                        self.regenerated_images.append(res)
                        img_count += 1

                    # 2. Re-render Pareto Frontier if Pareto_Rank exists
                    if std.PARETO_RANK in leads_df.columns:
                        p_png = analysis_dir / "pareto_frontier.png"
                        p_res = render_pareto_frontier(leads_df, output_png=p_png)
                        if p_res and p_res.exists():
                            self.regenerated_images.append(p_res)
                            img_count += 1
            except Exception as e:
                print(f"[WARN] Error re-rendering leads for {run_dir.name}: {e}")

            # 3. Re-render Pareto Epoch Evolution if smiles.tsv exists
            for smiles_tsv in run_dir.rglob("*_smiles.tsv"):
                try:
                    evo_png = smiles_tsv.parent / "pareto_epoch_evolution.png"
                    e_res = render_epoch_pareto_evolution(smiles_tsv, output_png=evo_png)
                    if e_res and e_res.exists():
                        self.regenerated_images.append(e_res)
                        img_count += 1
                except Exception as e:
                    print(f"[WARN] Error re-rendering evolution for {smiles_tsv.name}: {e}")

        print(f"      Re-generated {img_count} visual artifacts.")
        return img_count

    def rebuild_gallery(self) -> Path:
        """Generate unified HTML gallery and Markdown summary comparing all lead structures."""
        print("\n[3/3] Rebuilding lead structures gallery and Markdown summary...")
        out_html = self.target_dir / "lead_structures_gallery.html"
        out_md = self.target_dir / "lead_structures_comparison.md"

        runs_info: list[dict[str, str]] = []
        for leads_png in sorted(self.target_dir.rglob("lead_structures_2d.png")):
            rel_path = leads_png.relative_to(self.target_dir)
            run_name = rel_path.parts[0]
            if len(rel_path.parts) > 2:
                run_name = f"{rel_path.parts[0]} / {rel_path.parts[1]}"

            try:
                with open(leads_png, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                runs_info.append(
                    {
                        "name": run_name,
                        "rel_path": str(rel_path),
                        "b64": b64_data,
                    }
                )
            except Exception as e:
                print(f"[WARN] Failed to read image {leads_png}: {e}")

        # Write HTML gallery
        cards_html = []
        for item in runs_info:
            cards_html.append(
                f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; margin-bottom: 28px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h3 style="margin-top: 0; color: #0f172a; font-family: system-ui, sans-serif; font-size: 17px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px;">
                📂 {item['name']}
            </h3>
            <p style="color: #64748b; font-size: 13px; font-family: monospace; margin: 4px 0 14px 0;">{item['rel_path']}</p>
            <div style="text-align: center; overflow-x: auto;">
                <img src="data:image/png;base64,{item['b64']}" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #f8fafc;" alt="{item['name']}" />
            </div>
        </div>
                """
            )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DrugEx Standardized Lead Structures Gallery</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px;">
    <div style="max-width: 1200px; margin: 0 auto;">
        <header style="margin-bottom: 24px;">
            <h1 style="font-size: 26px; color: #0f172a; margin-bottom: 6px;">DrugEx Multi-Objective Lead Structures Gallery</h1>
            <p style="color: #64748b; font-size: 14px; margin: 0;">Standardized across all experimental eras (ROCS, Desired, MW, QED, SA).</p>
        </header>
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
        out_html.write_text(html_content, encoding="utf-8")

        # Write Markdown comparison
        md_lines = [
            "# Lead Structures Visual Comparison (Standardized)",
            "",
            "This document collects all historical runs re-generated under the canonical `std` schema.",
            "",
            "| Run Era / Name | Rel Path | Image |",
            "| :--- | :--- | :--- |",
        ]
        for item in runs_info:
            md_lines.append(
                f"| **{item['name']}** | `{item['rel_path']}` | ![{item['name']}]({item['rel_path']}) |"
            )
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        print(f"      Saved gallery to {out_html.name} and summary to {out_md.name}.")
        return out_html

    def run(self) -> None:
        """Execute full standardization and re-generation pipeline."""
        print("=" * 70)
        print("DRUGEX META-MIGRATION: STANDARDIZING ALL _david/ FILES")
        print("=" * 70)
        n_tab = self.migrate_all_tabular()
        n_img = self.regenerate_run_visualizations()
        self.rebuild_gallery()
        print("=" * 70)
        print(f"[COMPLETE] Migrated {n_tab} data tables, regenerated {n_img} figures.")
        print("=" * 70)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "output"
    migrator = StandardsMigrator(target)
    migrator.run()
