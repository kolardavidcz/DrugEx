"""Evaluation and Showcase Restructuring Script for Run 20261016_214024.

WHY:
Demonstrate the user's requirement:
1. Generate 500 molecules in final state before presenting and save to stage2_result_analysis.
2. Select top leads (best out of 500 molecules).
3. Restructure output folder: in 20261016_214024/ there will ONLY be:
   - property_distributions.png
   - lead_structures_2d.png
   - top_prioritized_leads.tsv (best out of 500 molecules)
   - rl_training_curves.png
   and all other files in subfolders (logs/, references/, stage1_graph_transformer/, stage2_result_analysis/).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.analysis import ResultAnalyzer
from _david.code.environment import DEFAULT_MOLECULE, create_environment
from _david.code.models import GraphTransformerOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager


def main() -> None:
    HardwareManager.init_runtime_environment()
    run_dir = PROJECT_ROOT / "_david" / "output" / "05_new_era" / "20261016_214024"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    ref_dir = run_dir / "references"
    ref_sdf = ref_dir / "query_molecule_3d.sdf"
    scaffold_tsv = ref_dir / "seed_core_scaffold.tsv"
    stage1_dir = run_dir / "stage1_graph_transformer"
    rl_dir = stage1_dir / "reinforcement_learning"
    peak_model_pkg = rl_dir / "david_agent_peak.pkg"
    if not peak_model_pkg.exists():
        peak_model_pkg = rl_dir / "david_agent.pkg"

    stage2_dir = run_dir / "stage2_result_analysis"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    gen_500_tsv = stage2_dir / "generated_500_molecules.tsv"

    print("=" * 70)
    print(f"EVALUATING AND RESTRUCTURING SHOWCASE: {run_dir.name}")
    print("=" * 70)

    # 1. Initialize optimizer and load peak model
    print("[1/4] Loading peak trained agent...")
    optimizer = GraphTransformerOptimizer(stage1_dir, gpu="auto")
    optimizer.AGENT.loadStatesFromFile(str(peak_model_pkg))

    # 2. Build environment and prepare 500 fragment prompts
    print("[2/4] Generating 500 candidate molecules from peak agent...")
    env, _ = create_environment(ref_sdf)
    loader, _, _ = optimizer.prepare_seed_fragments(
        DEFAULT_MOLECULE,
        n_prompts=500,
        scaffolds_path=scaffold_tsv,
    )
    smiles_list, frags_list = optimizer.AGENT.sample(loader)
    print(f"      Sampled {len(smiles_list)} raw molecules. Computing 3D ROCS & SA scores...")
    df = env.getScores(smiles_list, frags=frags_list)
    df.insert(0, std.SMILES, smiles_list)
    df.insert(1, std.FRAGS, frags_list)
    df = std.standardize(df)
    df.to_csv(gen_500_tsv, sep="\t", index=False)
    print(f"      Saved {len(df)} scored molecules to {gen_500_tsv.name}")

    # 3. Analyze 500 candidates and select top leads
    print("[3/4] Profiling generated 500 molecules and selecting top leads...")
    analyzer = ResultAnalyzer(run_dir)
    top_leads = analyzer.analyze(
        gen_500_tsv,
        ref_sdf=ref_sdf,
        top_k=10,
        render_pareto=False,
    )
    print(f"      Selected {len(top_leads)} top prioritized leads.")

    # 4. Restructure directory: showcase files in root, rest in subfolders
    print("[4/4] Restructuring folder layout for root showcase presentation...")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for log_name in ["pipeline.log", "pipeline_status.json", "run.pid"]:
        src = run_dir / log_name
        if src.is_file():
            shutil.move(str(src), str(logs_dir / log_name))

    # Ensure the 4 showcase files exist in the root
    shutil.copy2(stage2_dir / "property_distributions.png", run_dir / "property_distributions.png")
    shutil.copy2(stage2_dir / "lead_structures_2d.png", run_dir / "lead_structures_2d.png")
    shutil.copy2(stage2_dir / "top_prioritized_leads.tsv", run_dir / "top_prioritized_leads.tsv")
    rl_curves = rl_dir / "rl_training_curves.png"
    if rl_curves.is_file():
        shutil.copy2(rl_curves, run_dir / "rl_training_curves.png")

    print("=" * 70)
    print("[COMPLETE] Showcase folder restructure finished.")
    print("Top-level items in 20261016_214024/:")
    for p in sorted(run_dir.iterdir()):
        prefix = "📁 [DIR] " if p.is_dir() else "📄 [FILE]"
        print(f"  {prefix} {p.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
