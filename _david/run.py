"""One-Click Runner for DrugEx Ligand-Based De Novo Drug Design Pipeline.

Executes the complete end-to-end LBDD pipeline with ZERO command-line arguments required.
Outputs are automatically written to `_david/output/<YYYYMMDD_HHMMSS>/...`.

Stages executed:
1. Stage 0: 3D reference conformer ensemble, 75-D ElectroShape moments & seed core Murcko scaffold.
2. Stage 1: Sequence-RNN global hit discovery via unconstrained policy-gradient RL (Regime 1).
3. Stage 2: Graph-Transformer fragment-conditioned lead optimization directly on query seed core (Regime 2).
4. Stage 3: Post-generation lead selection & diversity clustering (continuous desirability ranking, non-gating).
5. Stage 4: Publication benchmark suite & statistical figure generation.
6. Health & Fallback Audit Report.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

# Ensure repository root and _david/code are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _david.code.fallback_monitor import FallbackMonitor
from _david.code.pipeline import MoleculePipeline
from _david.code.prepare_query_conformer import MY_MOLECULE


def main() -> int:
    """Execute the end-to-end pipeline with zero command-line arguments.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="One-Click Runner for DrugEx Ligand-Based De Novo Drug Design Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--molecules", "-m", type=str, default=MY_MOLECULE, help="Active query seed SMILES string")
    parser.add_argument("--mode", choices=["full", "test", "benchmark_only"], default=None, help="Execution mode")
    parser.add_argument("--test", action="store_true", help="Shortcut for --mode test (1 epoch, 50 samples)")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="RL epochs per stage")
    parser.add_argument("--n-samples", "-n", type=int, default=None, help="Samples generated per RL epoch")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Custom output directory")
    args = parser.parse_args()

    # Determine execution mode: command-line flag > argument > env var > default "full"
    env_mode = os.getenv("DRUGEX_RUN_MODE", "full").lower()
    if args.test:
        mode = "test"
    elif args.mode:
        mode = args.mode.lower()
    else:
        mode = env_mode

    # Configure epochs and sample counts
    if mode == "test":
        epochs = args.epochs if args.epochs is not None else 1
        n_samples = args.n_samples if args.n_samples is not None else 50
    else:
        epochs = args.epochs if args.epochs is not None else 10
        n_samples = args.n_samples if args.n_samples is not None else 200

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out_dir if args.out_dir is not None else (SCRIPT_DIR / "output" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input molecules: explicitly passed > in_molecule.txt (if present) > default MY_MOLECULE
    in_mol_file = SCRIPT_DIR / "in_molecule.txt"
    if args.molecules == MY_MOLECULE and in_mol_file.exists():
        molecules_input = in_mol_file
    else:
        molecules_input = args.molecules

    print("=" * 80)
    print(" DRUGEX LBDD PIPELINE - 1-CLICK AUTOMATED RUNNER")
    print("=" * 80)
    print(f" Execution Mode:   {mode.upper()}")
    print(f" Query Molecule:   {molecules_input}")
    print(f" Target Output:    {output_dir}")
    print(f" Epochs / Stage:   {epochs}")
    print(f" Samples / Epoch:  {n_samples}")
    print(f" Started At:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        pipeline = MoleculePipeline(
            output_dir=output_dir,
            mode=mode,
            epochs=epochs,
            n_samples=n_samples,
        )
        summary = pipeline.run(molecules=molecules_input)

        print("\n" + "=" * 80)
        print(" 1-CLICK PIPELINE EXECUTION COMPLETED")
        print("=" * 80)
        print(f" Status:            SUCCESS")
        print(f" Run Directory:     {output_dir}")
        print(f" Summary Metadata:  {output_dir / 'pipeline_summary.json'}")
        print(f" Fallback Audit:    {output_dir / 'fallback_audit.json'}")
        print(f" Fallbacks Count:   {summary.get('fallbacks_count', 0)}")
        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution encountered an unhandled exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        FallbackMonitor.record_fallback(
            fallback_type="fatal_pipeline_error",
            message=str(e),
            context={"mode": mode, "output_dir": str(output_dir)},
        )
        FallbackMonitor.print_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())
