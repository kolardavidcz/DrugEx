"""One-Click Runner for DrugEx Ligand-Based De Novo Drug Design Pipeline.

Executes the complete end-to-end LBDD pipeline with ZERO command-line arguments required.
Outputs are automatically written to `_david/output/<YYYYMMDD_HHMMSS>/...`.

Stages executed:
1. Stage 0: 3D reference conformer ensemble & 75-D ElectroShape moments for query seed ligand.
2. Stage 1: Sequence-RNN global hit discovery via policy-gradient RL with anti-cheating scalarization.
3. Stage 2: Bemis-Murcko chemotype funnel (gated two-tier 3D ROCS + PAINS removal + Butina clustering).
4. Stage 3: Graph-Transformer lead optimization on discovered scaffolds.
5. Stage 4: Publication benchmark suite & statistical figure generation.
6. Health & Fallback Audit Report.
"""
from __future__ import annotations

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check execution mode: default to 'full', allow 'test' via env var or --test flag for quick verification
    mode = os.getenv("DRUGEX_RUN_MODE", "full").lower()
    if "--test" in sys.argv:
        mode = "test"

    print("=" * 80)
    print(" DRUGEX LBDD PIPELINE - 1-CLICK AUTOMATED RUNNER")
    print("=" * 80)
    print(f" Execution Mode:   {mode.upper()}")
    print(f" Query Molecule:   {MY_MOLECULE}")
    print(f" Target Output:    {output_dir}")
    print(f" Started At:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        pipeline = MoleculePipeline(
            output_dir=output_dir,
            mode=mode,
        )
        summary = pipeline.run(molecules=MY_MOLECULE)

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
