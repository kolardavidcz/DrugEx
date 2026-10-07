"""IDE Direct Runner for DrugEx Ligand-Based De Novo Drug Design.

Press F5 or Run in your IDE (VS Code / PyCharm) to execute directly.
Easily tweak the hyperparameters below.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.pipeline import DEFAULT_MOLECULE, MoleculePipeline

# =============================================================================
# HYPERPARAMETERS (TWEAK FREELY IN IDE)
# =============================================================================
# Mode: 'full' (train RL), 'test' (1-epoch sanity check), 'benchmark_only' (rescore)
MODE: str = "full"

# RL training epochs (recommend: 10-20 on dichtator GPU, 1-2 on local CPU)
EPOCHS: int = 10

# Number of molecules sampled per RL epoch
N_SAMPLES: int = 200

# Target query template (default: EGCG flavan-3-ol ester proof molecule)
# Can be SMILES string or Path to .sdf / .smi file
MOLECULE: str = DEFAULT_MOLECULE

# Output directory: None auto-generates timestamped folder in _david/output/
OUTPUT_DIR: Path | None = None


def main() -> None:
    """Execute the DrugEx pipeline with the pre-set hyperparameters above."""
    print("=" * 80)
    print(" DRUGEX LBDD: IDE EXECUTION LAUNCHER")
    print(f" • Mode:       {MODE}")
    print(f" • Epochs:     {EPOCHS}")
    print(f" • Samples:    {N_SAMPLES}")
    print(f" • Target:     {MOLECULE[:50]}...")
    print("=" * 80)

    pipeline = MoleculePipeline(
        output_dir=OUTPUT_DIR,
        mode=MODE,
        epochs=EPOCHS,
        n_samples=N_SAMPLES,
    )
    pipeline.run(MOLECULE)


if __name__ == "__main__":
    main()
