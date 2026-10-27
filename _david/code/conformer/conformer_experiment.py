from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.config import DEFAULT_MOLECULE
from _david.code.conformer import (
    ConformerConvergenceAnalyzer,
    ConformerGenerator,
    ConformerIOManager,
    ConformerStudyOrchestrator,
    ConformerVisualizer,
    ForceFieldVariant,
    PhysicalVariantFactory,
)

# Backward-compatibility aliases for legacy scripts
ConformerEnsembleBuilder = ConformerGenerator
ConvergenceVisualizer = ConformerVisualizer

__all__ = [
    "ForceFieldVariant",
    "PhysicalVariantFactory",
    "ConformerGenerator",
    "ConformerEnsembleBuilder",
    "ConformerConvergenceAnalyzer",
    "ConformerIOManager",
    "ConformerVisualizer",
    "ConvergenceVisualizer",
    "ConformerStudyOrchestrator",
    "run_full_study",
    "main",
]


def run_full_study(
    molecule_smiles: str = DEFAULT_MOLECULE,
    output_dir: Path | str | None = None,
    n_raw: int = 200,
) -> None:
    """Execute the complete conformer self-similarity and dielectric convergence experiment.

    Coordinates generation across 5 physical environments (vacuum, UFF, pocket, aqueous, distance-dependent),
    exports multi-model 3D files (.sdf, .pdb) for VS Code viewer extensions, evaluates ROCS convergence surfaces,
    and renders publication figures and interactive HTML widgets.
    """
    target_output = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "_david" / "output" / "conformer_study"
    )

    orchestrator = ConformerStudyOrchestrator(
        molecule_smiles=molecule_smiles,
        output_dir=target_output,
        n_raw_conformers=n_raw,
    )
    orchestrator.run_study()


def main() -> None:
    """Command-line interface for the conformer convergence study."""
    parser = argparse.ArgumentParser(
        description="Conformer Self-Similarity & Dielectric Convergence Benchmark"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=DEFAULT_MOLECULE,
        help="Target molecule SMILES (default: EGCG)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "_david" / "output" / "conformer_study"),
        help="Directory to store generated 3D models and figures",
    )
    parser.add_argument(
        "--n-raw",
        type=int,
        default=200,
        help="Number of initial ETKDG conformers to embed per environment",
    )
    args = parser.parse_args()

    run_full_study(
        molecule_smiles=args.smiles,
        output_dir=args.output_dir,
        n_raw=args.n_raw,
    )


if __name__ == "__main__":
    main()
