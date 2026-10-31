"""Realistic candidate batch scoring benchmark.

Measures candidate conformer embedding, optional force-field minimization,
and ROCS alignment throughput across realistic RL candidate batches (32-256 SMILES)
without conflating one-time Stage 0 reference generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.chemistry import (
    CandidateConformerGenerator,
    Optimization,
    enable_double_precision_mol_pickling,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE
from _david.code.environment import PrepareMolecule, create_rocs_scorer
from _david.code.utils import HardwareManager

SAMPLE_CANDIDATE_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",  # Omeprazole
    "CNCCC(c1ccccc1)Oc1ccc(C(F)(F)F)cc1",  # Fluoxetine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "Clc1ccc(cc1)C(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",  # DDT derivative
    "COc1ccc(cc1)CCN(C)CCCC(C#N)(c1ccc(OC)c(OC)c1)C(C)C",  # Verapamil
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # Salbutamol
    "Cc1ncc(CCO)s1",  # Thiamine fragment
    "O=C(O)c1ccccc1O",  # Salicylic acid
    "COc1cc(O)c2c(c1)oc(-c1cc(O)c(O)c(O)c1)c(O)c2=O",  # Myricetin fragment
    "COc1cc2c(cc1OC)c1ccc(=O)n(C)c1cc2",  # Bicyclic quinolone
    "O=C1NC(=O)c2ccccc21",  # Phthalimide
    "CC(C)NC(=O)c1ccc(cc1)N(CC)CC",  # Amide probe
)


def get_benchmark_smiles(target_size: int = 32) -> list[str]:
    """Return a list of diverse candidate SMILES duplicated/extended to ``target_size``."""
    base = list(SAMPLE_CANDIDATE_SMILES)
    if target_size <= len(base):
        return base[:target_size]
    repeats = (target_size + len(base) - 1) // len(base)
    return (base * repeats)[:target_size]


def prepare_stage0_reference(
    reference_smiles: str = DEFAULT_MOLECULE,
    output_path: Path | str | None = None,
    num_conformers: int = 50,
    max_retained: int = 4,
    optimization: Optimization | None = None,
) -> list[Chem.Mol]:
    """Execute one-time Stage 0 reference generation and diversity ordering.

    This one-time setup cost (~1-5 seconds) is amortized over hundreds of RL epochs
    and must not be conflated with per-batch candidate scoring throughput.
    """
    opt = optimization or Optimization("MMFF94s", 78.4)
    preparer = PrepareMolecule(
        num_conformers=num_conformers,
        max_reference_conformers=max_retained,
        random_seed=104729,
        selection_mode="diverse",
        optimization=opt,
    )
    temp_dir: Path | None = None
    if output_path is None:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="stage0_ref_"))
        target = temp_dir / "reference_bank.sdf"
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        clean_smi = standardize_smiles(reference_smiles)
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        saved_path = preparer.generate_reference_conformers(mol, target)
        records = [
            m for m in Chem.SDMolSupplier(str(saved_path), removeHs=False) if m is not None
        ]
        if not records:
            raise RuntimeError("Stage 0 reference preparation produced 0 valid conformers")
        return records
    finally:
        if temp_dir is not None and temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def measure_batch_scoring(
    smiles_list: Sequence[str],
    references: Sequence[Chem.Mol],
    workers: int = 1,
    max_conformers: int = 15,
    minimize_candidates: bool = False,
    optimization: Optimization | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Measure single-batch candidate conformer embedding, minimization, and ROCS alignment.

    Instruments each stage separately to measure candidate generation throughput and
    verify multi-core scaling.
    """
    enable_double_precision_mol_pickling()
    opt = optimization or Optimization("MMFF94s", 78.4)
    scorer = create_rocs_scorer(
        references,
        max_conformers=max_conformers,
        max_isomers=4,
        minimize_candidates=minimize_candidates,
        optimization=opt,
        conformer_threads=1,
        n_jobs=workers,
    )
    generator: CandidateConformerGenerator = scorer.conformer_generator
    generator.embedding_rmsd = None
    generator.protonate = False
    generator.n_jobs = workers

    original_gen = generator.genConformers
    timings: dict[str, float] = {
        "embedding_seconds": 0.0,
        "minimization_seconds": 0.0,
        "conformer_count": 0.0,
    }

    def timed_genConformers(smiles: Sequence[str], out_dir: str | Path) -> str:
        t0 = time.perf_counter()
        sdf = original_gen(smiles, out_dir)
        elapsed = time.perf_counter() - t0
        timings["embedding_seconds"] = elapsed
        if Path(sdf).is_file() and Path(sdf).stat().st_size > 0:
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
            timings["conformer_count"] = float(sum(1 for m in suppl if m is not None))
        return sdf

    generator.genConformers = timed_genConformers

    t_batch_start = time.perf_counter()
    raw_scores = scorer.getScores(list(smiles_list))[:, 0]
    total_batch_seconds = time.perf_counter() - t_batch_start

    scores = raw_scores / 2.0  # Normalized [0, 1]
    alignment_seconds = max(0.0, total_batch_seconds - timings["embedding_seconds"])
    timings["total_batch_seconds"] = total_batch_seconds
    timings["alignment_seconds"] = alignment_seconds
    timings["molecules_per_second"] = len(smiles_list) / max(1e-6, total_batch_seconds)
    timings["workers"] = float(workers)
    timings["batch_size"] = float(len(smiles_list))
    timings["minimize"] = 1.0 if minimize_candidates else 0.0

    return scores, timings


def run_batch_probe(
    batch_sizes: Sequence[int] = (16, 32, 64),
    worker_counts: Sequence[int] = (1, 4),
    minimize_candidates: bool = False,
    repeats: int = 1,
    reference_smiles: str = DEFAULT_MOLECULE,
) -> tuple[pd.DataFrame, float]:
    """Run batch scoring probe across multiple batch sizes and worker counts.

    Returns the performance DataFrame and the maximum absolute score difference
    between serial (workers=1) and parallel executions (determinism verification).
    """
    enable_double_precision_mol_pickling()
    ref_records = prepare_stage0_reference(reference_smiles=reference_smiles)

    rows: list[dict[str, Any]] = []
    max_score_diff = 0.0

    for b_size in batch_sizes:
        batch_smiles = get_benchmark_smiles(b_size)
        baseline_scores: np.ndarray | None = None

        for workers in worker_counts:
            for rep in range(repeats):
                scores, metrics = measure_batch_scoring(
                    batch_smiles,
                    ref_records,
                    workers=workers,
                    minimize_candidates=minimize_candidates,
                )
                if workers == 1 and rep == 0:
                    baseline_scores = scores.copy()
                elif baseline_scores is not None:
                    diff = float(np.abs(scores - baseline_scores).max())
                    if diff > max_score_diff:
                        max_score_diff = diff

                row = dict(
                    batch_size=b_size,
                    workers=workers,
                    repeat=rep,
                    minimize=minimize_candidates,
                    total_seconds=metrics["total_batch_seconds"],
                    embedding_seconds=metrics["embedding_seconds"],
                    alignment_seconds=metrics["alignment_seconds"],
                    molecules_per_second=metrics["molecules_per_second"],
                    conformer_count=int(metrics["conformer_count"]),
                    mean_score=float(scores.mean()),
                )
                rows.append(row)

    df = pd.DataFrame(rows)
    # Compute speedup relative to workers=1 baseline within same batch size and minimize setting
    speedups: list[float] = []
    for _, r in df.iterrows():
        b = r["batch_size"]
        m = r["minimize"]
        base_sub = df[(df.batch_size == b) & (df.minimize == m) & (df.workers == 1)]
        if not base_sub.empty:
            base_t = base_sub.total_seconds.mean()
            speedups.append(base_t / max(1e-6, r["total_seconds"]))
        else:
            speedups.append(1.0)
    df["speedup"] = speedups

    return df, max_score_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Batch sizes to evaluate (default: 32 64)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, HardwareManager.get_safe_parallel_jobs()],
        help="Worker process counts to evaluate (default: 1 and safe host core count)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Enable candidate force-field minimization (default: False, RL default)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of evaluation repeats per setting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory to save results CSV and JSON summary",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(
        f"Running Candidate Batch Scoring Benchmark:\n"
        f"  Batch sizes: {args.batch_sizes}\n"
        f"  Worker counts: {args.workers}\n"
        f"  Minimize candidates: {args.minimize}\n"
    )

    df, max_diff = run_batch_probe(
        batch_sizes=args.batch_sizes,
        worker_counts=args.workers,
        minimize_candidates=args.minimize,
        repeats=args.repeats,
    )

    print("\nBenchmark Results:")
    summary_cols = [
        "batch_size",
        "workers",
        "total_seconds",
        "embedding_seconds",
        "alignment_seconds",
        "molecules_per_second",
        "speedup",
    ]
    print(df[summary_cols].to_string(index=False))
    print(f"\nMax Serial-Parallel Score Discrepancy: {max_diff:.2e} (Invariant: <= 1e-8)")

    if max_diff > 1e-8:
        raise ValueError(
            f"Serial/parallel score discrepancy {max_diff} exceeds allowable threshold 1e-8!"
        )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output / "batch_probe_results.csv", index=False)
        meta = {
            "batch_sizes": args.batch_sizes,
            "workers": args.workers,
            "minimize": args.minimize,
            "max_score_diff": max_diff,
            "passed_determinism": max_diff <= 1e-8,
        }
        (args.output / "batch_probe_manifest.json").write_text(json.dumps(meta, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
