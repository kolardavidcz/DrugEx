"""Inspect saved reference counts and measure fresh public-scorer batch calls.

Count inspection never regenerates old measurements. Timing uses one fixed EGCG
query and different frozen-panel candidates: it measures cost, not self-recovery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import socket
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from _david.code.chemistry import Optimization, prepare_candidate_molecule, standardize_smiles
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.conformer.visualization import render_reference_counts
from _david.code.environment import create_rocs_scorer
from _david.code.reference import PrepareMolecule


def score_chunk(job):
    """Parallelize complete public calls, with no nested alignment processes."""
    reference, smiles, minimize, protonate = job
    scorer = create_rocs_scorer(Path(reference), max_conformers=20, max_isomers=4,
                                minimize_candidates=minimize,
                                optimization=Optimization("MMFF94s", 78.4),
                                conformer_threads=1, n_jobs=1)
    scorer.conformer_generator.embedding_rmsd = None
    scorer.conformer_generator.protonate = protonate
    return (scorer.getScores(smiles)[:, 0] / 2).tolist()


def inspect_counts(source: Path, output: Path) -> None:
    """Keep all 512 inputs, including failed/colourless cases; count each once."""
    rows, hashes = [], {}
    for path in sorted((source / "checkpoints").glob("*/complete.json")):
        checkpoint = json.loads(path.read_text())
        hashes[str(path.relative_to(source))] = digest(path)
        for attempts, info in checkpoint["reference"].items():
            pool = path.parent / f"reference_{attempts}_pool.sdf"
            count = len(Chem.SDMolSupplier(str(pool), removeHs=False)) if pool.exists() and pool.stat().st_size else 0
            selections = checkpoint["reference_selections"][attempts]
            rows.append(dict(
                molecule_id=checkpoint["case"]["molecule_id"],
                name=checkpoint["case"]["name"], attempts=int(attempts),
                enumerated_isomers=info["enumerated_isomers"],
                embedded_reported=info["embedded_successful_identities"],
                converged_valid_reported=info["converged_successful_identities"],
                energy_survivors=count,
                retained_rmsd_05=len(selections["0.5"]),
                retained_rmsd_125=len(selections["1.25"]),
                failed_stereo_preparations=info["failed_identities"],
                color_excluded=checkpoint["color_excluded"],
            ))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["molecule_id", "attempts"]).any() or len(frame) != 1024:
        raise ValueError("Expected two reference preparations for all 512 inputs")
    frame.to_csv(output / "raw_data/reference_counts.csv", index=False)
    metrics = ["converged_valid_reported", "energy_survivors", "retained_rmsd_05", "retained_rmsd_125"]
    summary = []
    for attempts, group in frame.groupby("attempts"):
        for metric in metrics:
            values = group[metric]
            summary.append(dict(
                attempts=attempts, metric=metric, inputs=len(values),
                zeros=int(values.eq(0).sum()), mean=values.mean(), median=values.median(),
                p05=values.quantile(.05), p95=values.quantile(.95), maximum=values.max(),
                total=int(values.sum()),
            ))
    pd.DataFrame(summary).to_csv(output / "raw_data/reference_count_summary.csv", index=False)
    render_reference_counts(frame, output)
    atomic_json(output / "raw_data/count_provenance.json", dict(
        source=str(source), checkpoint_sha256=hashes, inputs=512,
        scope="Descriptive counts from the completed historical chemical-state policy; no regeneration. Converged counts are identity-valid counts reported by successful stereo preparations, before the 8 kcal/mol window. Failed stereo preparations are separate; their intermediate counts were not always saved.",
        script_sha256=digest(__file__),
    ))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def time_batches(source: Path, output: Path, batch_size: int, repeats: int, protonate: bool) -> None:
    """Direct getScores timing; shared query, serial generation, parallel alignment.

    Pool startup, SDF I/O and actual candidate minimization are included. Reference
    preparation and scorer construction are one-time costs outside the stopwatch.
    Compare the same inputs/order at every setting; failures stay in the batch.
    """
    raw = output / "raw_data"
    source_files = [Path(__file__), PROJECT_ROOT / "_david/code/chemistry.py",
                    PROJECT_ROOT / "_david/code/reference.py", PROJECT_ROOT / "_david/code/environment.py"]
    before = {str(p): digest(p) for p in source_files}
    panel = pd.read_csv(source / "panel.csv").head(batch_size)
    if len(panel) != batch_size:
        raise ValueError("Batch exceeds the frozen panel")
    profile = Optimization("MMFF94s", 78.4)
    prepared = prepare_candidate_molecule(standardize_smiles(DEFAULT_MOLECULE), protonate=protonate)
    preparer = PrepareMolecule(num_conformers=50, max_reference_conformers=3,
                              random_seed=43, rmsd_cutoff=None, embedding_rmsd=None,
                              num_threads=1, optimization=profile)
    ref_path = preparer.generate_reference_conformers(prepared, raw / "timing_query.sdf")
    refs = list(Chem.SDMolSupplier(str(ref_path), removeHs=False))
    panel.to_csv(raw / "timing_inputs.csv", index=False)
    settings = [(False, 1, "native"), (True, 1, "native"),
                (False, 8, "native"), (True, 8, "native"),
                (False, 8, "whole_call_pool"), (True, 8, "whole_call_pool")]
    rows, score_rows = [], []
    manifest = dict(
        state="running", purpose="RL scoring throughput diagnostic, NOT self-recovery or training",
        host=socket.gethostname(), python=platform.python_version(), rdkit=rdBase.rdkitVersion,
        dimorphite_dl=version("dimorphite_dl"), batch_size=batch_size, repeats=repeats,
        candidates="First frozen panel inputs, unchanged order, one common EGCG query",
        reference_count=len(refs), reference_preparation="R50/I1; seed43; first3 energy-ordered survivors; no RMSD; MMFF94s constant epsilon78.4",
        candidate_preparation="C20/I4; seed42; no embedding/post RMSD; same factory as RL",
        ph_reassignment=protonate, query_formal_charge=Chem.GetFormalCharge(prepared),
        source_sha256=before, git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        cpu_note="Eight alignment workers, one embedding thread; production generates the batch before starting its alignment pool. Wall time includes pool startup and SDF I/O.",
    )
    atomic_json(raw / "timing_manifest.json", manifest)
    for repeat in range(repeats):
        # Reverse settings on alternate repeats to reduce a fixed order effect.
        for minimize, workers, engine in settings[::1 if repeat % 2 == 0 else -1]:
            scorer = create_rocs_scorer(refs, max_conformers=20, max_isomers=4,
                                        minimize_candidates=minimize, optimization=profile,
                                        conformer_threads=1, n_jobs=workers)
            generator = scorer.conformer_generator
            generator.embedding_rmsd = None
            generator.protonate = protonate
            generate = generator.genConformers
            generation = {}

            def timed_generate(smiles, directory, generate=generate, generation=generation):
                started = time.perf_counter()
                path = generate(smiles, directory)
                generation["seconds"] = time.perf_counter() - started
                generation["records"] = len(Chem.SDMolSupplier(path, removeHs=False)) if Path(path).stat().st_size else 0
                return path

            generator.genConformers = timed_generate
            tick = time.perf_counter()
            if engine == "native":
                scores = scorer.getScores(panel.smiles.tolist())[:, 0] / 2
            else:
                # Fixed chunks preserve order; the selected 32 inputs are unique.
                # This is an engineering comparison, not a deployed RL adapter.
                smiles = panel.smiles.tolist()
                jobs = [(str(ref_path.resolve()), smiles[i:i+4], minimize, protonate)
                        for i in range(0, len(smiles), 4)]
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                    scores = np.array([score for chunk in pool.map(score_chunk, jobs) for score in chunk])
            elapsed = time.perf_counter() - tick
            stage = generation.get("seconds", float("nan"))
            rows.append(dict(repeat=repeat, minimize=minimize, workers=workers, engine=engine,
                             batch_size=batch_size, actual_conformers=generation.get("records"),
                             batch_seconds=elapsed, generation_and_minimization_seconds=stage,
                             alignment_io_pool_seconds=elapsed-stage,
                             smiles_per_second=batch_size/elapsed, zero_scores=int((scores == 0).sum())))
            score_rows.extend(dict(repeat=repeat, minimize=minimize, workers=workers, engine=engine,
                                   molecule_id=int(mid), score=float(score))
                              for mid, score in zip(panel.molecule_id, scores))
            pd.DataFrame(rows).to_csv(raw / "batch_timing.csv", index=False)
            pd.DataFrame(score_rows).to_csv(raw / "batch_scores.csv", index=False)
            print(json.dumps(rows[-1]), flush=True)
    if before != {str(p): digest(p) for p in source_files}:
        raise ValueError("Source changed during batch timing")
    # Parallelism must not change scores for the same chemistry/embedding policy.
    scores = pd.DataFrame(score_rows)
    spread = scores.groupby(["minimize", "molecule_id"]).score.agg(lambda x: x.max()-x.min()).max()
    if spread > 1e-8:
        raise ValueError(f"Serial/parallel score disagreement: {spread}")
    manifest.update(state="complete", max_serial_parallel_score_difference=float(spread),
                    output_sha256={p.name: digest(p) for p in (raw / "batch_timing.csv", raw / "batch_scores.csv", ref_path)})
    atomic_json(raw / "timing_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["counts", "timing"])
    parser.add_argument("--source", type=Path, required=True, help="Completed calibration raw_data directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--protonate", action="store_true", help="Explicit strict pH7.4 policy on both sides")
    args = parser.parse_args()
    (args.output / "raw_data").mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    if args.mode == "counts":
        inspect_counts(args.source, args.output)
    else:
        time_batches(args.source, args.output, args.batch_size, args.repeats, args.protonate)


if __name__ == "__main__":
    main()
