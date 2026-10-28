"""Checkpointed self-recovery: each input has its own independent reference pool.

No RNN training. Reference and candidate counts are independent experimental
axes around the shared AQME-style sampling plan. All settings are recorded.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolAlign, rdShapeAlign

from drugex.training.scorers import rocs_rdkit as public_rocs
from _david.code.chemistry import check_identity, optimize_conformers
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.conformer.mega import METHODS, SEEDS, prepared_molecule
from _david.code.conformer.sampling import plan_sampling, deduplicate_reference
from _david.code.environment import PrepareMolecule, create_rocs_scorer
from _david.code.visualization import render_self_recovery


SCALES = (0.5, 1.0, 2.0)
ISOMER_CAPS = (1, 3, 5, 10, 25)
REFERENCE_CAPS = (1, 2, 4, 8, 0)
SOURCE_PATHS = (
    "_david/code/conformer/self_recovery.py",
    "_david/code/conformer/sampling.py",
    "_david/code/conformer/mega.py",
    "_david/code/chemistry.py",
    "_david/code/environment.py",
    "_david/code/config.py",
    "_david/code/visualization.py",
    "drugex/training/scorers/conformer_generators.py",
    "drugex/training/scorers/rocs_rdkit.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_name(job):
    if job["kind"] == "reference":
        return f"r_{job['molecule_id']:04d}_{job['method']}_{job['scale_index']}_{job['variant']:02d}"
    return f"c_{job['molecule_id']:04d}_{job['seed']}_{job['scale_index']}_{job['effective_cap']}"


def score_pair(candidate, reference):
    """Expose both components at the SAME winning combo pose; replay-checked below.

    Reuses the public scorer's canonical frame and four proper-rotation starts.
    Calling shape-only and color-only scorers separately would optimize different
    poses and would not decompose the measured combo score.
    """
    ref = Chem.Mol(reference)
    probe = Chem.Mol(candidate)
    public_rocs._canonicalize_conformer_inplace(ref, -1)
    public_rocs._canonicalize_conformer_inplace(probe, -1)
    best = (0.0, 0.0, 0.0)
    for signs in public_rocs._AXIS_SIGN_FLIPS:
        aligned = public_rocs._flip_conformer(probe, -1, signs)
        try:
            shape, color = rdShapeAlign.AlignMol(
                ref, aligned, useColors=True, opt_param=0.5
            )
        except (RuntimeError, ValueError):
            continue
        combo = (shape + color) / 2
        if combo > best[0]:
            best = (combo, shape, color)
    return best


def outside_scope(case):
    if case["heavy_atoms"] > 45:
        return "training_size_filter: more than 45 heavy atoms"
    if case["rotatable_bonds"] > 15:
        return "training_rotor_filter: more than 15 rotatable bonds"
    return ""


def reference_job(job, case, output):
    directory = Path(output) / "checkpoints"
    stem = checkpoint_name(job)
    path = directory / f"{stem}.sdf"
    smiles = case["reference_isomers"][job["variant"]]
    result = dict(job, isomer_smiles=smiles, count=0, error="", seconds=0.0)
    start = time.perf_counter()
    try:
        if outside_scope(case):
            raise ValueError(outside_scope(case))
        preparer = PrepareMolecule(
            num_conformers=case["attempts"][job["scale_index"]],
            max_reference_conformers=25,
            random_seed=42,
            num_threads=1,
            rmsd_cutoff=1.25,
            energy_window_kcal=8.0,
            optimization=METHODS[job["method"]],
        )
        with tempfile.TemporaryDirectory() as temporary:
            sdf = preparer.generate_reference_conformers(
                Chem.MolFromSmiles(smiles), Path(temporary) / "pool.sdf"
            )
            records = list(Chem.SDMolSupplier(str(sdf), removeHs=False))
        if not records or any(m is None for m in records):
            raise ValueError("No valid reference conformers")
        for record in records:
            record.SetProp("EnumeratedIsomerSMILES", smiles)
            record.SetIntProp("ReferenceVariant", job["variant"])
            check_identity(record, Chem.MolFromSmiles(smiles))
        result.update(
            embedded=records[0].GetIntProp("EmbeddedConformers"),
            converged=records[0].GetIntProp("ConvergedConformers")
            if METHODS[job["method"]].method != "none"
            else None,
            energy_filtered=records[0].GetIntProp("EnergyWindowConformers"),
            before_final_dedup=len(records),
        )
        records = deduplicate_reference(records, threshold=1.25)
        result["preparation_seconds"] = time.perf_counter() - start
        with Chem.SDWriter(str(path)) as writer:
            for record in records:
                control = (
                    public_rocs._score_single_reference(
                        Chem.Mol(record), Chem.Mol(record), "TanimotoCombo", True
                    )
                    / 2
                )
                if not np.isclose(control, 1, atol=1e-5):
                    raise ValueError(f"Exact self-control failed: {control}")
                writer.write(record)
        result.update(
            count=len(records),
            sdf_sha256=digest(path),
            exact_self_controls=len(records),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if path.exists():
            path.unlink()
    result["seconds"] = time.perf_counter() - start
    atomic_json(directory / f"{stem}.json", result)
    return result


def reference_pool(case, method, scale_index, output):
    """Round-robin priority preserves variant coverage before adding more basins."""
    pools, seconds = [], 0.0
    for variant in range(len(case["reference_isomers"])):
        job = dict(
            kind="reference",
            molecule_id=case["molecule_id"],
            method=method,
            scale_index=scale_index,
            variant=variant,
        )
        stem = checkpoint_name(job)
        info = json.loads((Path(output) / "checkpoints" / f"{stem}.json").read_text())
        seconds += info.get("preparation_seconds", info["seconds"])
        pools.append(
            list(
                Chem.SDMolSupplier(
                    str(Path(output) / "checkpoints" / f"{stem}.sdf"), removeHs=False
                )
            )
            if info["count"]
            else []
        )
    # Each variant contributes its first representative before any contributes a second.
    pool = [
        records[i]
        for i in range(max(map(len, pools), default=0))
        for records in pools
        if i < len(records)
    ]
    return pool[:25], seconds


def candidate_job(job, case, output):
    start = time.perf_counter()
    records, generation_error = [], outside_scope(case)
    attempts = case["attempts"][job["scale_index"]]
    try:
        if generation_error:
            raise ValueError(generation_error)
        reference_for_generator, _ = reference_pool(case, 0, 1, output)
        if not reference_for_generator:
            raise ValueError("No target reference could be prepared")
        scorer = create_rocs_scorer(
            reference_for_generator[:1],
            max_conformers=attempts,
            max_isomers=job["effective_cap"],
            timeout=0,
            conformer_threads=1,
            n_jobs=1,
            random_seed=job["seed"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            sdf = scorer.conformer_generator.genConformers([case["smiles"]], temporary)
            records = list(Chem.SDMolSupplier(sdf, removeHs=False))
        if not records or any(m is None for m in records):
            raise ValueError("No valid candidate conformers")
        for record in records:
            check_identity(
                record, Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
            )
    except Exception as exc:
        generation_error = f"{type(exc).__name__}: {exc}"
        records = []
    generation_seconds = time.perf_counter() - start
    actual_isomers = len({m.GetProp("EnumeratedIsomerSMILES") for m in records})
    rows = []
    for method, settings in enumerate(METHODS):
        start = time.perf_counter()
        method_error = generation_error
        try:
            optimized, statuses = (
                (records, [0] * len(records))
                if settings.method == "none"
                else optimize_conformers(records, settings)
            )
        except Exception as exc:
            optimized, statuses = [], [-1] * len(records)
            method_error = f"{type(exc).__name__}: {exc}"
        optimization_seconds = (
            0.0 if settings.method == "none" else time.perf_counter() - start
        )
        for reference_scale in range(len(SCALES)):
            references, preparation_seconds = reference_pool(
                case, method, reference_scale, output
            )
            scores = np.zeros((len(optimized), len(references)))
            shapes, colors, timings = (np.zeros_like(scores) for _ in range(3))
            for i, candidate in enumerate(optimized):
                for j, reference in enumerate(references):
                    start = time.perf_counter()
                    scores[i, j], shapes[i, j], colors[i, j] = score_pair(
                        candidate, reference
                    )
                    timings[i, j] = time.perf_counter() - start
            for reference_cap in REFERENCE_CAPS:
                count = min(reference_cap or len(references), len(references))
                available = bool(references) and (
                    reference_cap == 0 or reference_cap <= len(references)
                )
                score, shape, color, winner, reference_winner, rmsd = (
                    0.0,
                    0.0,
                    0.0,
                    "",
                    "",
                    np.nan,
                )
                if optimized and count:
                    i, j = np.unravel_index(
                        scores[:, :count].argmax(), scores[:, :count].shape
                    )
                    score, shape, color = scores[i, j], shapes[i, j], colors[i, j]
                    winner = optimized[i].GetProp("EnumeratedIsomerSMILES")
                    reference_winner = references[j].GetProp("EnumeratedIsomerSMILES")
                    if winner == reference_winner:
                        rmsd = rdMolAlign.GetBestRMS(
                            Chem.RemoveHs(Chem.Mol(optimized[i])),
                            Chem.RemoveHs(Chem.Mol(references[j])),
                        )
                for requested_cap in job["requested_caps"]:
                    rows.append(
                        dict(
                            molecule_id=case["molecule_id"],
                            category=case["category"],
                            method=method,
                            seed=job["seed"],
                            candidate_scale=case["scales"][job["scale_index"]],
                            candidate_attempts=attempts,
                            isomer_cap=requested_cap,
                            reference_scale=case["scales"][reference_scale],
                            reference_attempts=case["attempts"][reference_scale],
                            reference_cap=reference_cap,
                            actual_references=count if available else 0,
                            available=available,
                            score=score if available else np.nan,
                            shape_tanimoto=shape if available else np.nan,
                            color_tanimoto=color if available else np.nan,
                            seconds=generation_seconds
                            + optimization_seconds
                            + timings[:, :count].sum()
                            if available
                            else np.nan,
                            reference_seconds=preparation_seconds,
                            generation_seconds=generation_seconds,
                            optimization_seconds=optimization_seconds,
                            overlay_seconds=timings[:, :count].sum(),
                            candidate_conformers=len(records),
                            usable_conformers=len(optimized),
                            failed_conformers=sum(s != 0 for s in statuses),
                            actual_isomers=actual_isomers,
                            enumerated_reference_isomers=len(case["reference_isomers"]),
                            actual_reference_isomers=len(
                                {
                                    m.GetProp("EnumeratedIsomerSMILES")
                                    for m in references[:count]
                                }
                            )
                            if available
                            else 0,
                            failure=not optimized or not references,
                            error=method_error,
                            winning_candidate_isomer=winner,
                            winning_reference_isomer=reference_winner,
                            same_identity_winner=bool(winner)
                            and winner == reference_winner,
                            winner_symmetry_rmsd=rmsd,
                        )
                    )
    target = Path(output) / "checkpoints" / f"{checkpoint_name(job)}.csv.gz"
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, compression="gzip")
    temporary.replace(target)
    return dict(job, rows=len(rows), sha256=digest(target))


def execute(job, case, output):
    return (
        reference_job(job, case, output)
        if job["kind"] == "reference"
        else candidate_job(job, case, output)
    )


def candidate_jobs(case):
    aliases = {}
    for cap in ISOMER_CAPS:
        aliases.setdefault(min(cap, case["isomer_upper_bound"]), []).append(cap)
    return [
        dict(
            kind="candidate",
            molecule_id=case["molecule_id"],
            seed=seed,
            scale_index=scale,
            effective_cap=cap,
            requested_caps=requested,
        )
        for seed, scale, (cap, requested) in itertools.product(
            SEEDS, range(3), aliases.items()
        )
    ]


def make_cases(panel, scales=SCALES):
    cases = []
    for row in panel.to_dict("records"):
        mol = prepared_molecule(row["smiles"])
        plan = plan_sampling(mol)
        cases.append(
            row
            | asdict(plan)
            | dict(
                reference_isomers=list(plan.reference_isomers),
                attempts=[plan.attempts(f) for f in scales],
                scales=list(scales),
                category=plan.category,
                heavy_atoms=mol.GetNumHeavyAtoms(),
                prepared_smiles=Chem.MolToSmiles(mol),
            )
        )
    return cases


def run(args):
    output = args.output.resolve()
    if args.render_only:
        manifest = json.loads((output / "manifest.json").read_text())
        before = digest(output / "data.csv.gz")
        render_self_recovery(pd.read_csv(output / "data.csv.gz"), output, manifest)
        if before != digest(output / "data.csv.gz"):
            raise ValueError("Rendering changed measurements")
        manifest["rendering"] = dict(
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            measurement_sha256=before,
            source_sha256={p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS},
        )
        manifest["sha256"] = {
            p.name: digest(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        }
        atomic_json(output / "manifest.json", manifest)
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    hashes = {p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS}
    manifest_path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != hashes:
            raise ValueError("Resume requires the exact measured source")
        cases = json.loads((output / "panel.json").read_text())
    else:
        panel = (
            pd.read_csv(args.panel)
            if args.panel
            else pd.DataFrame([dict(molecule_id=0, smiles=DEFAULT_MOLECULE)])
        )
        if args.ids:
            panel = panel[panel.molecule_id.isin(args.ids)]
        cases = make_cases(panel, args.scales)
        atomic_json(output / "panel.json", cases)
        manifest = dict(
            state="running",
            host=socket.gethostname(),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256=hashes,
            rdkit=rdBase.rdkitVersion,
            panel_size=len(cases),
            panel_input=str(args.panel),
            panel_sha256=digest(args.panel) if args.panel else None,
            command=[
                sys.executable,
                "-m",
                "_david.code.conformer.self_recovery",
                *sys.argv[1:],
            ],
            scales=args.scales,
            isomer_caps=ISOMER_CAPS,
            reference_caps=REFERENCE_CAPS,
            methods=[asdict(m) | {"name": m.name} for m in METHODS],
            candidate_seeds=SEEDS,
            reference_seed=42,
            clustering_rmsd=1.25,
            energy_window=8.0,
            reference_limit=25,
            isomer_limit=25,
            workers=args.workers,
            definition="Each parent vs its own independently generated reference pool; best shape+color / 2",
            selection="Reference variants are round-robin balanced; clustering only within a specified stereoisomer",
            timing="Decomposed instrumented wall seconds; reference cost separate; all attempted work charged, not just the winner",
            warning="AQME-style budgets are a heuristic under test. 0.9 is a recovery hypothesis, not biological accuracy.",
            progress={},
            completed_molecules=[],
        )
    by_id = {c["molecule_id"]: c for c in cases}
    references, pending_references = [], Counter()
    ready = deque()
    for case in cases:
        for method, scale, variant in itertools.product(
            range(len(METHODS)), range(3), range(len(case["reference_isomers"]))
        ):
            job = dict(
                kind="reference",
                molecule_id=case["molecule_id"],
                method=method,
                scale_index=scale,
                variant=variant,
            )
            if not (directory / f"{checkpoint_name(job)}.json").exists():
                references.append(job)
                pending_references[case["molecule_id"]] += 1
        if not pending_references[case["molecule_id"]]:
            ready.extend(candidate_jobs(case))
    np.random.default_rng(42).shuffle(references)
    pending = deque(references)
    counts = Counter(
        reference=sum(1 for _ in directory.glob("r_*.json")),
        candidate=sum(1 for _ in directory.glob("c_*.csv.gz")),
    )
    manifest["progress"] = dict(counts)
    atomic_json(manifest_path, manifest)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {}
        while pending or ready or futures:
            while (ready or pending) and len(futures) < args.workers * 2:
                job = ready.popleft() if ready else pending.popleft()
                if (
                    job["kind"] == "candidate"
                    and (directory / f"{checkpoint_name(job)}.csv.gz").exists()
                ):
                    continue
                futures[
                    executor.submit(
                        execute, job, by_id[job["molecule_id"]], str(output)
                    )
                ] = job
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                job = futures.pop(future)
                future.result()
                counts[job["kind"]] += 1
                if job["kind"] == "reference":
                    pending_references[job["molecule_id"]] -= 1
                    if pending_references[job["molecule_id"]] == 0:
                        ready.extend(candidate_jobs(by_id[job["molecule_id"]]))
                manifest["progress"] = dict(counts)
                manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(manifest_path, manifest)
                print(
                    json.dumps(dict(job=checkpoint_name(job), progress=dict(counts))),
                    flush=True,
                )
    paths = sorted(directory.glob("c_*.csv.gz"))
    frame = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    keys = [
        "molecule_id",
        "method",
        "seed",
        "candidate_scale",
        "isomer_cap",
        "reference_scale",
        "reference_cap",
    ]
    if frame.duplicated(keys).any() or len(frame) != len(cases) * len(
        METHODS
    ) * 3 * 3 * len(SEEDS) * len(ISOMER_CAPS) * len(REFERENCE_CAPS):
        raise ValueError("Incomplete or duplicate self-recovery grid")
    frame.to_csv(output / "data.csv.gz", index=False, compression="gzip")
    manifest.update(
        state="rendering",
        measurement_rows=len(frame),
        completed_molecules=sorted(by_id),
    )
    atomic_json(manifest_path, manifest)
    render_self_recovery(frame, output, manifest)
    if hashes != {p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS}:
        raise ValueError("Source changed during measurement")
    manifest["state"] = "complete"
    manifest["sha256"] = {
        p.name: digest(p)
        for p in output.iterdir()
        if p.is_file() and p != manifest_path
    }
    atomic_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs=3, default=SCALES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
