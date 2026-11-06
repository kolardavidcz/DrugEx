"""Fixed-budget self-recovery on saved inputs; no RNN training.

Each input has independent reference geometries. The two RMSD selections reuse
one saved pool. Groups depend only on the two optimized reference profiles.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from _david.code.chemistry import (
    CandidateConformerGenerator,
    Optimization,
    check_identity,
    optimize_conformers,
)
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.conformer.benchmarks.mega import prepared_molecule
from _david.code.conformer.core.sampling import deduplicate_reference
from _david.code.conformer.core.utils import atomic_json, digest, save_sdf
from _david.code.training.environment import PrepareMolecule, create_rocs_scorer
from _david.code.rocs import score_pair, has_color
from _david.code.analysis.visualization import render_mini_recovery

METHODS = (Optimization("none"), Optimization("MMFF94"), Optimization("MMFF94s", 78.4))
ATTEMPTS = (1, 5, 10, 15, 20, 25, 30, 50)
ISOMER_CAPS = (1, 2, 4, 8)
REFERENCE_CAPS = (1, 2, 4, 8, 0)
CUTOFFS = (1.25, 0.5)
SOURCE_PATHS = (
    "_david/code/conformer/self_recovery.py",
    "_david/code/conformer/sampling.py",
    "_david/code/conformer/mega.py",
    "_david/code/chemistry.py",
    "_david/code/environment.py",
    "_david/code/reference.py",
    "_david/code/rocs.py",
    "_david/code/config.py",
    "_david/code/visualization.py",
    "_david/code/conformer/visualization.py",
    "drugex/training/scorers/conformer_generators.py",
    "drugex/training/scorers/rocs_rdkit.py",
)


def reference_group(counts):
    """Use optimized profiles only; fixed membership across both RMSD panels."""
    largest = max(counts, default=0)
    if largest <= 0:
        return ""
    for limit, label in [(4, "1-4"), (8, "5-8"), (12, "9-12")]:
        if largest <= limit:
            return label
    return ""


def stem(case):
    return f"r_{case['molecule_id']:04d}"


def reference_job(case, output):
    """Save pre-compression pools and index selections; never share candidate coordinates."""
    directory = Path(output) / "checkpoints"
    result = dict(
        case, profiles=[], group="", grouping_max=0, error="", exclusion_reason=""
    )
    try:
        mol = prepared_molecule(case["smiles"])
        generator = CandidateConformerGenerator(
            max_isomers=4,
            max_heavy_atoms=45,
            max_rotatable_bonds=15,
            show_progress=False,
            stereo_seed=42,
        )
        if generator._filter_mol(case["smiles"], Chem.MolFromSmiles(case["smiles"])):
            raise ValueError("Outside the training size/rotor domain")
        isomers = list(generator._get_isomers(mol))
        result.update(
            prepared_smiles=Chem.MolToSmiles(mol), enumerated_isomers=len(isomers)
        )
        embedded, embedding_info = [], []
        for iso in isomers:
            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as temporary:
                path = PrepareMolecule(
                    num_conformers=50,
                    max_reference_conformers=50,
                    random_seed=43,
                    rmsd_cutoff=None,
                    num_threads=1,
                    optimization=Optimization("none"),
                ).generate_reference_conformers(iso, Path(temporary) / "raw.sdf")
                records = list(Chem.SDMolSupplier(str(path), removeHs=False))
            for record in records:
                record.SetProp(
                    "EnumeratedIsomerSMILES", iso.GetProp("EnumeratedIsomerSMILES")
                )
                check_identity(record, iso)
            embedded.append(records)
            embedding_info.append(
                dict(
                    count=len(records),
                    seconds=time.perf_counter() - started,
                    native_color_present=has_color(records[0]),
                )
            )
        result["embedding"] = embedding_info
        if not all(item["native_color_present"] for item in embedding_info):
            result["exclusion_reason"] = (
                "no_native_color_features"
                if not any(item["native_color_present"] for item in embedding_info)
                else "mixed_native_color_availability"
            )
            atomic_json(directory / f"{stem(case)}.json", result)
            return result
        for method, settings in enumerate(METHODS):
            started = time.perf_counter()
            pools, status_counts = [], Counter()
            for records in embedded:
                if settings.method == "none":
                    usable = records
                else:
                    usable, statuses = optimize_conformers(records, settings)
                    status_counts.update(statuses)
                    usable.sort(key=lambda m: m.GetDoubleProp("Energy_kcal_mol"))
                    if usable:
                        lowest = usable[0].GetDoubleProp("Energy_kcal_mol")
                        usable = [
                            m
                            for m in usable
                            if m.GetDoubleProp("Energy_kcal_mol") - lowest <= 8
                        ]
                pools.append(usable)
            pool = [m for records in pools for m in records]
            identity_to_index = {id(m): i for i, m in enumerate(pool)}
            optimization_seconds = time.perf_counter() - started
            selections, pruning_seconds = {}, {}
            for cutoff in CUTOFFS:
                tick = time.perf_counter()
                kept = [
                    deduplicate_reference(records, threshold=cutoff)
                    for records in pools
                ]
                balanced = [
                    records[i]
                    for i in range(max(map(len, kept), default=0))
                    for records in kept
                    if i < len(records)
                ]
                selections[str(cutoff)] = [identity_to_index[id(m)] for m in balanced]
                pruning_seconds[str(cutoff)] = time.perf_counter() - tick
            sdf = directory / f"{stem(case)}_{method}.sdf"
            with Chem.SDWriter(str(sdf)) as writer:
                for m in pool:
                    control, shape, color = score_pair(m, m)
                    if not np.isclose(control, 1, atol=1e-5):
                        raise ValueError(
                            f"Exact self-control failed: {control}, {shape}, {color}"
                        )
                    writer.write(m)
            result["profiles"].append(
                dict(
                    method=method,
                    count=len(pool),
                    selections=selections,
                    status_counts=dict(status_counts),
                    pruning_seconds=pruning_seconds,
                    seconds=time.perf_counter() - started,
                    optimization_seconds=optimization_seconds,
                    sdf_sha256=digest(sdf),
                    exact_self_controls=len(pool),
                )
            )
        counts = [
            len(p["selections"][str(c)])
            for p in result["profiles"][1:]
            for c in CUTOFFS
        ]
        result["grouping_max"] = max(counts, default=0)
        result["group"] = reference_group(counts)
        if not result["group"]:
            result["exclusion_reason"] = (
                "optimized_reference_count_gt12"
                if result["grouping_max"] > 12
                else "no_usable_optimized_references"
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["exclusion_reason"] = "reference_preparation_error"
    atomic_json(directory / f"{stem(case)}.json", result)
    return result


def candidate_jobs(case):
    """Alias only caps that enumerate the entire known set; keep requested axes."""
    aliases = {}
    for cap in ISOMER_CAPS:
        aliases.setdefault(min(cap, case["stereo_upper_bound"]), []).append(cap)
    return [
        dict(molecule_id=case["molecule_id"], attempts=c, cap=cap, aliases=requested)
        for c, (cap, requested) in itertools.product(ATTEMPTS, aliases.items())
    ]


def candidate_stem(job):
    return f"c_{job['molecule_id']:04d}_{job['attempts']}_{job['cap']}"


def candidate_job(job, output):
    directory = Path(output) / "checkpoints"
    case = json.loads((directory / f"r_{job['molecule_id']:04d}.json").read_text())
    tick = time.perf_counter()
    generator = CandidateConformerGenerator(
        max_conformers=job["attempts"],
        max_isomers=job["cap"],
        random_seed=42,
        stereo_seed=42,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=1,
        timeout=0,
        show_progress=False,
    )
    records, error = [], ""
    try:
        with tempfile.TemporaryDirectory() as temporary:
            path = generator.genConformers([case["smiles"]], temporary)
            records = list(Chem.SDMolSupplier(str(path), removeHs=False))
        records = [m for m in records if m is not None]
        for m in records:
            check_identity(m, Chem.MolFromSmiles(m.GetProp("EnumeratedIsomerSMILES")))
    except (OSError, ValueError, RuntimeError) as exc:
        records, error = [], f"{type(exc).__name__}: {exc}"
    generation_seconds = time.perf_counter() - tick
    rows = []
    for method, settings in enumerate(METHODS):
        tick = time.perf_counter()
        usable, statuses = (
            (records, []) if method == 0 else optimize_conformers(records, settings)
        )
        optimization_seconds = time.perf_counter() - tick if method else 0.0
        profile = case["profiles"][method]
        references = (
            list(
                Chem.SDMolSupplier(
                    str(directory / f"r_{job['molecule_id']:04d}_{method}.sdf"),
                    removeHs=False,
                )
            )
            if profile["count"]
            else []
        )
        needed = sorted(
            set(i for indices in profile["selections"].values() for i in indices)
        )
        columns = {index: j for j, index in enumerate(needed)}
        scores = np.zeros((len(usable), len(needed), 3))
        timings = np.zeros(scores.shape[:2])
        for i, candidate in enumerate(usable):
            for j, index in enumerate(needed):
                tick = time.perf_counter()
                scores[i, j] = score_pair(candidate, references[index])
                timings[i, j] = time.perf_counter() - tick
        for cutoff in CUTOFFS:
            selection = profile["selections"][str(cutoff)]
            for cap in REFERENCE_CAPS:
                available = bool(selection) and (cap == 0 or cap <= len(selection))
                indices = selection[: cap or len(selection)] if available else []
                selected_columns = [columns[i] for i in indices]
                score, shape, color = (0.0, 0.0, 0.0) if available else (np.nan,) * 3
                candidate_winner = reference_winner = ""
                if usable and indices:
                    values = scores[:, selected_columns, 0]
                    i, j = np.unravel_index(values.argmax(), values.shape)
                    score, shape, color = scores[i, selected_columns[j]]
                    c, r = usable[i], references[indices[j]]
                    candidate_winner = c.GetProp("EnumeratedIsomerSMILES")
                    reference_winner = r.GetProp("EnumeratedIsomerSMILES")
                for requested in job["aliases"]:
                    rows.append(
                        dict(
                            molecule_id=case["molecule_id"],
                            group=case["group"],
                            method=method,
                            cutoff=cutoff,
                            candidate_attempts=job["attempts"],
                            isomer_cap=requested,
                            reference_cap=cap,
                            available=available,
                            score=score,
                            shape_tanimoto=shape,
                            color_tanimoto=color,
                            candidate_conformers=len(records),
                            usable_conformers=len(usable),
                            actual_candidate_isomers=len(
                                {m.GetProp("EnumeratedIsomerSMILES") for m in usable}
                            ),
                            actual_references=len(indices),
                            reference_pool_count=len(selection),
                            failed_conformers=sum(s != 0 for s in statuses),
                            candidate_failure=available and not usable,
                            reference_failure=not selection,
                            generation_seconds=generation_seconds,
                            optimization_seconds=optimization_seconds,
                            overlay_seconds=float(timings[:, selected_columns].sum()),
                            seconds=generation_seconds
                            + optimization_seconds
                            + float(timings[:, selected_columns].sum())
                            if available
                            else np.nan,
                            reference_seconds=sum(
                                r["seconds"] for r in case["embedding"]
                            )
                            + profile["optimization_seconds"]
                            + profile["pruning_seconds"][str(cutoff)],
                            error=error,
                            winning_candidate_isomer=candidate_winner,
                            winning_reference_isomer=reference_winner,
                        )
                    )
    target = directory / f"{candidate_stem(job)}.csv.gz"
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, compression="gzip")
    temporary.replace(target)
    return job["molecule_id"]


def run_phase(executor, jobs, function, output, manifest, key):
    futures = {executor.submit(function, job, str(output)): job for job in jobs}
    remaining = Counter(j["molecule_id"] for j in jobs)
    for future in as_completed(futures):
        result = future.result()
        if key == "candidate":
            remaining[result] -= 1
            if remaining[result] == 0:
                manifest["completed_molecules"].append(result)
        manifest["progress"][key] += 1
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "manifest.json", manifest)
        print(json.dumps(manifest["progress"]), flush=True)


def run(args):
    output = args.output.resolve()
    if args.render_only:
        manifest = json.loads((output / "manifest.json").read_text())
        before = digest(output / "data.csv.gz")
        render_mini_recovery(pd.read_csv(output / "data.csv.gz"), output, manifest)
        assert before == digest(output / "data.csv.gz")
        manifest["rendering"] = dict(
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256={p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS},
        )
        atomic_json(output / "manifest.json", manifest)
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    hashes = {p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS}
    path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(path.read_text())
        if hashes != manifest["source_sha256"]:
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
        cases = []
        from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

        for row in panel.to_dict("records"):
            mol = prepared_molecule(row["smiles"])
            cases.append(
                dict(
                    molecule_id=int(row["molecule_id"]),
                    smiles=row["smiles"],
                    stereo_upper_bound=int(GetStereoisomerCount(mol)),
                )
            )
        if not cases:
            raise ValueError("Empty input panel")
        atomic_json(output / "panel.json", cases)
        manifest = dict(
            state="reference_preparation",
            host=socket.gethostname(),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256=hashes,
            rdkit=rdBase.rdkitVersion,
            panel_size=len(cases),
            panel_path=str(args.panel),
            panel_sha256=digest(args.panel) if args.panel else None,
            methods=[asdict(m) | dict(name=m.name) for m in METHODS],
            cutoffs=CUTOFFS,
            candidate_attempts=ATTEMPTS,
            isomer_caps=ISOMER_CAPS,
            reference_caps=REFERENCE_CAPS,
            reference_attempts=50,
            reference_isomer_cap=4,
            stereo_seed=42,
            candidate_geometry_seed=42,
            reference_geometry_seed=43,
            color_policy="Unmodified native RDKit color and combo; inputs with absent native reference color features excluded before scoring",
            grouping="Max retained structures across optimized methods and BOTH cutoffs; unminimized profile excluded from grouping; fixed 1-4/5-8/9-12 groups; >12 excluded",
            score="One max of (shape+effective_color)/2 over permitted pairs per input; no seed averaging",
            timing="Sum of instrumented generation, optimization, selected overlay wall seconds from shared work; NOT independent end-to-end cell runtime; reference preparation separate",
            retry="Reference embedding may retry 50 requests with random coordinates; at most 50 per call, not a hard total across retries",
            workers=args.workers,
            command=[
                sys.executable,
                "-m",
                "_david.code.conformer.self_recovery",
                *sys.argv[1:],
            ],
            progress=dict(reference=0, candidate=0),
        )
    atomic_json(path, manifest)
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            jobs = [c for c in cases if not (directory / f"{stem(c)}.json").exists()]
            manifest["progress"]["reference"] = len(cases) - len(jobs)
            run_phase(executor, jobs, reference_job, output, manifest, "reference")
            prepared = [
                json.loads((directory / f"{stem(c)}.json").read_text()) for c in cases
            ]
            cohort = pd.DataFrame(
                [
                    dict(
                        molecule_id=c["molecule_id"],
                        group=c["group"],
                        grouping_max=c["grouping_max"],
                        error=c["error"],
                        exclusion_reason=c["exclusion_reason"],
                    )
                    for c in prepared
                ]
            )
            cohort.to_csv(output / "cohort.csv", index=False)
            selected = [c for c in prepared if c["group"] and not c["error"]]
            manifest.update(
                state="candidate_scoring",
                selected_inputs=len(selected),
                group_counts=dict(Counter(c["group"] for c in selected)),
                excluded_inputs=len(cases) - len(selected),
                exclusion_counts=dict(
                    Counter(
                        c["exclusion_reason"]
                        for c in prepared
                        if not c["group"] or c["error"]
                    )
                ),
            )
            all_jobs = [j for c in selected for j in candidate_jobs(c)]
            jobs = [
                j
                for j in all_jobs
                if not (directory / f"{candidate_stem(j)}.csv.gz").exists()
            ]
            manifest["progress"]["candidate"] = len(all_jobs) - len(jobs)
            manifest["expected_candidate_jobs"] = len(all_jobs)
            unfinished = {j["molecule_id"] for j in jobs}
            manifest["completed_molecules"] = [
                c["molecule_id"] for c in selected if c["molecule_id"] not in unfinished
            ]
            atomic_json(path, manifest)
            run_phase(executor, jobs, candidate_job, output, manifest, "candidate")
        if not selected:
            raise ValueError(
                "No inputs in selected reference-size groups; inspect cohort.csv"
            )
        frame = pd.concat(
            [pd.read_csv(p) for p in sorted(directory.glob("c_*.csv.gz"))],
            ignore_index=True,
        )
        keys = [
            "molecule_id",
            "method",
            "cutoff",
            "candidate_attempts",
            "isomer_cap",
            "reference_cap",
        ]
        expected = (
            len(selected)
            * len(METHODS)
            * len(CUTOFFS)
            * len(ATTEMPTS)
            * len(ISOMER_CAPS)
            * len(REFERENCE_CAPS)
        )
        if len(frame) != expected or frame.duplicated(keys).any():
            raise ValueError("Incomplete or duplicated measurement grid")
        if not np.allclose(
            frame.score,
            (frame.shape_tanimoto + frame.color_tanimoto) / 2,
            equal_nan=True,
        ):
            raise ValueError("Component reduction mismatch")
        frame.to_csv(output / "data.csv.gz", index=False, compression="gzip")
        manifest.update(state="rendering", measurement_rows=len(frame))
        atomic_json(path, manifest)
        render_mini_recovery(frame, output, manifest)
        if hashes != {p: digest(PROJECT_ROOT / p) for p in SOURCE_PATHS}:
            raise ValueError("Source changed during measurements")
        manifest["state"] = "complete"
        manifest["sha256"] = {
            p.name: digest(p) for p in output.iterdir() if p.is_file() and p != path
        }
        atomic_json(path, manifest)
    except Exception as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
