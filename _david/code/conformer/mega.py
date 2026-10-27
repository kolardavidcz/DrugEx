"""Matched preparation benchmark on a fixed RNN evaluation panel; no RL training.

Each parent molecule contributes once to each summary after averaging its seeds.
``all`` is explicitly bounded at 30 isomers/reference representatives.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Lipinski
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

from drugex.training.scorers import conformer_generators as core_preparation
from drugex.training.scorers.rocs_rdkit import _score_single_reference
from _david.code.chemistry import Optimization, check_identity, optimize_conformers
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.environment import PrepareMolecule, create_rocs_scorer
from _david.code.visualization import render_mega_study


METHODS = (
    Optimization("none"),
    Optimization("MMFF94"),
    Optimization("MMFF94s"),
    Optimization("MMFF94s", 6.5),
    Optimization("MMFF94s", 78.4),
    Optimization("MMFF94s", 1.0, 2),
    Optimization("UFF"),
)
CONFORMERS = (1, 5, 10, 15, 20, 30, 50, 100)
ISOMERS = (1, 3, 5, 10, 30)
REFERENCES = (1, 2, 4, 8, 0)
SEEDS = (0xC0FFEE, 0xC0FFEF, 0xC0FFF0)
_REFS = {}


def prepared_molecule(smiles):
    """Audit the same tautomer/protonation operations used by the core generator."""
    mol = Chem.MolFromSmiles(smiles)
    try:
        mol = core_preparation._RDKIT_TAUTOMER_ENUMERATOR.Canonicalize(mol)
    except Exception:
        pass  # Same fallback as the selected core generator; no new chemistry rule.
    if core_preparation._rdkit_protonate_smiles is not None:
        parsed = Chem.MolFromSmiles(
            core_preparation._rdkit_protonate_smiles(Chem.MolToSmiles(mol))
        )
        if parsed is not None:
            mol = parsed
    return mol


def build_panel(paths, size, seed):
    molecules, audit = {}, []
    with rdBase.BlockLogs():
        for source, path in enumerate(paths):
            frame = pd.read_csv(path)
            valid = 0
            for smiles in frame.SMILES:
                mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
                if mol is None:
                    continue
                valid += 1
                key = Chem.MolToSmiles(mol)
                molecules.setdefault(key, set()).add(source)
            audit.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": len(frame),
                    "valid_rows": valid,
                }
            )
        keys = sorted(molecules)
        selected = np.random.default_rng(seed).choice(
            len(keys), min(size, len(keys)), replace=False
        )
        rows = []
        for index, selected_index in enumerate(selected):
            smiles = keys[int(selected_index)]
            mol = Chem.MolFromSmiles(smiles)
            prepared = prepared_molecule(smiles)
            rows.append(
                {
                    "molecule_id": index,
                    "smiles": smiles,
                    "source_membership": ";".join(map(str, sorted(molecules[smiles]))),
                    "prepared_smiles": Chem.MolToSmiles(prepared),
                    "isomer_upper_bound": int(GetStereoisomerCount(prepared)),
                    "specified_centers_input": len(
                        Chem.FindMolChiralCenters(mol, includeUnassigned=False)
                    ),
                    "specified_centers_prepared": len(
                        Chem.FindMolChiralCenters(prepared, includeUnassigned=False)
                    ),
                    "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
                    "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                }
            )
    return pd.DataFrame(rows), audit


def initialize_worker(reference_file):
    global _REFS
    _REFS = {i: [] for i in range(len(METHODS))}
    for mol in Chem.SDMolSupplier(reference_file, removeHs=False):
        _REFS[mol.GetIntProp("MethodIndex")].append(mol)


def score_geometry(mol, method_index, cache):
    """Cache work, but charge its measured cost to every configuration using it."""
    geometry = hashlib.sha256(
        Chem.MolToSmiles(mol, canonical=False).encode()
        + mol.GetConformer().GetPositions().tobytes()
    ).hexdigest()
    key = (method_index, geometry)
    if key not in cache:
        settings = METHODS[method_index]
        start = time.perf_counter()
        pool, statuses = (
            ([Chem.Mol(mol)], [0])
            if settings.method == "none"
            else optimize_conformers([mol], settings)
        )
        optimization_seconds = (
            0.0 if settings.method == "none" else time.perf_counter() - start
        )
        values, durations = [], []
        if pool:
            for reference in _REFS[method_index]:
                start = time.perf_counter()
                values.append(
                    _score_single_reference(
                        Chem.Mol(pool[0]), Chem.Mol(reference), "TanimotoCombo", True
                    )
                    / 2
                )
                durations.append(time.perf_counter() - start)
        else:
            values = [0.0] * len(_REFS[method_index])
            durations = [0.0] * len(values)
        cache[key] = (values, durations, optimization_seconds, statuses[0] == 0)
    return cache[key]


def measure_parent(case, parts_directory):
    """One process owns one parent's caches; molecules never cross worker queues."""
    rows = []
    expected = Chem.MolFromSmiles(case["prepared_smiles"])
    cap_aliases = {cap: min(cap, max(1, case["isomer_upper_bound"])) for cap in ISOMERS}
    for seed in SEEDS:
        geometry_cache = {}
        work = list(itertools.product(CONFORMERS, sorted(set(cap_aliases.values()))))
        np.random.default_rng(seed + case["molecule_id"]).shuffle(work)
        for conformers, effective_cap in work:
            start = time.perf_counter()
            error, records = "", []
            embedding_failures = 0
            try:
                scorer = create_rocs_scorer(
                    _REFS[1],
                    max_conformers=conformers,
                    max_isomers=effective_cap,
                    timeout=0,
                    conformer_threads=1,
                    n_jobs=1,
                    random_seed=seed,
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = scorer.conformer_generator.genConformers(
                        [case["smiles"]], directory
                    )
                    embedding_failures = (
                        scorer.conformer_generator.last_embedding_failures
                    )
                    records = list(Chem.SDMolSupplier(path, removeHs=False))
                if not records or any(m is None for m in records):
                    raise ValueError("No valid embedded conformers")
                for mol in records:
                    check_identity(mol, expected)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                records = []
            generation_seconds = time.perf_counter() - start
            isomer_count = len({Chem.MolToSmiles(Chem.RemoveHs(m)) for m in records})
            method_order = list(range(len(METHODS)))
            np.random.default_rng(
                seed + case["molecule_id"] + conformers + effective_cap
            ).shuffle(method_order)
            for method_index in method_order:
                values = []
                method_error = error
                try:
                    values = [
                        score_geometry(m, method_index, geometry_cache) for m in records
                    ]
                except Exception as exc:
                    method_error = f"{type(exc).__name__}: {exc}"
                successes = sum(value[3] for value in values)
                optimization_seconds = sum(value[2] for value in values)
                pool_size = len(_REFS[method_index])
                for requested in REFERENCES:
                    available = requested == 0 or requested <= pool_size
                    actual = min(requested or pool_size, pool_size)
                    score = max((max(v[0][:actual]) for v in values), default=0.0)
                    overlays = sum(sum(v[1][:actual]) for v in values)
                    for cap, alias in cap_aliases.items():
                        if alias != effective_cap:
                            continue
                        rows.append(
                            {
                                "molecule_id": case["molecule_id"],
                                "method": method_index,
                                "seed": seed,
                                "conformers": conformers,
                                "isomer_cap": cap,
                                "reference_cap": requested,
                                "actual_references": actual if available else 0,
                                "available": available,
                                "score": score if available else np.nan,
                                "seconds": generation_seconds
                                + optimization_seconds
                                + overlays
                                if available
                                else np.nan,
                                "generation_seconds": generation_seconds,
                                "optimization_seconds": optimization_seconds,
                                "overlay_seconds": overlays,
                                "candidate_conformers": len(records),
                                "embedding_failures": embedding_failures,
                                "scored_conformers": successes,
                                "actual_isomers": isomer_count,
                                "failed_conformers": len(records) - successes,
                                "failure": successes == 0,
                                "error": method_error,
                                "isomers_truncated": case["isomer_upper_bound"] > cap,
                            }
                        )
    target = Path(parts_directory) / f"{case['molecule_id']:05d}.csv.gz"
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, compression="gzip")
    temporary.replace(target)
    return case["molecule_id"], len(rows)


def summarize(frame, panel_size):
    keys = ["method", "isomer_cap", "reference_cap", "conformers"]
    parent = (
        frame.groupby(keys + ["molecule_id"], observed=True)
        .agg(
            score=("score", "mean"),
            seconds=("seconds", "mean"),
            failure=("failure", "mean"),
            available=("available", "first"),
        )
        .reset_index()
    )
    available = parent[parent.available]
    result = (
        available.groupby(keys)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q10_score=("score", lambda values: values.quantile(0.10)),
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            q95_seconds=("seconds", lambda values: values.quantile(0.95)),
            failure_fraction=("failure", "mean"),
            molecules=("molecule_id", "nunique"),
        )
        .reset_index()
    )
    k = max(1, math.ceil(0.05 * panel_size))
    anchors = {}
    for method, group in available[
        (available.conformers == max(CONFORMERS))
        & (available.isomer_cap == max(ISOMERS))
        & (available.reference_cap == 0)
    ].groupby("method"):
        ranked = group[group.score > 0].sort_values(
            ["score", "molecule_id"], ascending=[False, True]
        )
        anchors[method] = set(ranked.head(k).molecule_id) if len(ranked) >= k else set()
    overlap = {}
    for key, group in available.groupby(keys):
        ranked = group[group.score > 0].sort_values(
            ["score", "molecule_id"], ascending=[False, True]
        )
        anchor = anchors.get(key[0], set())
        overlap[key] = (
            len(set(ranked.head(k).molecule_id) & anchor) / k if anchor else np.nan
        )
    result["top5_overlap"] = [
        overlap[tuple(row)] for row in result[keys].itertuples(index=False, name=None)
    ]
    return result


def run(args):
    output = args.output.resolve()
    if args.render_only:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["state"] != "complete":
            raise ValueError("Finish measurements before rendering the final images")
        data_paths = [
            output / name
            for name in (
                "data.csv.gz",
                "summary.csv",
                "panel.csv",
                "references.sdf",
                "controls.csv",
            )
        ]
        before = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths
        }
        render_mega_study(pd.read_csv(output / "summary.csv"), output, manifest)
        assert before == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths
        }
        manifest["rendering"] = {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "command": [
                sys.executable,
                "-m",
                "_david.code.conformer.mega",
                *sys.argv[1:],
            ],
            "data_sha256": before,
            "source_sha256": {
                p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
                for p in (
                    "_david/code/conformer/mega.py",
                    "_david/code/visualization.py",
                )
            },
        }
        manifest["sha256"].update(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.glob("mega_*.png")
            }
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    parts = output / "parts"
    parts.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    source_paths = [
        "_david/code/conformer/mega.py",
        "_david/code/chemistry.py",
        "_david/code/environment.py",
        "_david/code/visualization.py",
        "drugex/training/scorers/conformer_generators.py",
        "drugex/training/scorers/rocs_rdkit.py",
    ]
    hashes = {
        p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
        for p in source_paths
    }
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_sha256"] == hashes, (
            "Resume requires the same scientific source"
        )
        panel = pd.read_csv(output / "panel.csv")
    else:
        panel, audit = build_panel(args.inputs, args.molecules, args.panel_seed)
        panel.to_csv(output / "panel.csv", index=False)
        manifest = {
            "state": "preparing",
            "host": socket.gethostname(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "code_root": str(PROJECT_ROOT),
            "source_sha256": hashes,
            "rdkit": rdBase.rdkitVersion,
            "python": sys.version,
            "command": [
                sys.executable,
                "-m",
                "_david.code.conformer.mega",
                *sys.argv[1:],
            ],
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "DRUGEX_SCORE_NJOBS",
                )
            },
            "cpu_threads_per_worker": 1,
            "embedding_timeout": 0,
            "panel_size": len(panel),
            "panel_seed": args.panel_seed,
            "input_audit": audit,
            "methods": [asdict(m) | {"name": m.name} for m in METHODS],
            "conformers": CONFORMERS,
            "isomer_caps": ISOMERS,
            "reference_caps": REFERENCES,
            "all_limit": 30,
            "stereo_enumeration_seed": 0xC0FFEE,
            "seeds": SEEDS,
            "workers": args.workers,
            "reference_seed": 42,
            "reference_attempts": args.reference_attempts,
            "rmsd_cutoff": 1.25,
            "energy_window_kcal": 8.0,
            "timing": "Decomposed wall seconds, charged per configuration despite internal reuse; worker-local generation, optimization and overlays; excludes pool startup and query preparation",
            "score": "RDKit Tanimoto shape + color, raw / 2; failures contribute zero",
            "completed_molecules": [],
            "references": {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        controls = []
        with Chem.SDWriter(str(output / "references.sdf")) as writer:
            with tempfile.TemporaryDirectory() as directory:
                for index, settings in enumerate(METHODS):
                    start = time.perf_counter()
                    preparer = PrepareMolecule(
                        num_conformers=args.reference_attempts,
                        max_reference_conformers=30,
                        random_seed=42,
                        num_threads=1,
                        optimization=settings,
                    )
                    path = preparer.generate_reference_conformers(
                        args.smiles, Path(directory) / "reference.sdf"
                    )
                    records = list(Chem.SDMolSupplier(str(path), removeHs=False))
                    manifest["references"][str(index)] = {
                        "count": len(records),
                        "seconds": time.perf_counter() - start,
                        "embedded": records[0].GetIntProp("EmbeddedConformers"),
                        "within_energy_window": records[0].GetIntProp(
                            "EnergyWindowConformers"
                        ),
                        "clusters": records[0].GetIntProp("ReferenceClusters"),
                        "energy_window_applied": settings.method != "none",
                    }
                    expected = Chem.MolFromSmiles(preparer.standardize(args.smiles))
                    for j, mol in enumerate(records):
                        check_identity(mol, expected)
                        mol.SetIntProp("MethodIndex", index)
                        mol.SetIntProp("ReferenceIndex", j)
                        writer.write(mol)
                        score = (
                            _score_single_reference(
                                Chem.Mol(mol), Chem.Mol(mol), "TanimotoCombo", True
                            )
                            / 2
                        )
                        assert np.isclose(score, 1, atol=1e-5)
                        controls.append(
                            {"method": index, "reference": j, "exact_self_score": score}
                        )
        pd.DataFrame(controls).to_csv(output / "controls.csv", index=False)
        manifest["reference_smiles"] = args.smiles
    initialize_worker(str(output / "references.sdf"))
    pending = [
        case
        for case in panel.to_dict("records")
        if not (parts / f"{case['molecule_id']:05d}.csv.gz").exists()
    ]
    pending.sort(
        key=lambda case: (min(case["isomer_upper_bound"], 30), case["rotatable_bonds"]),
        reverse=True,
    )
    manifest["state"] = "measuring"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(str(output / "references.sdf"),),
    ) as executor:
        futures = [
            executor.submit(measure_parent, case, str(parts)) for case in pending
        ]
        for future in as_completed(futures):
            molecule_id, count = future.result()
            manifest["completed_molecules"].append(molecule_id)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"Completed {len(manifest['completed_molecules'])}/{len(panel)} parents; id={molecule_id}, rows={count}",
                flush=True,
            )
    with gzip.open(output / "data.csv.gz", "wt") as destination:
        for index, path in enumerate(sorted(parts.glob("*.csv.gz"))):
            with gzip.open(path, "rt") as source:
                if index:
                    next(source)
                shutil.copyfileobj(source, destination)
    frame = pd.read_csv(output / "data.csv.gz")
    summary = summarize(frame, len(panel))
    summary.to_csv(output / "summary.csv", index=False)
    render_mega_study(summary, output, manifest)
    assert hashes == {
        p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
        for p in source_paths
    }
    manifest.update(
        state="complete",
        measurement_rows=len(frame),
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file() and p != manifest_path
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[])
    parser.add_argument("--molecules", type=int, default=512)
    parser.add_argument("--panel-seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reference-attempts", type=int, default=1000)
    parser.add_argument("--smiles", default=DEFAULT_MOLECULE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not (args.resume or args.render_only) and not args.inputs:
        parser.error("Provide saved evaluation CSVs with --inputs")
    run(args)


if __name__ == "__main__":
    main()
