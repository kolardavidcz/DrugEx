"""Four-method ROCS self-recovery and latency calibration, validation, and query budgeting."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from _david.code.chemistry import (
    check_identity,
    enable_double_precision_mol_pickling,
    standardize_smiles,
    verify_disjoint_conformer_seeds,
)
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.conformer.reference.reference_bank import (
    CANDIDATE_ISOMER_CAP,
    CANDIDATE_SEED,
    ENERGY_WINDOW_KCAL,
    LABELS,
    METHODS,
    REFERENCE_ATTEMPTS,
    REFERENCE_ISOMER_CAP,
    REFERENCE_SEED,
    STEREO_SEED,
    prepare_banks,
)
from _david.code.conformer.core.utils import (
    atomic_json,
    collect_jobs,
    digest,
    source_hashes as compute_source_hashes,
)
from _david.code.conformer.analysis.visualization import (
    render_reward_heatmaps,
    render_reward_pareto,
    render_rl_timing,
)
from _david.code.environment import create_rocs_scorer
from _david.code.rocs import score_pair

CANDIDATES: tuple[int, ...] = (5, 10, 15, 20, 35, 50)
CAPS: tuple[int, ...] = (1, 2, 4, 8, 12, 0)
KEYS: tuple[str, ...] = (
    "molecule_id",
    "method",
    "reference_cap",
    "candidate_attempts",
)
CHECK_SEEDS: tuple[int, int] = (42, 1_000_003)

SOURCES: tuple[str, ...] = (
    "_david/code/chemistry.py",
    "_david/code/config.py",
    "_david/code/environment.py",
    "_david/code/reference.py",
    "_david/code/rocs.py",
    "_david/code/conformer/utils.py",
    "_david/code/conformer/reference_bank.py",
    "_david/code/conformer/reward_calibration.py",
    "drugex/training/scorers/rocs_rdkit.py",
    "drugex/training/scorers/conformer_generators.py",
)


def source_hashes() -> dict[str, str]:
    """Return SHA-256 digests of the calibration source modules."""
    return compute_source_hashes(SOURCES)


def default_settings(
    candidates: Sequence[int] = CANDIDATES,
    caps: Sequence[int] = CAPS,
) -> list[dict[str, Any]]:
    """Build the Cartesian grid of (method, reference_cap, candidate_attempts) cells."""
    return [
        dict(method=method, reference_cap=int(cap), candidate_attempts=int(count))
        for method, cap, count in itertools.product(METHODS, caps, candidates)
    ]


def measure_case(
    case: Mapping[str, Any],
    output: str | Path,
    settings: Sequence[Mapping[str, Any]],
) -> int:
    """Measure one molecule across all requested settings in an isolated worker process."""
    enable_double_precision_mol_pickling()
    raw = Path(output)
    molecule_id = int(case["molecule_id"])
    directory = raw / "checkpoints" / str(molecule_id)
    if (directory / "complete.json").exists():
        return molecule_id
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    methods = tuple(dict.fromkeys(str(item["method"]) for item in settings))
    positive_caps = [
        int(item["reference_cap"]) for item in settings if item["reference_cap"]
    ]
    pools, detail = prepare_banks(
        str(case["smiles"]),
        directory,
        methods,
        diverse_prefix=max(positive_caps, default=12),
    )

    # Untimed warm-up: pay one-time per-process initialisation (rdShapeAlign C++
    # lazy setup, MMFF parameter tables, SDF I/O machinery) before the timed
    # grid, so it cannot contaminate one randomly shuffled cell per molecule.
    warmup_seconds = 0.0
    warmup_method = next((m for m in methods if pools.get(m)), None)
    if warmup_method is not None and not detail["color_excluded"]:
        warmup = create_rocs_scorer(
            pools[warmup_method][:1],
            max_conformers=5,
            max_isomers=CANDIDATE_ISOMER_CAP,
            random_seed=CANDIDATE_SEED,
            minimize_candidates=warmup_method != "etkdg",
            optimization=METHODS[warmup_method],
            timeout=10,
            conformer_threads=1,
            n_jobs=1,
        )
        tick = time.perf_counter()
        warmup.getScores([case["smiles"]])
        warmup_seconds = time.perf_counter() - tick

    ordered = list(settings)
    np.random.default_rng(17_000 + molecule_id).shuffle(ordered)
    rows: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[str, int], str] = {}

    for item in ordered:
        method = str(item["method"])
        cap = int(item["reference_cap"])
        attempts = int(item["candidate_attempts"])
        bank = pools.get(method, [])
        refs = bank[:cap] if cap else bank
        row: dict[str, Any] = dict(
            molecule_id=molecule_id,
            name=case["name"],
            split=case["split"],
            method=method,
            reference_cap=cap,
            candidate_attempts=attempts,
            actual_references=len(refs),
            actual_candidates=0,
            candidate_attempted=False,
            candidate_file="",
            color_excluded=bool(detail["color_excluded"]),
            reference_failure=not bool(refs) and not bool(detail["color_excluded"]),
            candidate_failure=False,
            score=np.nan if detail["color_excluded"] else 0.0,
            seconds=0.0,
            generation_minimization_seconds=0.0,
            alignment_io_seconds=0.0,
            embedding_failures=0,
            minimization_failures=0,
            error=detail["preparation_error"],
        )
        if detail["color_excluded"] or not refs:
            rows.append(row)
            continue

        scorer = create_rocs_scorer(
            refs,
            max_conformers=attempts,
            max_isomers=CANDIDATE_ISOMER_CAP,
            random_seed=CANDIDATE_SEED,
            minimize_candidates=method != "etkdg",
            optimization=METHODS[method],
            timeout=10,
            conformer_threads=1,
            n_jobs=1,
        )
        generator = scorer.conformer_generator
        generator.stereo_seed = STEREO_SEED
        generator.embedding_rmsd = None
        generator.protonate = False
        original_generate = generator.genConformers
        candidate_name = f"candidate_{method}_c{attempts}.sdf"
        stage: dict[str, Any] = {}

        def capture_generate(
            smiles_list: Sequence[str],
            temp_dir: str,
            generate: Any = original_generate,
            stage: dict[str, Any] = stage,
            candidate_name: str = candidate_name,
        ) -> str:
            tick = time.perf_counter()
            path = generate(smiles_list, temp_dir)
            stage["seconds"] = time.perf_counter() - tick
            saved = directory / candidate_name
            if not saved.exists():
                shutil.copy2(path, saved)
            records = (
                list(Chem.SDMolSupplier(str(saved), removeHs=False))
                if saved.stat().st_size
                else []
            )
            stage["count"] = len(records)
            return path

        generator.genConformers = capture_generate
        tick = time.perf_counter()
        raw_score = float(scorer.getScores([case["smiles"]])[0, 0])
        elapsed = time.perf_counter() - tick
        score = raw_score / 2.0
        gen_seconds = float(stage.get("seconds", 0.0))
        candidate_count = int(stage.get("count", 0))
        candidate_cache[(method, attempts)] = candidate_name
        statuses = list(generator.last_minimization_statuses)
        row.update(
            actual_candidates=candidate_count,
            candidate_attempted=True,
            candidate_file=candidate_name,
            candidate_failure=candidate_count == 0,
            score=0.0 if candidate_count == 0 else score,
            seconds=elapsed,
            generation_minimization_seconds=gen_seconds,
            alignment_io_seconds=elapsed - gen_seconds,
            embedding_failures=int(generator.last_embedding_failures),
            minimization_failures=sum(code != 0 for code in statuses),
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(list(KEYS)).reset_index(drop=True)
    frame.to_csv(directory / "scores.csv.gz", index=False)
    detail.update(
        warmup_method=warmup_method,
        warmup_seconds=warmup_seconds,
        case={
            key: (int(val) if isinstance(val, np.integer) else val)
            for key, val in case.items()
        },
        candidate_files= {f"{m}_c{c}": name for (m, c), name in candidate_cache.items()},
        source_sha256=source_hashes(),
        files_sha256={
            path.name: digest(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "complete.json"
        },
    )
    atomic_json(directory / "complete.json", detail)
    return molecule_id


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each configuration with one equal weight per input molecule."""
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate molecule/configuration rows in calibration table")
    rows: list[dict[str, Any]] = []
    for (method, cap, attempts), group in frame.groupby(
        list(KEYS[1:]), sort=True
    ):
        eligible = group[~group.color_excluded]
        scores = eligible.score
        times = group.seconds
        gen_times = group.generation_minimization_seconds
        align_times = group.alignment_io_seconds
        rows.append(
            dict(
                method=method,
                label=LABELS[str(method)],
                reference_cap=int(cap),
                candidate_attempts=int(attempts),
                inputs=len(group),
                color_excluded=int(group.color_excluded.sum()),
                scored_inputs=len(eligible),
                reference_failures=int(eligible.reference_failure.sum()),
                candidate_failures=int(eligible.candidate_failure.sum()),
                mean=float(scores.mean()),
                median=float(scores.median()),
                p05=float(scores.quantile(0.05)),
                p10=float(scores.quantile(0.10)),
                fraction_ge_090=float(scores.ge(0.90).mean()),
                fraction_ge_095=float(scores.ge(0.95).mean()),
                mean_seconds=float(times.mean()),
                median_seconds=float(times.median()),
                p95_seconds=float(times.quantile(0.95)),
                mean_generation_minimization_seconds=float(gen_times.mean()),
                mean_alignment_io_seconds=float(align_times.mean()),
                mean_references=float(eligible.actual_references.mean()),
                median_references=float(eligible.actual_references.median()),
                max_references=int(eligible.actual_references.max()),
                mean_candidates=float(eligible.actual_candidates.mean()),
                median_candidates=float(eligible.actual_candidates.median()),
            )
        )
    return pd.DataFrame(rows)


def shortlist(summary: pd.DataFrame) -> dict[str, Any]:
    """Select the fastest configuration per method meeting mean >= 0.95 on development."""
    selected: list[dict[str, Any]] = []
    for method in METHODS:
        group = summary[summary.method.eq(method)]
        if group.empty:
            continue
        passing = group[group["mean"].ge(0.95)]
        chosen = (
            passing.sort_values(
                ["mean_seconds", "p95_seconds", "reference_cap", "candidate_attempts"]
            ).iloc[0]
            if len(passing)
            else group.sort_values(
                ["mean", "mean_seconds"], ascending=[False, True]
            ).iloc[0]
        )
        selected.append(
            dict(
                method=method,
                reference_cap=int(chosen.reference_cap),
                candidate_attempts=int(chosen.candidate_attempts),
                development_mean=float(chosen["mean"]),
                development_p05=float(chosen.p05),
                development_fraction_ge_095=float(chosen.fraction_ge_095),
                development_mean_seconds=float(chosen.mean_seconds),
                development_target_met=bool(chosen["mean"] >= 0.95),
            )
        )
    return dict(
        rule="For each method, choose the lowest-mean-time development cell with mean ROCS >= 0.95; otherwise choose the highest-mean-ROCS cell for that method.",
        settings=selected,
    )


def select_replay_cells(eligible: pd.DataFrame, replays: int) -> pd.DataFrame:
    """Select representative saved-pose replay cells across methods plus any >1.0 overshoots."""
    pool = eligible.assign(
        preferred=eligible.reference_cap.eq(1) & eligible.candidate_attempts.eq(10),
        pair_work=eligible.actual_references * eligible.actual_candidates,
    ).sort_values(
        ["preferred", "pair_work", *KEYS],
        ascending=[False, True, *([True] * len(KEYS))],
    )
    pool = pool.drop_duplicates(["method", "molecule_id"])
    pool["replay_rank"] = pool.groupby("method").cumcount()
    selected = pool.sort_values(["replay_rank", "method"]).head(replays)
    return pd.concat([selected, eligible[eligible.score.gt(1.00001)]]).drop_duplicates(
        list(KEYS)
    )[eligible.columns]


class _SavedGenerator:
    """Replay helper that returns a pre-saved candidate SDF path."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def genConformers(self, smiles: Sequence[str], destination: str) -> str:
        return str(self.path)


def validate(output: Path, phase: str = "development", replays: int = 12) -> dict[str, Any]:
    """Validate saved geometry identities, complete grid invariants, and score replays."""
    raw = output / "raw_data" / phase
    path = raw / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Measurement writer must finish before validation")
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Current measurement imports differ from recorded source")

    verify_disjoint_conformer_seeds(
        int(manifest["reference_geometry_seed"]),
        50,
        int(manifest["candidate_geometry_seed"]),
        75,
    )
    frame = pd.read_csv(raw / "data.csv.gz")
    smiles_by_id = (
        pd.read_csv(raw / "panel.csv").set_index("molecule_id").smiles.to_dict()
    )
    if len(frame) != manifest["expected_rows"] or frame.duplicated(list(KEYS)).any():
        raise ValueError("Missing or duplicated grid cells")

    expected_settings = {
        tuple(setting[key] for key in KEYS[1:]) for setting in manifest["settings"]
    }
    for mid, group in frame.groupby("molecule_id"):
        actual = {
            tuple(row[key] for key in KEYS[1:]) for row in group.to_dict("records")
        }
        if actual != expected_settings:
            raise ValueError(f"Incomplete grid for {mid}")

    geometry_checks = 0
    hash_checks = 0
    for mid in manifest["input_ids"]:
        directory = raw / "checkpoints" / str(mid)
        saved = json.loads((directory / "complete.json").read_text())
        if saved["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"Mixed source at input {mid}")
        for name, expected in saved["files_sha256"].items():
            file = directory / name
            if digest(file) != expected:
                raise ValueError(f"Changed measurement: {file}")
            hash_checks += 1
            if file.suffix != ".sdf" or file.stat().st_size == 0:
                continue
            for record in Chem.SDMolSupplier(str(file), removeHs=False):
                if record is None:
                    raise ValueError(f"Invalid geometry file: {file}")
                if not record.HasProp("EnumeratedIsomerSMILES"):
                    raise ValueError(f"Missing stereo identity: {file}")
                check_identity(
                    record, Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
                )
                geometry_checks += 1

    eligible = frame[
        ~frame.color_excluded
        & ~frame.reference_failure
        & ~frame.candidate_failure
        & frame.candidate_attempted
    ]
    if not frame.loc[~frame.color_excluded, "score"].between(0, 1.0001).all():
        raise ValueError("Invalid native normalized score")

    selected = select_replay_cells(eligible, replays)
    replay_rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        directory = raw / "checkpoints" / str(row["molecule_id"])
        refs = list(
            Chem.SDMolSupplier(
                str(directory / f"reference_{row['method']}.sdf"), removeHs=False
            )
        )
        if row["reference_cap"]:
            refs = refs[: int(row["reference_cap"])]
        candidates = list(
            Chem.SDMolSupplier(str(directory / row["candidate_file"]), removeHs=False)
        )
        best = max(
            score_pair(candidate, ref) for candidate in candidates for ref in refs
        )
        scorer = create_rocs_scorer(refs, n_jobs=1, conformer_threads=1)
        scorer.conformer_generator = _SavedGenerator(directory / row["candidate_file"])
        public = float(scorer.getScores([smiles_by_id[row["molecule_id"]]])[0, 0]) / 2.0
        if max(abs(best[0] - row["score"]), abs(public - row["score"])) > 1e-8:
            raise ValueError(f"Score replay mismatch for {row['molecule_id']}")
        if abs(best[0] - (best[1] + best[2]) / 2.0) > 1e-12:
            raise ValueError("Component normalization mismatch")
        replay_rows.append(
            {key: row[key] for key in KEYS}
            | dict(
                measured=row["score"],
                public_replay=public,
                normalized_combo=best[0],
                shape_tanimoto=best[1],
                color_tanimoto=best[2],
            )
        )

    pd.DataFrame(replay_rows).to_csv(raw / "score_replays.csv", index=False)
    manifest["validation"] = dict(
        grid_rows=len(frame),
        inputs=int(frame.molecule_id.nunique()),
        file_hash_checks=hash_checks,
        geometry_identity_checks=geometry_checks,
        public_and_component_replays=len(replay_rows),
        score_tolerance=1e-8,
        native_upper_bound_epsilon=1e-4,
        disjoint_bounded_seed_streams=True,
        max_public_difference=max(
            (abs(row["public_replay"] - row["measured"]) for row in replay_rows),
            default=0.0,
        ),
        min_score=float(eligible.score.min()),
        max_score=float(eligible.score.max()),
        script_sha256=digest(__file__),
    )
    atomic_json(path, manifest)
    return manifest["validation"]


def budget_order(
    summary: pd.DataFrame,
    method: str,
    available: int,
) -> Iterator[dict[str, Any]]:
    """Yield deduplicated (reference_cap, candidate_attempts) budgets in ascending empirical cost."""
    selected = summary[summary.method.eq(method)].sort_values("mean_seconds")
    seen: set[tuple[int, int]] = set()
    for row in selected.to_dict("records"):
        requested = int(row["reference_cap"])
        actual = min(requested, available) if requested else available
        count = int(row["candidate_attempts"])
        if (actual, count) not in seen:
            seen.add((actual, count))
            yield dict(
                reference_cap=actual,
                candidate_attempts=count,
                panel_mean_seconds=float(row["mean_seconds"]),
            )


def calibrate_query(
    study: Path,
    smiles: str = DEFAULT_MOLECULE,
    target: float = 0.95,
) -> dict[str, Any]:
    """Suggest a fixed per-query budget using measured development cost order and self-checks."""
    if not 0 < target <= 1:
        raise ValueError("Target must be in (0, 1]")
    for seed in CHECK_SEEDS:
        verify_disjoint_conformer_seeds(REFERENCE_SEED, 50, seed, 75)
    verify_disjoint_conformer_seeds(CHECK_SEEDS[0], 75, CHECK_SEEDS[1], 75)

    recommended = json.loads((study / "raw_data/recommended_settings.json").read_text())
    method = str(recommended["method"])
    cleaned = standardize_smiles(smiles)
    query_id = hashlib.sha256((method + ":" + cleaned).encode()).hexdigest()[:12]
    raw = study / "raw_data/query_calibration" / query_id
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "suggestion.json").exists():
        previous = json.loads((raw / "suggestion.json").read_text())
        if (
            previous["source_sha256"] == digest(__file__)
            and previous["target"] == target
        ):
            return previous
        raise RuntimeError(
            "Query calibration with different settings already exists; use a new study output"
        )

    pools, preparation = prepare_banks(cleaned, raw, (method,))
    bank = pools[method]
    result: dict[str, Any] = dict(
        status="no_passing_budget",
        query=cleaned,
        method=method,
        target=target,
        actual_reference_pool=len(bank),
        preparation=preparation,
        check_seeds=CHECK_SEEDS,
        reference_seed=REFERENCE_SEED,
        candidate_isomer_cap=CANDIDATE_ISOMER_CAP,
        source_sha256=digest(__file__),
    )
    if not bank or preparation["color_excluded"]:
        result["status"] = "unsupported_reference"
        atomic_json(raw / "suggestion.json", result)
        return result

    summary = pd.read_csv(study / "raw_data/development/summary.csv")
    checks: list[dict[str, Any]] = []
    for budget in budget_order(summary, method, len(bank)):
        refs = bank[: budget["reference_cap"]]
        scores: list[float] = []
        for seed in CHECK_SEEDS:
            scorer = create_rocs_scorer(
                refs,
                max_conformers=budget["candidate_attempts"],
                max_isomers=CANDIDATE_ISOMER_CAP,
                random_seed=seed,
                minimize_candidates=method != "etkdg",
                optimization=METHODS[method],
                timeout=10,
                conformer_threads=1,
                n_jobs=1,
            )
            scorer.conformer_generator.embedding_rmsd = None
            scorer.conformer_generator.protonate = False
            score = float(scorer.getScores([Chem.MolFromSmiles(cleaned)])[0, 0]) / 2.0
            checks.append(dict(**budget, seed=seed, normalized_rocs=score))
            scores.append(score)
        pd.DataFrame(checks).to_csv(raw / "self_checks.csv", index=False)
        if min(scores) >= target:
            result.update(
                status="reference_self_check_passed",
                suggested=budget,
                minimum_self_score=min(scores),
            )
            break
    atomic_json(raw / "suggestion.json", result)
    return result


def render_reward_calibration(output: Path) -> None:
    """Render heatmaps, Pareto frontiers, and RL timing from saved reward calibration tables."""
    raw = output / "raw_data/development"
    summary = pd.read_csv(raw / "summary.csv")
    frame = pd.read_csv(raw / "data.csv.gz")
    status = json.loads((raw / "manifest.json").read_text()).get(
        "scientific_status", ""
    )
    warning = (
        "SUPERSEDED: reference/candidate RNG streams overlap — not valid for recommendations\n"
        if status == "superseded_seed_overlap"
        else ""
    )
    # Derive the measured grid from the saved summary so older runs (whose
    # candidate/cap constants differ from the current module defaults) remain
    # renderable; module constants are only defaults for new measurements.
    grid_caps = tuple(
        sorted((int(cap) for cap in summary.reference_cap.unique()), key=lambda c: (c == 0, c))
    )
    grid_candidates = tuple(
        sorted(int(count) for count in summary.candidate_attempts.unique())
    )
    grid_methods = {
        method: METHODS[method] for method in METHODS if method in set(summary.method)
    }
    render_reward_heatmaps(
        summary,
        output,
        methods=grid_methods,
        labels=LABELS,
        caps=grid_caps,
        candidates=grid_candidates,
        warning=warning,
    )
    render_reward_pareto(
        frame,
        summary,
        output,
        labels=LABELS,
        keys=KEYS,
        warning=warning,
    )
    render_rl_timing(output, labels=LABELS)


def run(args: argparse.Namespace) -> None:
    """Execute the reward calibration measurement phase."""
    enable_double_precision_mol_pickling()
    verify_disjoint_conformer_seeds(REFERENCE_SEED, 50, CANDIDATE_SEED, 75)
    raw = args.output / "raw_data" / args.phase
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "checkpoints").mkdir(exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)

    panel = pd.read_csv(args.panel)
    if args.split != "all":
        panel = panel[panel.split.eq(args.split)].copy()
    if args.limit:
        panel = panel.head(args.limit).copy()
    panel = panel.reset_index(drop=True)
    if panel.empty:
        raise ValueError("Selected panel split is empty")

    settings = (
        json.loads(args.settings.read_text())["settings"]
        if args.settings
        else default_settings(
            [int(x) for x in args.candidates.split(",")],
            [int(x) for x in args.caps.split(",")],
        )
    )
    settings = [
        {key: item[key] for key in ("method", "reference_cap", "candidate_attempts")}
        for item in settings
    ]
    hashes = source_hashes()
    manifest_path = raw / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != hashes:
            raise ValueError("Source changed; do not mix revisions in one phase")
    else:
        panel.to_csv(raw / "panel.csv", index=False)
        manifest = dict(
            state="measuring",
            phase=args.phase,
            split=args.split,
            started_utc=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            python=platform.python_version(),
            rdkit=rdBase.rdkitVersion,
            dimorphite_dl=version("dimorphite_dl"),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], text=True),
            command=[
                sys.executable,
                "-m",
                "_david.code.conformer.reward_calibration",
                *sys.argv[1:],
            ],
            panel_source=str(args.panel),
            panel_sha256=digest(raw / "panel.csv"),
            source_sha256=hashes,
            total_molecules=len(panel),
            input_ids=panel.molecule_id.astype(int).tolist(),
            expected_rows=len(panel) * len(settings),
            settings=settings,
            reference_attempts_per_isomer=REFERENCE_ATTEMPTS,
            reference_isomer_cap=REFERENCE_ISOMER_CAP,
            candidate_isomer_cap=CANDIDATE_ISOMER_CAP,
            stereo_enumeration_seed=STEREO_SEED,
            reference_geometry_seed=REFERENCE_SEED,
            candidate_geometry_seed=CANDIDATE_SEED,
            energy_window_kcal=ENERGY_WINDOW_KCAL,
            completed_molecules=[],
            errors={},
        )
        atomic_json(manifest_path, manifest)

    done = {
        int(path.parent.name) for path in (raw / "checkpoints").glob("*/complete.json")
    }
    manifest["completed_molecules"] = sorted(done)
    pending = [
        row for row in panel.to_dict("records") if int(row["molecule_id"]) not in done
    ]
    pending.sort(
        key=lambda r: (
            min(int(r["stereo_upper_bound"]), 8),
            int(r["rotatable_bonds"]),
            int(r["heavy_atoms"]),
        ),
        reverse=True,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            errors = collect_jobs(
                pool,
                pending,
                raw,
                manifest,
                worker_fn=lambda case, out: measure_case(case, out, settings),
            )
            if errors:
                raise RuntimeError(f"{len(errors)} inputs failed; see logs/")

    if hashes != source_hashes():
        raise ValueError("Source changed during reward calibration")

    frames = [
        pd.read_csv(raw / "checkpoints" / str(int(mid)) / "scores.csv.gz")
        for mid in panel.molecule_id
    ]
    frame = pd.concat(frames, ignore_index=True).sort_values(list(KEYS)).reset_index(drop=True)
    if len(frame) != len(panel) * len(settings) or frame.duplicated(list(KEYS)).any():
        raise ValueError("Incomplete or duplicate calibration grid")
    frame.to_csv(raw / "data.csv.gz", index=False)
    summary = summarize(frame)
    summary.to_csv(raw / "summary.csv", index=False)
    if args.phase == "development":
        atomic_json(raw / "shortlist.json", shortlist(summary))
    manifest.update(
        state="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        rows=len(frame),
        output_sha256={
            name: digest(raw / name) for name in ("data.csv.gz", "summary.csv")
        },
    )
    atomic_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["pilot", "development", "confirmation"], default="development"
    )
    parser.add_argument(
        "--split", choices=["development", "confirmation", "all"], default="development"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--candidates", default=",".join(map(str, CANDIDATES))
    )
    parser.add_argument("--caps", default=",".join(map(str, CAPS)))
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate(args.output, args.phase), indent=2))
    elif args.analyze:
        render_reward_calibration(args.output)
    else:
        if args.panel is None:
            parser.error("--panel is required when running measurements")
        run(args)


if __name__ == "__main__":
    main()
