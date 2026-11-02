"""1:1 training-interface cost and early-accuracy calibration across methods.

Measures time x ROCS accuracy with the EXACT production training interface,
not isolated single-molecule calls. The reward-calibration Pareto
(``reward_calibration.py``) times one molecule per serial ``getScores`` call on
DrugCentral drugs; the live-RL reference showed that underestimates real RL
cost by ~3.2x. This module closes that gap:

- **Batch parity**: the RL loop (``SequenceExplorer.fit`` ->
  ``env.getRewards`` -> ``getScores``) scores the full epoch's unique sample
  set (~``n_samples`` after deduplication) in ONE call — not the 64-molecule
  RNN gradient batch. Each arm scores the SAME frozen-prior molecule batch
  through ``create_environment`` with the sprint's ``PipelineConfig``, so
  arms are directly comparable (paired inputs).

- **Stage accounting**: per call and per epoch we separate (1) candidate
  generation + minimization, (2) ROCS alignment + scorer I/O, (3) other
  environment work (validity/SA/desirability), (4) RNN/optimizer wall. On the
  dichtator smoke these were ~51%/47%/2% of an epoch — candidate preparation
  is roughly HALF the cost, not 95%; both halves are tracked.

- **Fairness guards** (because a minimized arm measuring FASTER than an
  unminimized one is a red flag, not a result): each arm runs in a fresh
  subprocess with no shared pools; the input batch is identical across arms;
  per-batch rows record valid/zero-score counts (dropped candidates would
  reduce alignment work and fake a speedup); cold (first call: pool spawn +
  library warm-up) is separated from warm repeats. Pools are re-created per
  ``getScores``/``genConformers`` call in BOTH training and this benchmark,
  so warm repeats measure the same steady state production pays each epoch.

- **Phases**: (1) batch timing: ``--repeats`` scoring calls of the full unique
  batch; (2) ``--epochs``-epoch RL pilot per arm through the real
  ``explorer.fit`` with the sprint H1 configuration (2-objective [ROCS, SA],
  Pareto crowding, eps=0.05, tanimoto_combo raw scale), recording per-epoch
  cost breakdown and validation ROCS trajectory. Five epochs is an EARLY
  TRAJECTORY and engineering-cost comparison, not a final learning
  comparison; extend ``--epochs`` (and add seeds) before claiming accuracy
  differences.

Usage (repository root):

.. code-block:: bash

    # Full run (MetaCentrum sprint-parity node: 4 CPUs -> 3 workers):
    python -m _david.code.conformer.rl_batch_calibration \
        --output _david/output/<run_id> --samples 500 --repeats 3 --epochs 5

    # Quick local smoke (2 arms, 48 samples, 2 repeats, 1 epoch):
    python -m _david.code.conformer.rl_batch_calibration --output <dir> --quick

    # Re-render figures/summary from saved arm data only:
    python -m _david.code.conformer.rl_batch_calibration --output <dir> --render-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from drugex.training.monitors import FileMonitor

from _david.code.chemistry import standardize_smiles
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT, PipelineConfig
from _david.code.conformer.reference_bank import METHODS, prepare_banks, save_records
from _david.code.conformer.utils import atomic_json, digest, source_hashes
from _david.code.environment import create_environment
from _david.code.models import SequenceRNNOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager

SOURCES: tuple[str, ...] = (
    "_david/code/chemistry.py",
    "_david/code/config.py",
    "_david/code/environment.py",
    "_david/code/models.py",
    "_david/code/reference.py",
    "_david/code/rocs.py",
    "_david/code/schema.py",
    "_david/code/utils.py",
    "_david/code/conformer/reference_bank.py",
    "_david/code/conformer/rl_batch_calibration.py",
    "drugex/training/explorers/sequence_explorer.py",
    "drugex/training/environment.py",
    "drugex/training/scorers/rocs_rdkit.py",
)

# (method, reference cap K, candidate attempts C). Defaults cover the sprint's
# production Regime-2 setting plus the reward-calibration shortlist cells.
# Ordered cheapest-first so time-boxed runs (weak CPUs) complete the most
# decision-relevant arms before the expensive all-bank cells.
DEFAULT_ARMS: tuple[tuple[str, int, int], ...] = (
    ("etkdg", 1, 30),  # sprint Regime 2 production setting (unminimized)
    ("mmff_high_dielectric", 4, 20),  # fast minimized shortlist
    ("etkdg", 0, 75),  # development shortlist (all-bank ETKDG)
    ("uff", 0, 75),  # development shortlist
    ("mmff_vacuum", 0, 75),  # development shortlist
    ("mmff_high_dielectric", 0, 50),  # mean>=0.95 shortlist
)
QUICK_ARMS: tuple[tuple[str, int, int], ...] = (
    ("etkdg", 1, 30),
    ("mmff_high_dielectric", 4, 20),
)

# Stacked-bar stage styles shared by the epoch-cost panel (order matters).
STAGE_STYLES: tuple[tuple[str, str, str], ...] = (
    ("generation_minimization_seconds", "#31688e", "Candidate generation + minimization"),
    ("align_io_seconds", "#35b779", "ROCS alignment + I/O"),
    ("env_other_seconds", "#d581b0", "Env (validity/SA/desirability)"),
    ("rnn_residual_seconds", "#8c6bb1", "RNN + optimizer + monitor"),
)

TOTAL_KEYS: tuple[str, ...] = (
    "rocs_seconds",
    "generation_minimization_seconds",
    "environment_seconds",
    "environment_rows",
    "scorer_calls",
)


@dataclass(frozen=True)
class Arm:
    """One (method, reference cap, candidate budget) measurement arm."""

    method: str
    reference_cap: int
    candidate_attempts: int

    @property
    def label(self) -> str:
        cap = "all" if self.reference_cap == 0 else str(self.reference_cap)
        return f"{self.method}_k{cap}_c{self.candidate_attempts}"


def arm_label(method: str, cap: int, attempts: int) -> str:
    """Canonical directory-safe label for one arm."""
    return Arm(method, cap, attempts).label


def parse_arms(spec: str) -> list[Arm]:
    """Parse a comma-separated ``method:cap:attempts`` arm specification."""
    arms: list[Arm] = []
    for item in spec.split(","):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid arm {item!r}; expected method:cap:attempts")
        method, cap, attempts = parts
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
        arm = Arm(method, int(cap), int(attempts))
        if arm not in arms:
            arms.append(arm)
    if not arms:
        raise ValueError("At least one arm is required")
    return arms


def _git_revision() -> str:
    """Best-effort source revision (staged MetaCentrum releases lack .git)."""
    environment = os.environ.get("DRUGEX_SOURCE_REVISION")
    if environment:
        return environment
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _sprint_config(arm: Arm, *, samples: int, epochs: int, gpu: int) -> PipelineConfig:
    """PipelineConfig 1:1 with the sprint H1 training configuration."""
    return PipelineConfig(
        epochs=epochs,
        min_epochs=epochs,
        patience=max(6, epochs),
        n_samples=samples,
        training_seed=42,
        random_seed=42,
        cpu_ratio=0.90,
        max_conformers=arm.candidate_attempts,
        max_isomers=4,
        minimize_candidates=arm.method != "etkdg",
        optimization=METHODS[arm.method],
        # Production parity: sprint H1 passes no --timeout, so the scorer
        # uses the PipelineConfig default of 30 s — not the study-scorer 10 s.
        timeout_seconds=30,
        conformer_threads=1,
        gpu=gpu,
        epsilon=0.05,
        use_epsilon_decay=False,
        score_variant="tanimoto_combo",
        reward_scheme="pareto_crowding",
        include_qed=False,
    )


def score_row(df_smiles: pd.DataFrame, epoch: int) -> dict[str, Any]:
    """Summarize one epoch's validation scores frame into a single row."""
    has_valid = "Valid" in df_smiles
    valid = df_smiles[df_smiles.Valid.eq(1)] if has_valid else df_smiles
    rocs = valid[std.ROCS] if std.ROCS in valid else pd.Series(dtype=float)
    return dict(
        epoch=epoch,
        rows=len(df_smiles),
        valid=int(df_smiles.Valid.sum()) if has_valid else len(df_smiles),
        unique=(
            int(df_smiles.SMILES.nunique())
            if "SMILES" in df_smiles
            else len(df_smiles)
        ),
        mean_rocs=float(rocs.mean()) if len(rocs) else float("nan"),
        median_rocs=float(rocs.median()) if len(rocs) else float("nan"),
        p90_rocs=float(rocs.quantile(0.9)) if len(rocs) else float("nan"),
        desired_ratio=(
            float(df_smiles.Desired.mean()) if "Desired" in df_smiles else float("nan")
        ),
    )


def sample_prior_batch(output: Path, *, samples: int, gpu: int) -> Path:
    """Sample a fixed, deduplicated molecule batch from the frozen prior.

    The batch is sampled once per run and shared by every arm so arms see
    identical inputs (paired comparison). Sampled from the pretrained
    Papyrus05.5 SMILES-RNN exactly like epoch 0 of training.
    """
    target = output / "raw_data" / "prior_batch.csv"
    if target.exists():
        cached_rows = len(pd.read_csv(target))
        if cached_rows != samples:
            raise RuntimeError(
                f"Cached prior batch has {cached_rows} rows but --samples is "
                f"{samples}; use a fresh --output directory or delete {target}"
            )
        return target
    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    optimizer = SequenceRNNOptimizer(
        output / "raw_data" / "models", gpu=gpu, cpu_ratio=0.8
    )
    agent, _, _ = optimizer.load_models()
    collected: list[str] = []
    seen: set[str] = set()
    while len(seen) < samples and len(collected) < 20 * samples:
        batch = agent.sample(samples)
        collected.extend(batch)
        seen.update(smiles for smiles in batch if smiles)
    unique = sorted(seen)[:samples]
    if len(unique) < samples:
        raise RuntimeError(f"Prior sampling produced only {len(unique)} unique molecules")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"SMILES": unique})
    frame.to_csv(target, index=False)
    return target


def prepare_shared_references(output: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    """Prepare per-method reference banks once for all arms.

    ``prepare_banks`` embeds and optimizes the reference molecule (R50/I8
    diverse protocol, the same banks the reward calibration shortlist was
    measured against) and writes ``reference_<method>.sdf`` diversity-ordered.
    Preparing once in the parent guarantees byte-identical references across
    arms; each child subprocess then truncates to its own cap.
    """
    references = output / "raw_data" / "references"
    references.mkdir(parents=True, exist_ok=True)
    marker = references / "banks_complete.json"
    if marker.exists():
        cached = json.loads(marker.read_text())
        missing = [m for m in methods if m not in cached.get("methods", [])]
        if missing:
            raise RuntimeError(
                f"Cached reference banks do not cover {missing}; use a fresh "
                "--output directory or remove the marker to re-prepare"
            )
        return dict(cached=True)
    pools, preparation = prepare_banks(
        standardize_smiles(DEFAULT_MOLECULE), references, methods
    )
    for method, bank in pools.items():
        if not bank:
            raise RuntimeError(f"Reference preparation failed for {method}")
        save_records(references / f"reference_{method}.sdf", bank)
    atomic_json(
        marker,
        dict(
            methods=list(methods),
            reference_preparation=preparation,
            sha256={
                file.name: digest(file)
                for file in sorted(references.glob("reference_*.sdf"))
            },
        ),
    )
    return preparation


def _load_reference_bank(references: Path, method: str) -> list[Chem.Mol]:
    """Load a prepared, diversity-ordered reference bank from SDF."""
    path = references / f"reference_{method}.sdf"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared reference bank: {path}")
    bank = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol]
    if not bank:
        raise RuntimeError(f"Reference bank {path} contains no valid records")
    return bank


class ArmMonitor(FileMonitor):
    """FileMonitor recording per-epoch cost and validation quality per arm."""

    def __init__(self, root: Path, raw: Path, totals: dict[str, Any]) -> None:
        super().__init__(root / "logs" / raw.name, save_smiles=True)
        self.raw = raw
        self.models = raw / "models"
        self.models.mkdir(exist_ok=True)
        self.outDF = raw / "performance.tsv"
        self.outSmiles = raw / "validation_smiles.tsv"
        # FileMonitor append tables would duplicate epoch records if this arm
        # is retried after a partial failure; restart them cleanly.
        self.outDF.unlink(missing_ok=True)
        self.outSmiles.unlink(missing_ok=True)
        self.totals = totals
        self.previous = dict(totals)
        self.previous_time = time.perf_counter()
        self.epochs: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []

    def saveModel(self, model, identifier=None) -> None:
        self.currentState = model.getModel()
        suffix = f"_{identifier}" if identifier else ""
        torch.save(self.currentState, self.models / f"agent{suffix}.pkg")

    def savePerformanceInfo(self, performance_dict, df_smiles=None) -> None:
        super().savePerformanceInfo(performance_dict, df_smiles)
        now = time.perf_counter()
        differences = {key: self.totals[key] - self.previous[key] for key in self.totals}
        self.epochs.append(
            dict(
                epoch=len(self.epochs) + 1,
                epoch_seconds=now - self.previous_time,
                **differences,
            )
        )
        pd.DataFrame(self.epochs).to_csv(self.raw / "epoch_timings.csv", index=False)
        if df_smiles is not None and len(df_smiles):
            self.scores.append(score_row(df_smiles, len(self.scores) + 1))
            pd.DataFrame(self.scores).to_csv(self.raw / "epoch_scores.csv", index=False)
        self.previous, self.previous_time = dict(self.totals), now


def run_arm(
    arm: Arm,
    output: Path,
    *,
    batch_file: Path,
    samples: int,
    repeats: int,
    epochs: int,
    gpu: int,
) -> Path:
    """Child-process entry: measure one arm end-to-end (batch + RL pilot)."""
    torch.set_num_threads(1)
    os.environ.setdefault("DRUGEX_MP_CONTEXT", "forkserver")
    Chem.SetDefaultPickleProperties(
        Chem.GetDefaultPickleProperties() | Chem.PropertyPickleOptions.CoordsAsDouble
    )
    raw = output / "raw_data" / arm.label
    state = raw / "complete.json"
    if state.exists():
        print(f"Arm {arm.label} already complete; skipping")
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        _run_arm_inner(
            arm,
            output,
            raw,
            batch_file=batch_file,
            samples=samples,
            repeats=repeats,
            epochs=epochs,
            gpu=gpu,
            started=started,
        )
    except BaseException as exc:
        atomic_json(
            raw / "failed.json",
            dict(
                arm=asdict(arm),
                label=arm.label,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
                failed_utc=datetime.now(timezone.utc).isoformat(),
                git_revision=_git_revision(),
            ),
        )
        raise
    print(f"[{arm.label}] complete in {time.perf_counter() - started:.0f}s")
    return raw


def _run_arm_inner(
    arm: Arm,
    output: Path,
    raw: Path,
    *,
    batch_file: Path,
    samples: int,
    repeats: int,
    epochs: int,
    gpu: int,
    started: float,
) -> None:
    """Actual arm measurement; failure state is recorded by the caller."""
    references = output / "raw_data" / "references"
    bank = _load_reference_bank(references, arm.method)
    chosen = bank[: arm.reference_cap] if arm.reference_cap else bank
    if not chosen:
        raise RuntimeError(f"No reference records selected for {arm.label}")
    ref_path = raw / "reference.sdf"
    save_records(ref_path, chosen)

    config = _sprint_config(arm, samples=samples, epochs=epochs, gpu=gpu)
    # Normalize the inherited DRUGEX_SCORE_NJOBS to sprint parity (0.90)
    # BEFORE environment construction: candidate embedding pools read the
    # environment variable via capped_n_jobs(), while the alignment pool
    # takes an explicit n_jobs — both must agree with production training.
    HardwareManager.init_runtime_environment(ratio=config.cpu_ratio)
    environment = create_environment(ref_path, config=config)
    scorer = environment.scorers[0]
    totals = dict.fromkeys(TOTAL_KEYS, 0.0)
    original_generate = scorer.conformer_generator.genConformers
    original_scorer = scorer.getScores
    original_environment = environment.getScores

    def timed_generate(smiles_list, directory):
        tick = time.perf_counter()
        result = original_generate(smiles_list, directory)
        totals["generation_minimization_seconds"] += time.perf_counter() - tick
        return result

    def timed_scorer(*args, **kwargs):
        tick = time.perf_counter()
        result = original_scorer(*args, **kwargs)
        totals["rocs_seconds"] += time.perf_counter() - tick
        totals["scorer_calls"] += 1
        return result

    def timed_environment(smiles, *args, **kwargs):
        tick = time.perf_counter()
        result = original_environment(smiles, *args, **kwargs)
        totals["environment_seconds"] += time.perf_counter() - tick
        totals["environment_rows"] += len(smiles)
        return result

    scorer.conformer_generator.genConformers = timed_generate
    scorer.getScores = timed_scorer
    environment.getScores = timed_environment

    # ---- Phase 1: batch timing through the exact training call path ----
    batch = pd.read_csv(batch_file)
    smiles = batch.SMILES.tolist()
    timing_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for key in totals:
            totals[key] = 0.0
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        scores = environment.getScores(smiles)
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        valid = int(scores.Valid.sum())
        zero_score_valid = int(
            ((scores.Valid == 1) & (scores[std.ROCS] == 0.0)).sum()
        )
        # Fairness counters: a minimized arm whose optimization drops conformers
        # (or filters/embedding failures) does LESS alignment work per valid
        # molecule and would falsely appear faster. The generator's last_*
        # attributes are reset per genConformers call, so they describe exactly
        # this repeat's batch.
        generator = scorer.conformer_generator
        minimization_failures = sum(
            1 for status in generator.last_minimization_statuses if status != 0
        )
        timing_rows.append(
            dict(
                arm=arm.label,
                method=arm.method,
                reference_cap=arm.reference_cap,
                candidate_attempts=arm.candidate_attempts,
                repeat=repeat,
                cold=repeat == 0,
                inputs=len(smiles),
                references=len(chosen),
                valid=valid,
                zero_score_valid=zero_score_valid,
                embedding_failures=int(generator.last_embedding_failures),
                identity_rejections=len(generator.last_identity_rejections),
                filtered_molecules=len(generator.last_filtered_molecules),
                minimization_attempts=len(generator.last_minimization_statuses),
                minimization_failures=minimization_failures,
                wall_seconds=wall,
                cpu_seconds=cpu,
                generate_seconds=totals["generation_minimization_seconds"],
                scorer_seconds=totals["rocs_seconds"],
                environment_seconds=totals["environment_seconds"],
                mean_rocs=float(scores.loc[scores.Valid == 1, std.ROCS].mean()),
                median_rocs=float(scores.loc[scores.Valid == 1, std.ROCS].median()),
            )
        )
        scores.assign(repeat=repeat).to_csv(
            raw / f"batch_scores_repeat{repeat}.csv", index=False
        )
        pd.DataFrame(timing_rows).to_csv(raw / "batch_timing.csv", index=False)
        print(
            f"[{arm.label}] repeat {repeat}: wall {wall:.1f}s, "
            f"gen {totals['generation_minimization_seconds']:.1f}s, "
            f"valid {valid}/{len(smiles)}, zero-score valid {zero_score_valid}, "
            f"min-fail {minimization_failures}/"
            f"{len(generator.last_minimization_statuses)}, "
            f"embed-fail {generator.last_embedding_failures}"
        )

    # ---- Phase 2: short RL pilot through the real explorer.fit ----
    # The batch-timing phase above leaves pools warm, matching steady-state
    # training; epoch 1 therefore excludes cold-start cost by design.
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    optimizer = SequenceRNNOptimizer(
        output / "raw_data" / "models", gpu=gpu, cpu_ratio=config.cpu_ratio
    )
    agent, prior, _ = optimizer.load_models()
    explorer = optimizer.build_explorer(agent, environment, prior, config)
    # Construct the monitor only now: its previous_time baseline must not
    # include model loading / explorer assembly, which would otherwise be
    # charged to epoch 1's RNN residual.
    monitor = ArmMonitor(output, raw, totals)
    fit_started = time.perf_counter()
    try:
        explorer.fit(
            monitor=monitor,
            epochs=config.epochs,
            patience=max(6, config.epochs),
            min_epochs=config.epochs,
            reload_interval=config.epochs + 1,
            criteria=config.criteria,
        )
    finally:
        monitor.close()
    fit_seconds = time.perf_counter() - fit_started

    package, vocabulary = config.resolve_pretrained_paths()
    atomic_json(
        raw / "complete.json",
        dict(
            arm=asdict(arm),
            label=arm.label,
            state="complete",
            host=os.uname().nodename,
            started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            seconds_total=time.perf_counter() - started,
            fit_seconds=fit_seconds,
            config=asdict(config),
            workers=int(HardwareManager.get_safe_parallel_jobs(ratio=config.cpu_ratio)),
            actual_references=len(chosen),
            reference_bank_protocol="R50/I8 diverse (reward-calibration banks)",
            batch_file_sha256=digest(batch_file),
            reference_sha256=digest(ref_path),
            base_checkpoint=str(package),
            base_checkpoint_sha256=digest(package),
            vocabulary_sha256=digest(vocabulary),
            source_sha256=source_hashes(SOURCES),
            git_revision=_git_revision(),
            fairness_note=(
                "Paired prior inputs; fresh subprocess per arm; cold/warm separated; "
                "valid and zero-score counts recorded per repeat so dropped "
                "candidates cannot fake a speedup"
            ),
        ),
    )


def summarize_timing(timing: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repeat batch timing rows into one summary row per arm.

    Per-molecule stage seconds are derived directly from the stage
    accumulators (not from shares of wall time), so non-ROCS environment
    overhead stays a separate stage and a zero scorer reading yields NaN
    instead of a division error.
    """
    timing = timing.copy()
    timing["per_molecule_seconds"] = timing.wall_seconds / timing.inputs
    timing["scorer_seconds"] = timing.scorer_seconds.replace(0.0, np.nan)
    rows: list[dict[str, Any]] = []
    for label, group in timing.groupby("arm", sort=True):
        cold = group[group.cold]
        warm = group[~group.cold]
        if warm.empty:
            warm = group  # single cold repeat only (quick mode)
        rows.append(
            dict(
                arm=label,
                method=group.method.iloc[0],
                reference_cap=int(group.reference_cap.iloc[0]),
                candidate_attempts=int(group.candidate_attempts.iloc[0]),
                references=int(group.references.iloc[0]),
                cold_per_molecule_seconds=(
                    float(cold.per_molecule_seconds.mean())
                    if len(cold)
                    else float("nan")
                ),
                warm_per_molecule_seconds=float(warm.per_molecule_seconds.mean()),
                gen_per_molecule=float((warm.generate_seconds / warm.inputs).mean()),
                align_io_per_molecule=float(
                    ((warm.scorer_seconds - warm.generate_seconds) / warm.inputs).mean()
                ),
                env_other_per_molecule=float(
                    ((warm.wall_seconds - warm.scorer_seconds) / warm.inputs).mean()
                ),
                valid_rate=float((group.valid / group.inputs).mean()),
                zero_score_valid_rate=float(
                    (group.zero_score_valid / group.inputs).mean()
                ),
                embedding_failures=float(group.embedding_failures.mean()),
                identity_rejections=float(group.identity_rejections.mean()),
                filtered_molecules=float(group.filtered_molecules.mean()),
                minimization_failures=float(group.minimization_failures.mean()),
                mean_rocs=float(group.mean_rocs.mean()),
            )
        )
    return pd.DataFrame(rows)


def epoch_stage_breakdown(epochs: pd.DataFrame) -> pd.DataFrame:
    """Split mean per-epoch seconds into non-overlapping cost stages per arm.

    Stages: generation+minimization, ROCS alignment+I/O (rocs - generation),
    other environment work (env - rocs), and the RNN/optimizer residual
    (epoch - env). Residuals are clipped at zero so a stage can never go
    negative if a nested clock overlaps its parent.
    """
    rows: list[dict[str, Any]] = []
    for label, group in epochs.groupby("arm", sort=True):
        mean = group.mean(numeric_only=True)
        generation = float(mean.generation_minimization_seconds)
        rocs = float(mean.rocs_seconds)
        environment = float(mean.environment_seconds)
        epoch = float(mean.epoch_seconds)
        rows.append(
            dict(
                arm=label,
                epoch_seconds=epoch,
                generation_minimization_seconds=generation,
                align_io_seconds=max(rocs - generation, 0.0),
                env_other_seconds=max(environment - rocs, 0.0),
                rnn_residual_seconds=max(epoch - environment, 0.0),
                scorer_calls=float(mean.scorer_calls),
                environment_rows=float(mean.environment_rows),
            )
        )
    return pd.DataFrame(rows)


def render_cost_summary(output: Path) -> Path:
    """Aggregate all completed arms into a summary table and 3-panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted(
        path.parent.name for path in (output / "raw_data").glob("*/complete.json")
    )
    if not arms:
        raise ValueError("No completed arms found under raw_data/")
    timing = pd.concat(
        [
            pd.read_csv(output / "raw_data" / name / "batch_timing.csv")
            for name in arms
        ],
        ignore_index=True,
    )
    summary = summarize_timing(timing)
    summary.to_csv(output / "cost_summary.csv", index=False)

    epochs = _collect_arm_tables(arms, output, "epoch_timings.csv")
    scores = _collect_arm_tables(arms, output, "epoch_scores.csv")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=180)
    panel_a, panel_b, panel_c = axes

    # Panel A: warm batch timing, per-molecule stage split from accumulators.
    ordered = summary.sort_values("warm_per_molecule_seconds")
    y = np.arange(len(ordered))
    gen = ordered.gen_per_molecule
    align = ordered.align_io_per_molecule
    other = ordered.env_other_per_molecule
    panel_a.barh(y, gen, color="#31688e", label="Candidate generation + minimization")
    panel_a.barh(y, align, left=gen, color="#35b779", label="ROCS alignment + I/O")
    panel_a.barh(
        y, other, left=gen + align, color="#d581b0", label="Env (validity/SA/desirability)"
    )
    panel_a.axvline(1, color="gray", linestyle=":", linewidth=1, label="1 s/molecule")
    panel_a.set_yticks(y, ordered.arm, fontsize=8)
    panel_a.set_xlabel("Warm per-molecule seconds (full-epoch batch, paired inputs)")
    panel_a.set_title(
        "A. Batch scoring cost by arm (training interface)\n"
        "K=all arms align against full banks; reference counts differ (see 'references')"
    )
    panel_a.legend(fontsize=8)
    panel_a.grid(alpha=0.2, axis="x")

    # Panel B: per-epoch cost breakdown from the RL pilots (stacked stages).
    if epochs is not None:
        breakdown = epoch_stage_breakdown(epochs)
        labels = breakdown.arm.tolist()
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        for key, color, label in STAGE_STYLES:
            values = breakdown[key].to_numpy()
            panel_b.barh(y, values, left=left, color=color, label=label)
            left = left + values
        panel_b.set_yticks(y, labels, fontsize=8)
        panel_b.set_xlabel("Mean seconds per epoch")
        panel_b.set_title("B. RL pilot epoch cost breakdown")
        panel_b.legend(fontsize=8)
        panel_b.grid(alpha=0.2, axis="x")

    # Panel C: validation ROCS trajectory per arm.
    if scores is not None:
        for name, group in scores.groupby("arm", sort=True):
            panel_c.plot(group.epoch, group.mean_rocs, marker="o", label=name)
        panel_c.set_xlabel("Epoch")
        panel_c.set_ylabel("Validation mean ROCS (raw TanimotoCombo)")
        panel_c.set_title("C. Early accuracy trajectory (pilot, not final)")
        panel_c.grid(alpha=0.25)
        panel_c.legend(fontsize=7)

    fig.suptitle(
        "Training-interface cost calibration: full-epoch batches, paired prior inputs, fresh subprocess per arm\n"
        "Cold/warm separated; 5-epoch pilots are early trajectories and engineering costs, not final accuracy",
        fontsize=13,
    )
    fig.tight_layout()
    target = output / "rl_batch_calibration.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# RL batch calibration summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from {len(arms)} completed arms.",
        "",
        "## Batch timing (paired inputs)",
        "",
        summary.to_string(index=False),
    ]
    if epochs is not None:
        lines += [
            "",
            "## Per-epoch cost",
            "",
            epoch_stage_breakdown(epochs).to_string(index=False),
        ]
    if scores is not None:
        lines += ["", "## Validation trajectory", "", scores.to_string(index=False)]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return target


def _collect_arm_tables(
    arms: list[str], output: Path, name: str
) -> pd.DataFrame | None:
    """Concatenate one per-arm CSV table (if any arm produced it)."""
    frames = []
    for arm in arms:
        path = output / "raw_data" / arm / name
        if path.exists():
            frame = pd.read_csv(path)
            if "arm" not in frame.columns:
                frame.insert(0, "arm", arm)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default=",".join(f"{m}:{k}:{c}" for m, k, c in DEFAULT_ARMS),
        help="Comma-separated method:cap:attempts arms (e.g. etkdg:1:30)",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="2 arms, 48 samples, 2 repeats, 1 epoch"
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", nargs=3, metavar=("METHOD", "CAP", "ATTEMPTS"))
    parser.add_argument("--batch-file", type=Path)
    args = parser.parse_args()

    if args.render_only:
        print(f"Saved {render_cost_summary(args.output)}")
        return

    if args.quick:
        args.arms = ",".join(f"{m}:{k}:{c}" for m, k, c in QUICK_ARMS)
        args.samples, args.repeats, args.epochs = 48, 2, 1

    try:
        arms = parse_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))

    if args.child:
        if not args.arm or not args.batch_file:
            parser.error("--child requires --arm and --batch-file")
        run_arm(
            Arm(args.arm[0], int(args.arm[1]), int(args.arm[2])),
            args.output,
            batch_file=args.batch_file,
            samples=args.samples,
            repeats=args.repeats,
            epochs=args.epochs,
            gpu=args.gpu,
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    # Force forkserver BEFORE any torch/pool use in this parent: the prior
    # sampler's optimizer init defaults DRUGEX_MP_CONTEXT to fork (utils only
    # sets it when unset), and prepare_shared_references opens pools in this
    # process after torch is initialized — a fork-after-threads hang hazard
    # for an unattended run. Setting it here wins over every later setdefault.
    os.environ["DRUGEX_MP_CONTEXT"] = "forkserver"
    batch_file = sample_prior_batch(args.output, samples=args.samples, gpu=args.gpu)
    prepare_shared_references(
        args.output, tuple(dict.fromkeys(arm.method for arm in arms))
    )
    environment = dict(os.environ)
    # Forkserver for every child regardless of parent state (the parent's
    # early assignment above is already forkserver; this keeps children
    # correct even if that ever changes).
    environment["DRUGEX_MP_CONTEXT"] = "forkserver"
    environment["MPLBACKEND"] = "Agg"

    def write_manifest(state_failures: list[str]) -> list[str]:
        """Persist run state and re-render; a stopped run stays inspectable."""
        completed = sorted(
            path.parent.name
            for path in (args.output / "raw_data").glob("*/complete.json")
        )
        atomic_json(
            args.output / "manifest.json",
            dict(
                state=(
                    "complete"
                    if not state_failures
                    and {arm.label for arm in arms} <= set(completed)
                    else "partial"
                ),
                completed_utc=datetime.now(timezone.utc).isoformat(),
                arms=[asdict(arm) for arm in arms],
                completed_arms=completed,
                failed_arms=state_failures,
                samples=args.samples,
                repeats=args.repeats,
                epochs=args.epochs,
                batch_file_sha256=digest(batch_file),
                git_revision=_git_revision(),
                source_sha256=source_hashes(SOURCES),
                host=os.uname().nodename,
                note="1:1 training interface; paired prior inputs; fresh subprocess per arm; "
                "5-epoch pilots are early trajectories, not final learning comparisons",
            ),
        )
        if completed:
            try:
                print(f"Saved {render_cost_summary(args.output)}")
            except Exception as exc:  # noqa: BLE001 - rendering must not kill the run
                print(f"Incremental render failed: {exc}", file=sys.stderr)
        return completed

    failures: list[str] = []
    for arm in arms:
        command = [
            sys.executable,
            "-m",
            "_david.code.conformer.rl_batch_calibration",
            "--child",
            "--arm",
            arm.method,
            str(arm.reference_cap),
            str(arm.candidate_attempts),
            "--output",
            str(args.output),
            "--batch-file",
            str(batch_file),
            "--samples",
            str(args.samples),
            "--repeats",
            str(args.repeats),
            "--epochs",
            str(args.epochs),
            "--gpu",
            str(args.gpu),
        ]
        print(f"=== Arm {arm.label} ===", flush=True)
        result = subprocess.run(command, env=environment, check=False)
        if result.returncode != 0:
            failures.append(arm.label)
            print(
                f"Arm {arm.label} failed with exit code {result.returncode}; "
                "continuing with remaining arms",
                file=sys.stderr,
            )
        write_manifest(failures)
    completed = write_manifest(failures)
    if not completed:
        raise RuntimeError(f"All arms failed: {', '.join(failures)}")
    if failures:
        raise SystemExit(
            f"Completed {len(completed)} arm(s); failed: {', '.join(failures)}"
        )


if __name__ == "__main__":
    main()
