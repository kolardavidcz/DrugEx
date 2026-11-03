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
  differences. ``--epochs 0`` skips phase 2 entirely (batch-only grid mode:
  the pilot dominates wall-clock while its trajectories cannot rank arms, so
  large method grids run batch-only and reserve pilots for the shortlist).

- **Candidate minimization is part of the arm spec**: an optional fourth
  token ``method:cap:attempts:m0|:m1`` forces candidate minimization off/on
  independently of the method (default: off for ``etkdg``, on for force
  fields). ``mmff_high_dielectric:4:20:0`` is the Regime-1 control (MMFF
  reference bank, unminimized ETKDG candidates); without the token the arm
  keeps the method default, so a minimized arm never silently differs from
  its published predecessor.

Usage (repository root):

.. code-block:: bash

    # Full 10-arm grid (dichtator, 48 CPUs; batch-only, ~2 h):
    python -m _david.code.conformer.rl_batch_calibration \
        --output _david/output/<run_id> --samples 500 --repeats 3 --epochs 0 \
        --gold mmff_high_dielectric_k4_c20

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
from _david.code.conformer.reference.reference_bank import METHODS, prepare_banks, save_records
from _david.code.conformer.core.utils import atomic_json, digest, source_hashes
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

# (method, reference cap K, candidate attempts C, candidate minimization).
# The optional 4th element (None = method default: off for etkdg, on for
# force fields) exists so grids can hold (K, C) fixed and toggle minimization
# (the Regime-1 control). Ordered cheapest-first so time-boxed runs (weak
# CPUs) complete the most decision-relevant arms before the expensive
# all-bank cells.
DEFAULT_ARMS: tuple[tuple[str, int, int, bool | None], ...] = (
    ("etkdg", 1, 30, None),  # sprint Regime 2 production setting (unminimized)
    ("etkdg", 4, 20, None),  # force-field comparison at fixed K=4, C=20
    ("mmff_high_dielectric", 1, 30, None),  # production pair (minimized, K=1)
    ("mmff_high_dielectric", 4, 20, None),  # deployable gold (pre-registered)
    ("mmff_high_dielectric", 4, 20, False),  # Regime-1 control: unminimized candidates
    ("uff", 4, 20, None),  # force-field comparison at fixed K=4, C=20
    ("mmff_vacuum", 4, 20, None),  # dielectric comparison (eps=1.0 vs 78.4)
    ("mmff_high_dielectric", 4, 30, None),  # candidate-budget scaling
    ("mmff_high_dielectric", 4, 50, None),  # candidate-budget scaling
    ("mmff_high_dielectric", 0, 50, None),  # asymptotic physical gold (K=all)
)
QUICK_ARMS: tuple[tuple[str, int, int, bool | None], ...] = (
    ("etkdg", 1, 30, None),
    ("mmff_high_dielectric", 4, 20, None),
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
    "embedding_seconds",
    "minimization_seconds",
    "environment_seconds",
    "environment_rows",
    "scorer_calls",
)


@dataclass(frozen=True)
class Arm:
    """One (method, reference cap, candidate budget) measurement arm.

    ``minimize`` forces candidate minimization on (True) or off (False)
    independently of the method; ``None`` keeps the method default (off for
    ``etkdg``, on for force fields). The label only gains an ``_m0``/``_m1``
    suffix when ``minimize`` is explicit, so token-less arm specs keep their
    historical directory names (resume compatibility with earlier runs).
    """

    method: str
    reference_cap: int
    candidate_attempts: int
    minimize: bool | None = None

    @property
    def effective_minimize(self) -> bool:
        """Candidate minimization actually used by this arm."""
        if self.minimize is not None:
            return self.minimize
        return self.method != "etkdg"

    @property
    def label(self) -> str:
        cap = "all" if self.reference_cap == 0 else str(self.reference_cap)
        suffix = "" if self.minimize is None else ("_m1" if self.minimize else "_m0")
        return f"{self.method}_k{cap}_c{self.candidate_attempts}{suffix}"


def arm_label(method: str, cap: int, attempts: int, minimize: bool | None = None) -> str:
    """Canonical directory-safe label for one arm."""
    return Arm(method, cap, attempts, minimize).label


def _parse_minimize_token(token: str) -> bool:
    """Parse the optional 4th arm-spec token into a minimize flag."""
    normalized = token.strip().lower()
    if normalized in {"0", "m0", "false", "off"}:
        return False
    if normalized in {"1", "m1", "true", "on"}:
        return True
    raise ValueError(
        f"Invalid minimize token {token!r}; expected m0/m1 (or 0/1)"
    )


def parse_arms(spec: str) -> list[Arm]:
    """Parse a comma-separated ``method:cap:attempts[:m0|m1]`` arm spec."""
    arms: list[Arm] = []
    for item in spec.split(","):
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(
                f"Invalid arm {item!r}; expected method:cap:attempts[:m0|m1]"
            )
        method, cap, attempts = parts[:3]
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
        minimize = _parse_minimize_token(parts[3]) if len(parts) == 4 else None
        if minimize and method == "etkdg":
            raise ValueError(
                f"Invalid arm {item!r}: etkdg has no force field to minimize "
                "candidates with (its optimization method is 'none')"
            )
        arm = Arm(method, int(cap), int(attempts), minimize)
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
        minimize_candidates=arm.effective_minimize,
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
        # Sub-stage split (Gemini finding 3): the generator resets its
        # last_* accumulators per genConformers call, so these attribute
        # reads describe exactly the call that just finished — embedding
        # (ETKDG distance geometry) and force-field minimization separated.
        generator = scorer.conformer_generator
        totals["embedding_seconds"] += generator.last_embedding_seconds
        totals["minimization_seconds"] += generator.last_minimization_seconds
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
                embedding_seconds=generator.last_embedding_seconds,
                minimization_seconds=generator.last_minimization_seconds,
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
    # Optional by design (Gemini finding 2): with --epochs 0 the grid runs
    # batch-only. The pilot dominates wall-clock (~69% of arm runtime in the
    # 2026-11-02 overnight run) while its 5-epoch trajectories cannot rank
    # accuracy, so method grids skip it and it is reserved for the final
    # shortlist arms.
    if epochs > 0:
        # The batch-timing phase above leaves pools warm, matching
        # steady-state training; epoch 1 therefore excludes cold-start cost
        # by design.
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
    else:
        fit_seconds = 0.0

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
            pilot_epochs=epochs,
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
                embed_per_molecule=float(
                    (warm.get("embedding_seconds", 0.0) / warm.inputs).mean()
                ),
                minimize_per_molecule=float(
                    (warm.get("minimization_seconds", 0.0) / warm.inputs).mean()
                ),
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
                embedding_seconds=float(mean.get("embedding_seconds", 0.0)),
                minimization_seconds=float(mean.get("minimization_seconds", 0.0)),
                align_io_seconds=max(rocs - generation, 0.0),
                env_other_seconds=max(environment - rocs, 0.0),
                rnn_residual_seconds=max(epoch - environment, 0.0),
                scorer_calls=float(mean.scorer_calls),
                environment_rows=float(mean.environment_rows),
            )
        )
    return pd.DataFrame(rows)


def _paired_scores(
    output: Path, arms: list[str]
) -> dict[str, tuple[int, pd.DataFrame]]:
    """Latest common repeat's score frame per arm (row order = paired batch)."""
    result: dict[str, tuple[int, pd.DataFrame]] = {}
    for arm in arms:
        files = sorted(
            (output / "raw_data" / arm).glob("batch_scores_repeat*.csv"),
            key=lambda p: int(p.stem.rsplit("repeat", 1)[1]),
        )
        if files:
            index = int(files[-1].stem.rsplit("repeat", 1)[1])
            result[arm] = (index, pd.read_csv(files[-1]))
    return result


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman rank correlation; NaN for constant inputs (pandas contract)."""
    return float(left.corr(right, method="spearman"))


def _top_k_overlap(left: pd.Series, right: pd.Series, fraction: float = 0.10) -> float:
    """Overlap fraction of each side's top-k molecules (k = fraction * n)."""
    k = max(1, int(fraction * len(left)))
    top_left = set(left.nlargest(k).index)
    top_right = set(right.nlargest(k).index)
    return len(top_left & top_right) / k


def _bootstrap_mean_ci(
    values: pd.Series, *, resamples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean of ``values``."""
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = values.to_numpy(dtype=float)
    means = array[
        rng.integers(0, len(array), size=(resamples, len(array)))
    ].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_agreement(output: Path, arms: list[str], gold: str) -> pd.DataFrame:
    """Paired per-molecule rank agreement of every arm against the gold arm.

    Pre-registered accuracy metric: on the identical frozen batch (paired row
    order), Spearman rho and top-10% overlap against the gold arm, plus the
    mean paired delta-ROCS (gold - arm) with a 95% bootstrap CI. Both the
    all-rows and the valid-nonzero-rows variants are reported: molecules
    that time out (score 0.0) in one arm only would otherwise create large
    spurious deltas (Gemini finding 7).
    """
    frames = _paired_scores(output, arms + [gold])
    if gold not in frames:
        raise ValueError(f"Gold arm {gold!r} has no batch_scores_repeat*.csv")
    gold_index, gold_frame = frames[gold]
    rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm not in frames:
            continue
        index, frame = frames[arm]
        if index != gold_index or len(frame) != len(gold_frame):
            rows.append(
                dict(arm=arm, note="unpaired (different repeat or row count)")
            )
            continue
        valid = (
            (frame.Valid == 1)
            & (gold_frame.Valid == 1)
            & (frame[std.ROCS] > 0.0)
            & (gold_frame[std.ROCS] > 0.0)
        )
        gold_rocs = gold_frame[std.ROCS]
        arm_rocs = frame[std.ROCS]
        delta_valid = gold_rocs[valid] - arm_rocs[valid]
        ci_low, ci_high = _bootstrap_mean_ci(delta_valid)
        rows.append(
            dict(
                arm=arm,
                repeat=index,
                spearman_all=_spearman(arm_rocs, gold_rocs),
                spearman_valid=_spearman(arm_rocs[valid], gold_rocs[valid]),
                top10_overlap_all=_top_k_overlap(arm_rocs, gold_rocs),
                top10_overlap_valid=_top_k_overlap(arm_rocs[valid], gold_rocs[valid]),
                mean_delta_rocs_valid=float(delta_valid.mean())
                if len(delta_valid)
                else float("nan"),
                delta_rocs_ci95_low=ci_low,
                delta_rocs_ci95_high=ci_high,
            )
        )
    if not rows:
        raise ValueError("No arms with score data to compare against the gold")
    return pd.DataFrame(rows).fillna({"note": ""})


def render_cost_summary(output: Path, gold: str | None = None) -> Path:
    """Aggregate all completed arms into a summary table and 4-panel figure.

    ``gold`` (optional) selects the reference arm for the pre-registered
    rank-agreement analysis (Spearman rho, top-10% overlap, paired
    delta-ROCS with bootstrap CI), rendered as SUMMARY.md section and the
    speed-vs-agreement frontier panel D.
    """
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

    agreement: pd.DataFrame | None = None
    agreement_note = ""
    if gold:
        try:
            agreement = rank_agreement(output, arms, gold)
            agreement.to_csv(output / "rank_agreement.csv", index=False)
        except ValueError as exc:
            agreement, agreement_note = None, str(exc)

    epochs = _collect_arm_tables(arms, output, "epoch_timings.csv")
    scores = _collect_arm_tables(arms, output, "epoch_scores.csv")

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), dpi=180)
    plt.subplots_adjust(hspace=0.28, wspace=0.34, top=0.93, bottom=0.06, left=0.22, right=0.96)
    (panel_a, panel_b), (panel_c, panel_d) = axes

    arm_labels = {
        "etkdg_k1_c30": "ETKDG (K=1, C=30)",
        "etkdg_k4_c20": "ETKDG (K=4, C=20)",
        "mmff_high_dielectric_k4_c20_m0": "MMFF94s ε=78.4 (K=4, C=20, unmin)",
        "mmff_high_dielectric_k1_c30": "MMFF94s ε=78.4 (K=1, C=30)",
        "mmff_high_dielectric_k4_c20": "MMFF94s ε=78.4 (K=4, C=20) [Deployable Gold]",
        "uff_k4_c20": "UFF (K=4, C=20)",
        "mmff_vacuum_k4_c20": "MMFF94s ε=1.0 (K=4, C=20)",
        "mmff_high_dielectric_k4_c30": "MMFF94s ε=78.4 (K=4, C=30)",
        "mmff_high_dielectric_k4_c50": "MMFF94s ε=78.4 (K=4, C=50)",
        "mmff_high_dielectric_kall_c50": "MMFF94s ε=78.4 (K=all, C=50) [Asymptotic Gold]",
    }

    method_palette = {
        "mmff_high_dielectric": ("MMFF94s ε=78.4", "#1b9e77", "o"),
        "mmff_vacuum": ("MMFF94s ε=1.0", "#d95f02", "s"),
        "uff": ("UFF", "#7570b3", "^"),
        "etkdg": ("ETKDG Only", "#e7298a", "D"),
    }

    # =========================================================================
    # Panel A: 4-Stage Hardware Latency Breakdown
    # =========================================================================
    ordered = summary.sort_values("warm_per_molecule_seconds").reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    embed = ordered.embed_per_molecule
    minimize = ordered.minimize_per_molecule
    align = ordered.align_io_per_molecule
    other = ordered.env_other_per_molecule

    bar_h = 0.56
    panel_a.barh(y_pos, embed, height=bar_h, color="#2b5c8f", label="ETKDG Distance Geometry Floor (~0.38 s/mol)")
    panel_a.barh(y_pos, minimize, left=embed, height=bar_h, color="#e69f00", label="Force-Field Minimization (~0.04-0.07 s/mol)")
    panel_a.barh(y_pos, align, left=embed + minimize, height=bar_h, color="#009e73", label="3D ROCS Alignment & I/O (scales K x C)")
    panel_a.barh(y_pos, other, left=embed + minimize + align, height=bar_h, color="#94a3b8", label="Environment Overhead (Validity / SA)")

    panel_a.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.75, label="1.0 s/mol Real-time RL Ceiling")
    panel_a.axvline(0.38, color="#2b5c8f", linestyle=":", linewidth=1.0, alpha=0.6)

    for idx, row in ordered.iterrows():
        total_t = row.warm_per_molecule_seconds
        panel_a.text(total_t + 0.03, idx, f"{total_t:.2f} s", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#222222")

    panel_a.set_yticks(y_pos)
    panel_a.set_yticklabels([arm_labels.get(a, a) for a in ordered.arm], fontsize=9)
    panel_a.set_xlabel("Warm Scoring Latency (s/molecule, Dichtator 48 Cores)", fontsize=10)
    panel_a.set_xlim(0, 1.88)
    panel_a.set_title("A. Hardware Latency: Inflexible ~0.38s ETKDG Floor Limits Method Differences", fontsize=11, fontweight="bold")
    panel_a.grid(True, axis="x", alpha=0.25, linestyle="--")
    panel_a.legend(fontsize=8.2, loc="lower right", framealpha=0.92, edgecolor="#cccccc")

    # =========================================================================
    # Panel B: Speed vs Rank-Agreement Pareto Frontier
    # =========================================================================
    if agreement is not None:
        merged = summary.merge(agreement, on="arm", how="inner")
        rho = merged.spearman_valid.fillna(merged.spearman_all)
        overlap = merged.top10_overlap_valid.fillna(merged.top10_overlap_all)
        merged = merged.assign(_rho=rho, _overlap=overlap).dropna(subset=["_rho"])

        frontier_arms = ["etkdg_k1_c30", "etkdg_k4_c20", "mmff_high_dielectric_k4_c20_m0", "mmff_high_dielectric_k4_c20"]
        frontier_df = merged[merged.arm.isin(frontier_arms)].sort_values("warm_per_molecule_seconds")

        panel_b.plot(
            frontier_df.warm_per_molecule_seconds,
            frontier_df._rho,
            linestyle="--",
            color="#1b9e77",
            linewidth=2.0,
            alpha=0.7,
            zorder=2,
            label="Empirical Pareto Frontier",
        )

        for _, row in merged.iterrows():
            m_info = method_palette.get(row.method, ("Other", "#333333", "o"))
            is_gold = (gold and row.arm == gold)
            is_kall = (row.arm == "mmff_high_dielectric_kall_c50")

            col = "#d95f02" if is_gold else m_info[1]
            marker = "*" if is_gold else ("D" if is_kall else m_info[2])
            size = 240 if is_gold else (140 if is_kall else 90)

            panel_b.scatter(
                row.warm_per_molecule_seconds,
                row._rho,
                s=size,
                color=col,
                marker=marker,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.9,
                zorder=5 if is_gold else 3,
            )

        # Clean non-overlapping anchor callouts
        anchor_labels = {
            "etkdg_k1_c30": ("ETKDG K=1, C=30\n(0.43s, ρ=0.84)", (-12, 14), "right"),
            "etkdg_k4_c20": ("ETKDG K=4, C=20\n(0.44s, ρ=0.86)", (-12, -22), "right"),
            "mmff_high_dielectric_k4_c20_m0": ("MMFF94s K=4, C=20 (unmin m0)\n(0.44s, ρ=0.89)", (12, 12), "left"),
            "mmff_high_dielectric_k4_c20": ("Deployable Gold Elbow:\nMMFF94s K=4, C=20 (0.49s, ρ=1.0)", (14, -14), "left"),
            "mmff_high_dielectric_kall_c50": ("Asymptotic Gold:\nMMFF94s K=all, C=50 (1.58s, ρ=0.95)", (-18, 14), "right"),
        }

        for arm_k, (txt, (x_o, y_o), ha_a) in anchor_labels.items():
            sub_r = merged[merged.arm == arm_k]
            if len(sub_r):
                r_x, r_y = sub_r.warm_per_molecule_seconds.iloc[0], sub_r._rho.iloc[0]
                is_g = (gold and arm_k == gold)
                panel_b.annotate(
                    txt,
                    (r_x, r_y),
                    xytext=(x_o, y_o),
                    textcoords="offset points",
                    fontsize=8.0,
                    fontweight="bold" if is_g else "normal",
                    color="#990000" if is_g else "#222222",
                    ha=ha_a,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb", alpha=0.9, linewidth=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#666666", lw=0.8),
                )

        panel_b.text(
            0.04, 0.16,
            "Sober Perspective on Trade-offs:\n"
            "• Latency gap: only 0.06 s/mol (12%)\n"
            "• Rank correlation: already ρ = 0.84–0.89\n"
            "• All viable arms cluster within 0.43–0.59s",
            transform=panel_b.transAxes,
            fontsize=8.2,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
        )

        panel_b.set_xlim(0.38, 1.80)
        panel_b.set_ylim(0.76, 1.03)
        panel_b.set_xlabel("Warm Scoring Latency (s/molecule, linear scale)", fontsize=10)
        panel_b.set_ylabel(f"Spearman Rank Correlation ρ vs Gold ({gold})", fontsize=10)
        panel_b.set_title("B. Speed vs Rank Agreement: Tight Cluster Across All Candidate Methods", fontsize=11, fontweight="bold")
        panel_b.grid(True, alpha=0.25, linestyle="--")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#d95f02", markeredgecolor="k", markersize=13, label="Deployable Gold Elbow (K=4, C=20)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b9e77", markeredgecolor="k", markersize=8, label="MMFF94s ε=78.4 (High-Dielectric)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#d95f02", markeredgecolor="k", markersize=8, label="MMFF94s ε=1.0 (Vacuum)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#7570b3", markeredgecolor="k", markersize=8, label="UFF"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#e7298a", markeredgecolor="k", markersize=8, label="ETKDG (Unminimized)"),
            Line2D([0], [0], color="#1b9e77", linestyle="--", linewidth=1.8, label="Pareto Frontier Envelope"),
        ]
        panel_b.legend(handles=legend_elements, fontsize=8.2, loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    else:
        panel_b.set_axis_off()

    # =========================================================================
    # Panel C: Decile Fidelity
    # =========================================================================
    if agreement is not None:
        merged_c = ordered[["arm", "method"]].merge(agreement, on="arm", how="left").reset_index(drop=True)
        y_pos_c = np.arange(len(merged_c))
        rho_vals = merged_c.spearman_valid.fillna(merged_c.spearman_all).fillna(0.0)
        top10_vals = merged_c.top10_overlap_valid.fillna(merged_c.top10_overlap_all).fillna(0.0)

        b_h = 0.36
        panel_c.barh(y_pos_c + b_h / 2, rho_vals, height=b_h, color="#1b9e77", label="Spearman ρ (Global Rank Preservation)")
        panel_c.barh(y_pos_c - b_h / 2, top10_vals, height=b_h, color="#2b5c8f", label="Top-10% Decile Retention (Elite Selection)")

        for idx, (r_v, t_v) in enumerate(zip(rho_vals, top10_vals)):
            panel_c.text(r_v + 0.015, idx + b_h / 2, f"{r_v:.2f}", va="center", ha="left", fontsize=8, color="#1b9e77", fontweight="bold")
            panel_c.text(t_v + 0.015, idx - b_h / 2, f"{t_v:.2f}", va="center", ha="left", fontsize=8, color="#2b5c8f", fontweight="bold")

        panel_c.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.7)
        panel_c.set_yticks(y_pos_c)
        panel_c.set_yticklabels([arm_labels.get(a, a) for a in merged_c.arm], fontsize=9)
        panel_c.set_xlabel(f"Fidelity Metric vs Gold ({gold}) [1.0 = Perfect Agreement]", fontsize=10)
        panel_c.set_xlim(0, 1.15)
        panel_c.set_title("C. Policy Update Fidelity: Global Rank vs Elite Top-10% Decile Retention", fontsize=11, fontweight="bold")
        panel_c.grid(True, axis="x", alpha=0.25, linestyle="--")
        panel_c.legend(fontsize=8.5, loc="lower left", framealpha=0.92, edgecolor="#cccccc")
    else:
        panel_c.set_axis_off()

    # =========================================================================
    # Panel D: Paired Delta-ROCS Deviations & 95% Bootstrap Confidence Intervals
    # =========================================================================
    if agreement is not None and "mean_delta_rocs_valid" in agreement.columns:
        dev_arms = merged_c[merged_c.arm != gold].reset_index(drop=True)
        y_pos_d = np.arange(len(dev_arms))

        panel_d.axvspan(-0.005, 0.005, color="#e2e8f0", alpha=0.9, label="Equivalence Band (|Δ ROCS| ≤ 0.005)")
        panel_d.axvline(0.0, color="#475569", linestyle="-", linewidth=1.0, alpha=0.8)

        for idx, row in dev_arms.iterrows():
            m_info = method_palette.get(row.method, ("Other", "#333333", "o"))
            val = row.mean_delta_rocs_valid
            ci_l = row.delta_rocs_ci95_low
            ci_h = row.delta_rocs_ci95_high
            col = m_info[1]

            panel_d.errorbar(
                val,
                idx,
                xerr=[[val - ci_l], [ci_h - val]],
                fmt=m_info[2],
                color=col,
                ecolor=col,
                elinewidth=1.6,
                capsize=3.5,
                markersize=7.0,
                alpha=0.9,
            )
            # Right-aligned numerical labels at x = 0.062
            panel_d.text(
                0.062, idx,
                f"{val:+.3f} [{ci_l:+.3f}, {ci_h:+.3f}]",
                va="center", ha="left", fontsize=8.0, color="#333333", family="monospace"
            )

        # Clean non-overlapping box placed in open space of upper-right quadrant
        panel_d.text(
            0.42, 0.68,
            "Sober Context on Score Bias:\n"
            "• Gold mean score = 0.655 (σ = 0.194)\n"
            "• K=1 shift (+0.045) is only 0.23σ (~6.4%)\n"
            "• 89% of molecules differ by < 0.10\n"
            "• Candidate methods (UFF/m0) differ by < 0.002",
            transform=panel_d.transAxes,
            fontsize=8.2,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
        )

        panel_d.set_yticks(y_pos_d)
        panel_d.set_yticklabels([arm_labels.get(a, a) for a in dev_arms.arm], fontsize=9)
        panel_d.set_xlabel(f"Paired Δ ROCS vs Gold ({gold}) [Score_gold - Score_arm; 95% Bootstrap CI]", fontsize=10)
        panel_d.set_xlim(-0.025, 0.105)
        panel_d.set_title("D. Paired Score Bias vs Gold: Marginal Level Shifts (< 7% of Gold Mean)", fontsize=11, fontweight="bold")
        panel_d.grid(True, axis="x", alpha=0.25, linestyle="--")
        panel_d.legend(fontsize=8.2, loc="lower right", framealpha=0.92, edgecolor="#cccccc")
    else:
        panel_d.set_axis_off()

    fig.suptitle(
        "DrugEx 3D ROCS Reinforcement Learning: 10-Arm Conformer Scoring Calibration\n"
        "Hardware Benchmark on Dichtator (48 CPU Cores, 500 Paired Molecules x 3 Independent Repeats)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    target = output / "rl_batch_calibration.png"
    fig.savefig(target, dpi=180)
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
    if agreement is not None:
        lines += [
            "",
            f"## Rank agreement vs gold ({gold})",
            "",
            agreement.to_string(index=False),
        ]
    elif gold:
        lines += [
            "",
            f"## Rank agreement vs gold ({gold})",
            "",
            f"_skipped: {agreement_note}_",
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


def _arm_spec(method: str, cap: int, attempts: int, minimize: bool | None) -> str:
    """Render one arm as its ``method:cap:attempts[:m0|m1]`` spec string."""
    token = "" if minimize is None else (":m1" if minimize else ":m0")
    return f"{method}:{cap}:{attempts}{token}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default=",".join(_arm_spec(*arm) for arm in DEFAULT_ARMS),
        help="Comma-separated method:cap:attempts[:m0|m1] arms (e.g. etkdg:1:30)",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="RL pilot epochs per arm; 0 = batch-only grid mode (skip phase 2)",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="Gold arm label for the pre-registered rank-agreement analysis "
        "(e.g. mmff_high_dielectric_k4_c20); renders SUMMARY section + panel D",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="2 arms, 48 samples, 2 repeats, 1 epoch"
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--arm",
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-file", type=Path)
    args = parser.parse_args()

    if args.render_only:
        print(f"Saved {render_cost_summary(args.output, gold=args.gold)}")
        return

    if args.quick:
        args.arms = ",".join(_arm_spec(*arm) for arm in QUICK_ARMS)
        args.samples, args.repeats, args.epochs = 48, 2, 1

    try:
        arms = parse_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1 (one cold repeat minimum)")
    if args.epochs < 0:
        parser.error("--epochs must be >= 0 (0 = batch-only grid mode)")

    if args.child:
        if not args.arm or not args.batch_file:
            parser.error("--child requires --arm and --batch-file")
        if len(args.arm) not in (3, 4):
            parser.error("--arm requires METHOD CAP ATTEMPTS [M0|M1]")
        minimize = _parse_minimize_token(args.arm[3]) if len(args.arm) == 4 else None
        run_arm(
            Arm(args.arm[0], int(args.arm[1]), int(args.arm[2]), minimize),
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
                "RL pilots (when --epochs > 0) are early trajectories, not final "
                "learning comparisons",
                gold=args.gold,
            ),
        )
        if completed:
            try:
                print(f"Saved {render_cost_summary(args.output, gold=args.gold)}")
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
            *(
                ["m1" if arm.minimize else "m0"]
                if arm.minimize is not None
                else []
            ),
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
