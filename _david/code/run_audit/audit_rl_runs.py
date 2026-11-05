"""Audit production RL runs against the current gold-standard config and figure suite.

The thesis has two kinds of output that are easy to confuse:

* a *real* reinforcement-learning run - a SequenceRNN trained for a meaningful
  horizon and then evaluated with a baseline arm and a reinforced arm;
* a *calibration* or *smoke* study, which shares the run-directory layout but
  answers a different question.

Only the first kind belongs in the thesis results table. This module discovers
them from ``manifest.json``, folds byte-identical copies together, compares each
one's recorded ``config`` with ``PipelineConfig()`` (the gold standard), checks
the deliverable suite on disk, and optionally rebuilds every missing figure from
the stored raw arms.

Usage
-----
    # report only, no writes outside the report files
    python -m _david.code.audit_rl_runs

    # rebuild every missing artifact first, then report
    python -m _david.code.audit_rl_runs --regenerate

    # explicit destinations
    python -m _david.code.audit_rl_runs \
        --write _david/code/RL_RUN_AUDIT.md \
        --csv   _david/output/RL_RUN_AUDIT.csv

The report is deliberately honest about what cannot be recovered: a run whose
raw arm is missing, or whose model did not record per-epoch SMILES, cannot be
made comparable after the fact. Those are marked ``UNVERIFIABLE`` rather than
silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT, PipelineConfig, RunPaths
from _david.code.schema import std

OUTPUT_ROOT: Path = PROJECT_ROOT / "_david" / "output"
DEFAULT_REPORT: Path = PROJECT_ROOT / "_david" / "code" / "RL_RUN_AUDIT.md"
DEFAULT_CSV: Path = OUTPUT_ROOT / "RL_RUN_AUDIT.csv"

# The current pipeline renders these thirteen files. Figure 04 is a historical
# diagnostic that the current ResultAnalyzer no longer calls, so it is audited
# separately and never decides the verdict. Rows retired on 2026-11-04
# (``06_relative_property_shifts_tornado``, ``08_rocs_shape_vs_color_decomposition``,
# ``reference_3d_superposition``) are no longer required or regenerated.
CURRENT_FIGURES: tuple[str, ...] = (
    "01_before_after_distributions.png",
    "02_multiobjective_tradeoffs.png",
    "03_training_trajectories.png",
    "05_lead_structures_gallery.png",
    "06_objective_conflict_correlations.png",
    "07_bemis_murcko_scaffold_diversity.png",
    "10_lead_to_seed_3d_superposition.png",
    "11_epoch_property_trajectories.png",
    "12_multiobjective_hypervolume.png",
    "13_exploration_exploitation.png",
    "14_sampling_budget.png",
    "15_lead_set_stability.png",
    "16_rocs_enrichment.png",
)
LEGACY_FIGURES: tuple[str, ...] = ("04_diagnostic_unfavorable_features.png",)
SUPPORT_FILES: tuple[str, ...] = (
    "top_prioritized_leads.tsv",
    "comparison_scorecard.csv",
    "summary.md",
)

# The fields a reader must know to reproduce a run. Every one is compared with
# the gold standard; the rest of the 50+ config keys are kept in the CSV.
DECISION_FIELDS: tuple[tuple[str, str], ...] = (
    ("epochs", "training horizon"),
    ("criteria", "checkpoint criterion"),
    ("score_variant", "similarity variant"),
    ("rocs_use_modifier", "raw vs clipped ROCS"),
    ("rocs_clip_upper", "ROCS clip bound (inactive when raw)"),
    ("n_eval_samples", "final evaluation size"),
    ("reward_scheme", "reward scheme"),
    ("include_qed", "QED objective"),
    ("epsilon", "exploration"),
    ("use_epsilon_decay", "epsilon schedule"),
    ("beta", "baseline penalty"),
    ("thresholds", "desirability gates"),
    ("max_conformers", "candidate budget C"),
    ("max_isomers", "isomer cap I"),
    ("max_reference_conformers", "reference cap K"),
    ("minimize_candidates", "candidate minimization"),
    ("optimization", "force field"),
    ("timeout_seconds", "per-molecule timeout"),
    ("random_seed", "random seed"),
)

SMOKE_EPOCH_LIMIT = 5


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _looks_like_rl(manifest: dict[str, Any] | None) -> bool:
    config = (manifest or {}).get("config") or {}
    return "reward_scheme" in config and "n_samples" in config


def _looks_like_legacy_run(run_dir: Path) -> bool:
    """A pre-manifest run: model telemetry plus both evaluation arms."""
    models = run_dir / "models"
    raw = run_dir / "raw_data"
    return (
        (models / "SequenceRNN_reinforced_fit.tsv").is_file()
        and (raw / "reinforced_raw.csv.gz").is_file()
        and (raw / "baseline_raw.csv.gz").is_file()
    )


def _fingerprint(run_dir: Path) -> str | None:
    """Content identity of a run: both evaluation arms, byte for byte."""
    raw = run_dir / "raw_data"
    arms = (raw / "baseline_raw.csv.gz", raw / "reinforced_raw.csv.gz")
    if not all(arm.is_file() for arm in arms):
        return None
    digest = hashlib.sha256()
    for arm in arms:
        digest.update(_sha256(arm).encode())
    return digest.hexdigest()


@dataclass
class RunAudit:
    path: Path
    rel: str
    manifest: dict[str, Any] | None
    provenance: dict[str, Any] | None
    state: str
    host: str
    pbs_job_id: str
    source_revision: str
    elapsed_seconds: float | None
    query: str
    config: dict[str, Any]
    fingerprint: str | None
    archived: bool = False
    copies: list[str] = field(default_factory=list)

    @property
    def figures(self) -> Path:
        return self.path / "figures"

    def figure_presence(self) -> dict[str, bool]:
        return {name: (self.figures / name).is_file() for name in CURRENT_FIGURES}

    def legacy_presence(self) -> dict[str, bool]:
        return {name: (self.figures / name).is_file() for name in LEGACY_FIGURES}

    def support_presence(self) -> dict[str, bool]:
        return {name: (self.path / name).is_file() for name in SUPPORT_FILES}

    @property
    def is_complete(self) -> bool:
        return self.state.startswith("complete")

    @property
    def is_smoke(self) -> bool:
        epochs = self.config.get("epochs")
        return isinstance(epochs, (int, float)) and epochs <= SMOKE_EPOCH_LIMIT

    @property
    def raw_ready(self) -> bool:
        raw = self.path / "raw_data"
        return (raw / "reinforced_raw.csv.gz").is_file() and (
            raw / "baseline_raw.csv.gz"
        ).is_file()

    @property
    def can_regenerate(self) -> bool:
        models = self.path / "models"
        return (
            self.raw_ready
            and (models / "SequenceRNN_reinforced_smiles.tsv").is_file()
            and (models / "SequenceRNN_reinforced_fit.tsv").is_file()
        )

    def current_figure_count(self) -> int:
        return sum(self.figure_presence().values())

    def legacy_figure_count(self) -> int:
        return sum(self.legacy_presence().values())

    def verdict(self) -> str:
        if not self.is_complete:
            return "INCOMPLETE"
        if not self.raw_ready:
            return "UNVERIFIABLE (missing raw arm)"
        if not self.can_regenerate:
            return "UNVERIFIABLE (missing model telemetry)"
        missing = self.current_figure_count()
        if missing < len(CURRENT_FIGURES):
            return f"UNVERIFIABLE (figures {missing}/{len(CURRENT_FIGURES)})"
        if not all(self.support_presence().values()):
            return "UNVERIFIABLE (missing scorecard/leads/summary)"
        legacy = self.legacy_figure_count()
        if legacy < len(LEGACY_FIGURES):
            return "VERIFIABLE (legacy 04 missing)"
        return "VERIFIABLE"


def _resolve_query(run_dir: Path, manifest: dict[str, Any] | None) -> str:
    query = (manifest or {}).get("query")
    if query:
        return str(query)
    sdf = run_dir / "raw_data" / "query_molecule_3d.sdf"
    if sdf.is_file():
        from rdkit import Chem

        mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return DEFAULT_MOLECULE


def _legacy_provenance(run_dir: Path) -> dict[str, Any] | None:
    provenance = run_dir / "provenance.json"
    if not provenance.is_file():
        return None
    try:
        return json.loads(provenance.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _make_audit(run_dir: Path) -> RunAudit | None:
    manifest = _read_manifest(run_dir)
    provenance = _legacy_provenance(run_dir)
    rel = run_dir.relative_to(OUTPUT_ROOT).as_posix()
    archived = "/archive/" in f"/{rel}/"

    if manifest is not None:
        if not _looks_like_rl(manifest):
            return None
        config = manifest.get("config") or {}
        state = str(manifest.get("state", "unknown"))
        host = str(manifest.get("host", ""))
        pbs = str(manifest.get("pbs_job_id") or "")
        revision = str(manifest.get("source_revision") or "")
        elapsed = manifest.get("elapsed_seconds")
        query = _resolve_query(run_dir, manifest)
    elif provenance is not None and _looks_like_legacy_run(run_dir):
        # Supervisor reproduction: provenance.json carries only a handful of
        # fields, so the rest of the config stays unknown.
        config = {
            "epochs": provenance.get("epochs"),
            "n_samples": provenance.get("n_samples"),
            "epsilon": provenance.get("epsilon"),
            "random_seed": provenance.get("seed"),
        }
        state = "complete (legacy, no manifest)"
        host = ""
        pbs = ""
        revision = ""
        elapsed = (
            float(provenance["training_minutes"]) * 60.0
            if provenance.get("training_minutes")
            else None
        )
        query = _resolve_query(run_dir, None)
    elif _looks_like_legacy_run(run_dir):
        config = {}
        state = "complete (legacy, no manifest)"
        host = ""
        pbs = ""
        revision = ""
        elapsed = None
        query = _resolve_query(run_dir, None)
    else:
        return None

    return RunAudit(
        path=run_dir,
        rel=rel,
        manifest=manifest,
        provenance=provenance,
        state=state,
        host=host,
        pbs_job_id=pbs,
        source_revision=revision,
        elapsed_seconds=float(elapsed) if elapsed else None,
        query=query,
        config=config,
        fingerprint=_fingerprint(run_dir),
        archived=archived,
    )


def discover(output_root: Path = OUTPUT_ROOT) -> list[RunAudit]:
    """Find every run directory, including archived copies, then deduplicate."""
    found: list[RunAudit] = []
    for manifest in sorted(output_root.rglob("manifest.json")):
        audit = _make_audit(manifest.parent)
        if audit is not None:
            found.append(audit)
    for telemetry in sorted(output_root.rglob("SequenceRNN_reinforced_fit.tsv")):
        run_dir = telemetry.parent.parent
        if (run_dir / "manifest.json").is_file():
            continue
        audit = _make_audit(run_dir)
        if audit is not None:
            found.append(audit)
    return _deduplicate(found)


def _deduplicate(runs: list[RunAudit]) -> list[RunAudit]:
    """Fold byte-identical runs into one entry, keeping the fullest copy."""
    groups: dict[str, list[RunAudit]] = {}
    for run in runs:
        key = run.fingerprint or f"unique:{run.rel}"
        groups.setdefault(key, []).append(run)

    primaries: list[RunAudit] = []
    for _key, members in groups.items():
        # Preference: more current figures > not archived > has a manifest >
        # deeper path > descriptive (longer) basename. "Deeper" keeps a run
        # nested in its dated suite (``.../20261030_0200_rl_suite/rl01_...``)
        # over a flat copy at the category root; the basename rule then keeps
        # ``20261102_0630_sprint_h2_...`` over the bare ``sprint_h2_...`` copy.
        winner = max(
            members,
            key=lambda r: (
                r.current_figure_count(),
                not r.archived,
                r.manifest is not None,
                len(r.path.parts),
                len(r.path.name),
                r.rel,
            ),
        )
        winner.copies = [m.rel for m in members if m is not winner]
        primaries.append(winner)

    primaries.sort(key=lambda r: (not r.is_complete, r.rel))
    return primaries


# --------------------------------------------------------------------------- #
# Gold-standard comparison
# --------------------------------------------------------------------------- #


def _gold() -> PipelineConfig:
    return PipelineConfig()


def _normalize_optimization(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        method = value.get("method")
        dielectric = value.get("dielectric")
        iterations = value.get("max_iterations")
    else:
        method = getattr(value, "method", None)
        dielectric = getattr(value, "dielectric", None)
        iterations = getattr(value, "max_iterations", None)
    if method is None:
        return None
    return f"{method} eps={float(dielectric):g} it={iterations}"


def _gold_value(field_name: str, cfg: PipelineConfig) -> Any:
    if field_name == "optimization":
        return _normalize_optimization(cfg.optimization)
    if field_name == "thresholds":
        return tuple(cfg.thresholds or ())
    return getattr(cfg, field_name, None)


def _run_value(field_name: str, config: dict[str, Any], cfg: PipelineConfig) -> Any:
    if field_name == "optimization":
        return _normalize_optimization(config.get("optimization"))
    if field_name == "thresholds":
        value = config.get("thresholds")
        return tuple(value) if isinstance(value, (list, tuple)) else value
    return config.get(field_name)


def compare(run: RunAudit, cfg: PipelineConfig) -> list[dict[str, Any]]:
    """Return one row per decision field: recorded, gold, and whether it matches."""
    rows: list[dict[str, Any]] = []
    for field_name, label in DECISION_FIELDS:
        recorded = _run_value(field_name, run.config, cfg)
        gold = _gold_value(field_name, cfg)
        present = field_name in run.config and recorded is not None
        if field_name == "rocs_clip_upper" and not run.config.get("rocs_use_modifier"):
            status = "inactive"
        elif not present:
            status = "unknown"
        elif recorded == gold:
            status = "match"
        else:
            status = "DIFFERS"
        rows.append(
            {
                "field": field_name,
                "label": label,
                "recorded": recorded,
                "gold": gold,
                "status": status,
            }
        )
    return rows


def headline_match(run: RunAudit, cfg: PipelineConfig) -> tuple[int, int]:
    """The three standards the user named: gmean, raw TanimotoCombo, 5k eval."""
    checks = [
        run.config.get("criteria") == cfg.criteria,
        run.config.get("score_variant") == cfg.score_variant
        and run.config.get("rocs_use_modifier") in (False, None),
        run.config.get("n_eval_samples") == cfg.n_eval_samples,
    ]
    return sum(1 for check in checks if check), len(checks)


# --------------------------------------------------------------------------- #
# Regeneration
# --------------------------------------------------------------------------- #


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the column spellings used by pre-2026-11-03 runs."""
    from rdkit import Chem

    out = frame.copy()
    # The supervisor notebook exported ``VALID`` (upper case), ``Raw_SA`` and a
    # raw [0, 2] ``ROCS`` column, but no ``Raw_ROCS``. Without this marker the
    # canonical unscale heuristic would double the score (max <= 1) and silently
    # turn a 0.94 hit into 1.87. Our own pre-2026-10-31 runs stored normalized
    # ROCS and *did* need the doubling, which is why the two cases are explicit.
    supervisor_schema = (
        "VALID" in frame.columns
        and "Raw_SA" in frame.columns
        and "Raw_ROCS" not in frame.columns
        and std.ROCS in frame.columns
    )
    if std.VALID not in out.columns:
        for column in list(out.columns):
            if str(column).strip().lower() == std.VALID.lower():
                out = out.rename(columns={column: std.VALID})
                break
    if supervisor_schema:
        out["Raw_ROCS"] = out[std.ROCS]
    if std.VALID in out.columns:
        out[std.VALID] = (
            pd.to_numeric(out[std.VALID], errors="coerce").fillna(0).astype(int)
        )
    if std.CANONICAL_SMILES not in out.columns and std.SMILES in out.columns:
        canonical: list[str] = []
        for smiles in out[std.SMILES]:
            mol = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
            canonical.append(Chem.MolToSmiles(mol) if mol is not None else "")
        out[std.CANONICAL_SMILES] = canonical
    return out


def _write_normalized_monitor_tsv(
    run: RunAudit, models: Path, target: Path
) -> list[str]:
    """Write a monitor TSV on the physical scale for the figure renderer.

    The production monitor records the *desirability* SA (clipped to [0, 1]) and
    older runs recorded ROCS divided by two. Figure 11 must show physical
    quantities, so the audit passes a normalized copy instead of changing the
    production renderer:

    * ``ROCS`` (or its historical aliases) is mapped to the raw [0, 2] scale;
    * ``SA`` is recomputed from the SMILES because the stored column is the
      clipped reward input, not synthetic accessibility.
    """
    from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
    from rdkit import Chem

    notes: list[str] = []
    smiles_path = models / "SequenceRNN_reinforced_smiles.tsv"
    frame = pd.read_csv(smiles_path, sep="\t")

    if std.ROCS not in frame.columns:
        for alias in ("Raw_ROCS", "RDKit_Supermol_TanimotoCombo", "TanimotoCombo"):
            if alias in frame.columns:
                frame[std.ROCS] = pd.to_numeric(frame[alias], errors="coerce")
                notes.append(f"ROCS<-{alias}")
                break

    # Pre-manifest 2026-10-26 runs stored the monitor ROCS divided by two.
    legacy_normalized = run.manifest is None and run.provenance is None
    if std.ROCS in frame.columns and legacy_normalized:
        frame[std.ROCS] = pd.to_numeric(frame[std.ROCS], errors="coerce") * 2.0
        notes.append("ROCS x2 (legacy normalized)")
    elif std.ROCS in frame.columns:
        notes.append("ROCS already raw")

    mols = [
        Chem.MolFromSmiles(str(value)) if pd.notna(value) and str(value).strip() else None
        for value in frame[std.SMILES]
    ]
    frame[std.SA] = [
        float(calculate_sa_score(mol)) if mol is not None else float("nan")
        for mol in mols
    ]
    notes.append("SA recomputed from SMILES")

    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "SequenceRNN_reinforced_smiles.tsv", sep="\t", index=False)
    shutil.copy2(
        models / "SequenceRNN_reinforced_fit.tsv",
        target / "SequenceRNN_reinforced_fit.tsv",
    )
    return notes


def regenerate(run: RunAudit) -> list[str]:
    """Rebuild every missing deliverable from the stored arms. Returns messages."""
    from rdkit import Chem

    from _david.code.visualization import (
        ResultAnalyzer,
        render_unfavorable_features_diagnostics,
    )

    messages: list[str] = []
    raw = run.path / "raw_data"
    models = run.path / "models"
    figures = run.figures
    figures.mkdir(parents=True, exist_ok=True)

    after = _normalize_frame(pd.read_csv(raw / "reinforced_raw.csv.gz"))
    before = _normalize_frame(pd.read_csv(raw / "baseline_raw.csv.gz"))

    paths = RunPaths(
        root=run.path,
        models_dir=models,
        raw_data_dir=raw,
        logs_dir=run.path / "logs",
    )
    with tempfile.TemporaryDirectory(prefix="rl-monitor-") as monitor_tmp:
        monitor_dir = Path(monitor_tmp)
        notes = _write_normalized_monitor_tsv(run, models, monitor_dir)
        ResultAnalyzer(paths).analyze(
            df_after=after,
            df_before=before,
            ref_smiles=run.query,
            monitor_prefix=str(monitor_dir / "SequenceRNN_reinforced"),
        )
    messages.append("analyze(): 01,02,03,05,06,07,10,11")
    messages.append("monitor TSV: " + "; ".join(notes))

    ref_mol = Chem.MolFromSmiles(run.query)
    ref_heavy = ref_mol.GetNumHeavyAtoms() if ref_mol is not None else 33
    render_unfavorable_features_diagnostics(
        before,
        after,
        figures / LEGACY_FIGURES[0],
        ref_heavy_atoms=ref_heavy,
    )
    messages.append(f"04 (legacy diagnostic, ref heavy atoms {ref_heavy})")

    return messages


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(item) for item in value) + ")"
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _short_rel(rel: str) -> str:
    return rel.replace("20261027_reinforcement_learning/", "")


def _verdict_mark(verdict: str) -> str:
    if verdict.startswith("VERIFIABLE (legacy"):
        return "OK*"
    if verdict == "VERIFIABLE":
        return "OK"
    if verdict == "INCOMPLETE":
        return "INCOMPLETE"
    return "UNVERIFIABLE"


def build_markdown(
    runs: list[RunAudit],
    regeneration: dict[str, list[str]],
    cfg: PipelineConfig,
) -> str:
    complete = [r for r in runs if r.is_complete and not r.is_smoke]
    incomplete = [r for r in runs if not r.is_complete and not r.is_smoke]
    smoke = [r for r in runs if r.is_smoke]

    lines: list[str] = []
    lines.append("# Production RL run audit")
    lines.append("")
    lines.append(
        "Generated by `python -m _david.code.audit_rl_runs` (add `--regenerate` to "
        "rebuild missing artifacts). One row per *distinct* run: byte-identical "
        "copies are folded into the `Copies` column."
    )
    lines.append("")
    lines.append("## Gold standard used for comparison")
    lines.append("")
    lines.append("| Knob | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Checkpoint criterion | `{cfg.criteria}` (geometric mean) |")
    lines.append(
        f"| Similarity | `{cfg.score_variant}`, raw `[0, 2]` "
        f"(`rocs_use_modifier={cfg.rocs_use_modifier}`), `use_colors=True` |"
    )
    lines.append(f"| Final evaluation | `n_eval_samples = {cfg.n_eval_samples}` |")
    lines.append("")
    lines.append(
        "Historical runs are **not** retroactively re-sampled: their figures are "
        "rebuilt from the stored arms (typically 1,000 rows). A new run must "
        "match the three rows above; a historical run can still be *verifiable* "
        "while differing from the gold standard."
    )
    lines.append("")

    if regeneration:
        lines.append("## Regeneration performed in this pass")
        lines.append("")
        lines.append("| Run | Rebuilt |")
        lines.append("| :--- | :--- |")
        for rel, messages in regeneration.items():
            lines.append(f"| `{_short_rel(rel)}` | {'; '.join(messages)} |")
        lines.append("")

    lines.append("## Complete runs")
    lines.append("")
    lines.append(
        "| Run | Host / job | Epochs | Criterion | Similarity | Eval n | QED | "
        "Reward | ε | K / C / min | Figures | Copies | Verdict |"
    )
    lines.append(
        "| :--- | :--- | ---: | :--- | :--- | ---: | :--- | :--- | :--- | :--- | "
        ":---: | :---: | :--- |"
    )
    for run in complete:
        cfg_map = run.config
        epochs = _fmt(cfg_map.get("epochs"))
        criterion = _fmt(cfg_map.get("criteria"))
        similarity = _fmt(cfg_map.get("score_variant"))
        if cfg_map.get("rocs_use_modifier"):
            similarity += " (clipped)"
        eval_n = _fmt(cfg_map.get("n_eval_samples"))
        qed = _fmt(cfg_map.get("include_qed"))
        reward = _fmt(cfg_map.get("reward_scheme"))
        epsilon = _fmt(cfg_map.get("epsilon"))
        if cfg_map.get("use_epsilon_decay"):
            epsilon += f" -> {_fmt(cfg_map.get('epsilon_end'))}"
        layout = (
            f"{_fmt(cfg_map.get('max_reference_conformers'))} / "
            f"{_fmt(cfg_map.get('max_conformers'))} / "
            f"{_fmt(cfg_map.get('minimize_candidates'))}"
        )
        figures = f"{run.current_figure_count()}/{len(CURRENT_FIGURES)}"
        if run.legacy_figure_count():
            figures += " (+04)"
        copies = len(run.copies)
        verdict = run.verdict()
        lines.append(
            f"| `{_short_rel(run.rel)}` | {run.host or '-'}"
            f"{(' / ' + run.pbs_job_id) if run.pbs_job_id else ''} | {epochs} | "
            f"`{criterion}` | `{similarity}` | {eval_n} | {qed} | `{reward}` | "
            f"{epsilon} | {layout} | {figures} | {copies} | {_verdict_mark(verdict)} |"
        )
    lines.append("")
    if any(r.verdict().startswith("VERIFIABLE (legacy") for r in complete):
        lines.append(
            f"`OK*` = all {len(CURRENT_FIGURES)} current figures plus the scorecard "
            "and summary are present; only the retired `04` diagnostic is missing. "
            "`04` is not part of the current `ResultAnalyzer` contract."
        )
        lines.append("")

    lines.append("## What differs from the gold standard")
    lines.append("")
    lines.append(
        "Only fields that actually differ are listed. `-` means the field was "
        "absent from the manifest (typically a pre-manifest run)."
    )
    lines.append("")
    for run in complete:
        deviations = [row for row in compare(run, cfg) if row["status"] == "DIFFERS"]
        headline, total = headline_match(run, cfg)
        lines.append(f"### `{_short_rel(run.rel)}`")
        lines.append("")
        lines.append(
            f"- Headline gold conformance: **{headline}/{total}** "
            f"(criterion, similarity, evaluation size)"
        )
        if run.source_revision:
            lines.append(f"- Source revision: `{run.source_revision}`")
        if run.elapsed_seconds:
            lines.append(f"- Wall clock: {run.elapsed_seconds / 3600.0:.1f} h")
        if not run.config:
            lines.append(
                "- Configuration unavailable: pre-manifest run. Only the stored "
                "artifacts can be audited; hyperparameters cannot be reconstructed "
                "from the repository."
            )
        elif not deviations:
            lines.append("- No deviations from the gold standard.")
        else:
            lines.append("")
            lines.append("| Knob | Run used | Gold |")
            lines.append("| :--- | :--- | :--- |")
            for row in deviations:
                lines.append(
                    f"| {row['label']} (`{row['field']}`) | "
                    f"`{_fmt(row['recorded'])}` | `{_fmt(row['gold'])}` |"
                )
        lines.append("")

    if incomplete:
        lines.append("## Incomplete runs (not verifiable, kept for provenance)")
        lines.append("")
        lines.append("| Run | State | Host / job | Last epoch | Raw arms |")
        lines.append("| :--- | :--- | :--- | ---: | :---: |")
        for run in incomplete:
            last_epoch = _last_epoch(run)
            lines.append(
                f"| `{_short_rel(run.rel)}` | {run.state} | "
                f"{run.host or '-'}"
                f"{(' / ' + run.pbs_job_id) if run.pbs_job_id else ''} | "
                f"{last_epoch} | {_fmt(run.raw_ready)} |"
            )
        lines.append("")

    if smoke:
        lines.append("## Excluded smoke/calibration runs")
        lines.append("")
        lines.append(
            "These share the RL manifest shape but train for "
            f"`epochs <= {SMOKE_EPOCH_LIMIT}`; they are engineering probes, not "
            "thesis runs."
        )
        lines.append("")
        lines.append("| Run | Epochs | Samples | Purpose |")
        lines.append("| :--- | ---: | ---: | :--- |")
        for run in smoke:
            purpose = "-"
            if run.manifest:
                purpose = str(run.manifest.get("purpose") or "-")
            lines.append(
                f"| `{_short_rel(run.rel)}` | {_fmt(run.config.get('epochs'))} | "
                f"{_fmt(run.config.get('n_samples'))} | {purpose} |"
            )
        lines.append("")

    duplicates = [run for run in runs if run.copies]
    if duplicates:
        lines.append("## Duplicate locations")
        lines.append("")
        lines.append(
            "Byte-identical `baseline_raw.csv.gz` + `reinforced_raw.csv.gz`. The "
            "primary is the copy with the most complete figure suite; the rest "
            "can be removed without losing evidence."
        )
        lines.append("")
        for run in duplicates:
            lines.append(f"- `{run.rel}`")
            for copy in run.copies:
                lines.append(f"  - copy: `{copy}`")
        lines.append("")

    lines.append("## How to reproduce this audit")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m _david.code.audit_rl_runs --regenerate")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _last_epoch(run: RunAudit) -> str:
    fit = run.path / "models" / "SequenceRNN_reinforced_fit.tsv"
    if not fit.is_file():
        return "-"
    try:
        frame = pd.read_csv(fit, sep="\t")
    except Exception:
        return "-"
    for column in ("Epoch", "epoch"):
        if column in frame.columns and not frame.empty:
            return str(frame[column].max())
    return "-"


def build_csv(runs: Iterable[RunAudit], cfg: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "run": run.rel,
                "state": run.state,
                "host": run.host,
                "pbs_job_id": run.pbs_job_id,
                "source_revision": run.source_revision,
                "elapsed_hours": (
                    round(run.elapsed_seconds / 3600.0, 3)
                    if run.elapsed_seconds
                    else None
                ),
                "query": run.query,
                "fingerprint": run.fingerprint,
                "copies": ";".join(run.copies),
                "current_figures": run.current_figure_count(),
                "legacy_figures": run.legacy_figure_count(),
                "verdict": run.verdict(),
                **{f"cfg_{key}": _fmt(value) for key, value in run.config.items()},
                **{
                    f"gold_{key}": _fmt(_gold_value(key, cfg))
                    for key, _ in DECISION_FIELDS
                },
                **{
                    f"diff_{row['field']}": row["status"]
                    for row in compare(run, cfg)
                },
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rebuild missing figures and scorecards before reporting",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"markdown destination (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV destination (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"run root to scan (default: {OUTPUT_ROOT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _gold()
    runs = discover(args.output_root)

    regeneration: dict[str, list[str]] = {}
    if args.regenerate:
        for run in runs:
            if run.is_smoke or not run.is_complete:
                continue
            if not run.can_regenerate:
                continue
            # Rebuild when any current figure, legacy diagnostic, support file
            # or the reference superposition is missing. Rebuilding everything
            # unconditionally would churn identical PNGs on every audit.
            missing = (
                run.current_figure_count() < len(CURRENT_FIGURES)
                or run.legacy_figure_count() < len(LEGACY_FIGURES)
                or not all(run.support_presence().values())
            )
            if not missing:
                continue
            try:
                regeneration[run.rel] = regenerate(run)
            except Exception as error:  # noqa: BLE001 - report, do not abort the audit
                regeneration[run.rel] = [f"FAILED: {type(error).__name__}: {error}"]

    report = build_markdown(runs, regeneration, cfg)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(report, encoding="utf-8")

    frame = build_csv(runs, cfg)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv, index=False)

    complete = [r for r in runs if r.is_complete and not r.is_smoke]
    verifiable = [r for r in complete if r.verdict().startswith("VERIFIABLE")]
    print(f"runs discovered: {len(runs)}")
    print(f"complete: {len(complete)}  verifiable: {len(verifiable)}")
    if regeneration:
        print(f"regenerated: {len(regeneration)}")
    print(f"report: {args.write}")
    print(f"csv:    {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
