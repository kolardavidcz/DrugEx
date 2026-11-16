"""Pipeline health audit and failure diagnosis: ``python -m _david.code.analysis.health``.

Inspects every run directory under ``_david/output/``, classifies each run by the exact
pipeline milestone it reached, diagnoses why failed/stalled runs broke from ``run.log`` and
``manifest.json``, optionally cross-checks live MetaCentrum queue/storage (``--remote``),
caches per-run diagnostics to ``<run>/cache_stats/health.json``, and writes
``health_report.md`` + ``health_report.tsv`` with ready-to-run remediation commands:

- ``no_model_saved``: no ``models/desired_ratio_at_stop.pkg`` (or empty); diagnoses failure
  cause and emits a ``submit submit`` re-run command.
- ``no_eval_dataset``: model checkpoint saved (or training finished), yet ``prior.tsv`` or
  ``final_desired_ratio_at_stop.tsv`` is missing or incomplete; emits a ``submit rebuild``
  (or checkpoint scoring) command.
- ``missing_figures_or_report``: evaluation TSVs complete, yet ``report.md`` is missing,
  fewer than 9 PNGs exist in ``figures/``, or a late analysis error occurred; repairable
  locally with ``--fix-reports``.
- ``healthy``: model, complete evaluation TSVs, all 9 figure slots, and ``report.md`` exist.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from _david.code.analysis._helpers import iter_run_dirs
from _david.code.analysis.report import (
    EXPECTED_FIGURE_SLOTS,
    analyse,
    parse_run_log_diagnostics,
)
from _david.code.config import PROJECT_ROOT, Config
from _david.code.run import arm_is_complete, recorded_arm_size

logger = logging.getLogger(__name__)

HEALTH_CACHE_VERSION = 1
_RUN_TAG_RE = re.compile(r"^\d{8}_\d{4}_(.+)$")
_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+):")


def _run_tag(run_name: str) -> str:
    match = _RUN_TAG_RE.match(run_name)
    return match.group(1) if match else run_name


def _health_sources(run_dir: Path) -> dict[str, Any]:
    """Fingerprints of files that determine a run's local health verdict."""
    files = (
        "manifest.json",
        "config.json",
        "submission.json",
        "prior.tsv",
        "final_desired_ratio_at_stop.tsv",
        "rl_fit.tsv",
        "report.md",
        "run.log",
        "models/prior.pkg",
        "models/desired_ratio.pkg",
        "models/desired_ratio_at_stop.pkg",
        "models/gmean.pkg",
    )
    out: dict[str, Any] = {}
    for rel in files:
        path = run_dir / rel
        if path.is_file():
            st = path.stat()
            out[rel] = [int(st.st_size), int(st.st_mtime_ns)]
        else:
            out[rel] = [-1, -1]
    fig_dir = run_dir / "figures"
    if fig_dir.is_dir():
        out["figures"] = sorted(
            [p.name, int(p.stat().st_size), int(p.stat().st_mtime_ns)]
            for p in fig_dir.glob("*.png")
        )
    else:
        out["figures"] = []
    return out


def _config_deltas(config: dict[str, Any]) -> list[str]:
    """Minimal ``name=value`` overrides reconstructing ``config`` relative to ``Config()``."""
    defaults = Config()
    deltas: list[str] = []
    skip = {"output_root", "n_jobs"}
    for field in dataclasses.fields(Config):
        if field.name in skip or field.name not in config:
            continue
        raw = config[field.name]
        default = getattr(defaults, field.name)
        if isinstance(default, tuple) and isinstance(raw, list):
            raw_cmp: Any = tuple(raw)
        else:
            raw_cmp = raw
        if raw_cmp == default or raw is None:
            continue
        if isinstance(raw_cmp, tuple):
            formatted = ",".join(str(x) for x in raw_cmp)
            deltas.append(f"{field.name}={formatted}")
        else:
            deltas.append(f"{field.name}={raw_cmp}")
    return deltas


def _rerun_command(tag: str, config: dict[str, Any]) -> str:
    deltas = " ".join(shlex.quote(d) for d in _config_deltas(config))
    base = f"python -m _david.code.metacentrum.submit submit --name {shlex.quote(tag)}"
    return f"{base} -- {deltas}" if deltas else base


def _parse_log_stage(run_dir: Path, log_diag: dict[str, Any]) -> dict[str, Any]:
    """Determine the deepest pipeline stage and likely root cause from ``run.log``."""
    log_path = run_dir / "run.log"
    if not log_path.is_file():
        return {
            "stage": "no_log",
            "last_epoch": None,
            "target_epochs": None,
            "training_summary_seen": False,
            "prior_written": (run_dir / "prior.tsv").is_file(),
            "stop_arm_started": False,
            "reason": "no_run_log",
            "detail": "run.log missing",
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    last_epoch = None
    target_epochs = None
    for match in _EPOCH_RE.finditer(text):
        last_epoch = int(match.group(1))
        target_epochs = int(match.group(2))
    if last_epoch is None:
        saved_m = re.findall(r"Model saved at epoch\s+(\d+)", text)
        if saved_m:
            last_epoch = int(saved_m[-1])
        elif (run_dir / "rl_fit.tsv").is_file():
            try:
                ep_rows = max(0, len((run_dir / "rl_fit.tsv").read_text(errors="replace").splitlines()) - 1)
                if ep_rows > 0:
                    last_epoch = ep_rows
            except OSError:
                pass

    has_summary = "training summary:" in text
    has_prior = "prior.tsv:" in text or (run_dir / "prior.tsv").is_file()
    has_done = any("done: " in line for line in lines[-10:])

    lowered = text.lower()
    reason = "unknown"
    detail = lines[-1].strip() if lines else ""
    elapsed_log_hours = None
    ts_matches = re.findall(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text, flags=re.M)
    if len(ts_matches) >= 2:
        try:
            t0 = datetime.strptime(ts_matches[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(ts_matches[-1], "%Y-%m-%d %H:%M:%S")
            elapsed_log_hours = (t1 - t0).total_seconds() / 3600.0
        except ValueError:
            pass

    if (
        "cannot start a worker" in lowered
        or "no worker can start" in lowered
        or "serial_run_refusal" in lowered
    ):
        reason = "forkserver_pool_failure"
    elif "out of memory" in lowered or "memoryerror" in lowered or "sigkill" in lowered:
        reason = "oom_or_sigkill"
    elif "cuda" in lowered and ("error" in lowered or "out of memory" in lowered):
        reason = "cuda_error"
    elif log_diag.get("analysis_failures") or log_diag.get("figure_failures"):
        reason = "analysis_or_figure_error"
    elif has_summary and not has_done:
        reason = "stopped_post_training"
    elif last_epoch is not None:
        if elapsed_log_hours is not None and elapsed_log_hours >= 23.0:
            reason = "walltime_during_training"
        else:
            reason = "stopped_mid_training"
    elif has_prior:
        reason = "stopped_before_epoch_1"
    else:
        reason = "stopped_in_setup_or_prior"

    if has_done:
        stage = "done"
    elif log_diag.get("analysis_failures"):
        stage = "analysis_failed"
    elif has_summary:
        stage = "post_training"
    elif last_epoch is not None:
        stage = f"epoch_{last_epoch}"
    elif has_prior:
        stage = "prior_calibrated"
    else:
        stage = "setup"

    return {
        "stage": stage,
        "last_epoch": last_epoch,
        "target_epochs": target_epochs,
        "training_summary_seen": has_summary,
        "prior_written": has_prior,
        "elapsed_log_hours": round(elapsed_log_hours, 2) if elapsed_log_hours is not None else None,
        "reason": reason,
        "detail": detail[:220],
    }


def _figure_status(run_dir: Path) -> dict[str, Any]:
    fig_dir = run_dir / "figures"
    files = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir.is_dir() else []
    missing_slots = [
        slot for slot in EXPECTED_FIGURE_SLOTS if not any(name.startswith(slot) for name in files)
    ]
    return {
        "count": len(files),
        "expected": len(EXPECTED_FIGURE_SLOTS),
        "files": files,
        "missing_slots": missing_slots,
    }


def inspect_run(run_dir: Path, *, use_cache: bool = True) -> dict[str, Any] | None:
    """Inspect one run directory locally and cache to ``cache_stats/health.json``."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    sources = _health_sources(run_dir)
    cache_path = run_dir / "cache_stats" / "health.json"
    if use_cache and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("version") == HEALTH_CACHE_VERSION
                and cached.get("sources") == sources
                and isinstance(cached.get("record"), dict)
            ):
                return cached["record"]
        except (OSError, json.JSONDecodeError):
            pass

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        record = {
            "run": run_dir.name,
            "path": str(run_dir),
            "tag": _run_tag(run_dir.name),
            "state": "corrupt_manifest",
            "category": "no_model_saved",
            "reason": "corrupt_manifest",
            "detail": str(error),
            "action": "rerun",
            "command": "",
            "epochs": 0,
            "hours": None,
            "has_stop_pkg": False,
            "has_prior_tsv": False,
            "has_stop_tsv": False,
            "figures_count": 0,
            "missing_figure_slots": list(EXPECTED_FIGURE_SLOTS),
            "has_report": False,
            "log_diagnostics": {},
        }
        return record

    config = manifest.get("config") or {}
    if not config and (run_dir / "config.json").is_file():
        try:
            config = json.loads((run_dir / "config.json").read_text())
        except (OSError, json.JSONDecodeError):
            config = {}

    state = str(manifest.get("state", "unknown"))
    n_expected = int(config.get("n_final") or config.get("n_eval_samples") or 5000)
    n_prior = int(config.get("n_prior") or 5000)
    stop_expected = recorded_arm_size(run_dir, "desired_ratio_at_stop") or n_expected
    prior_expected = recorded_arm_size(run_dir, "prior") or n_prior

    models_dir = run_dir / "models"
    stop_pkg = models_dir / "desired_ratio_at_stop.pkg"
    has_stop_pkg = stop_pkg.is_file() and stop_pkg.stat().st_size > 0
    pkgs = sorted(p.name for p in models_dir.glob("*.pkg")) if models_dir.is_dir() else []

    prior_tsv = run_dir / "prior.tsv"
    stop_tsv = run_dir / "final_desired_ratio_at_stop.tsv"
    has_prior_tsv = prior_tsv.is_file() and arm_is_complete(prior_tsv, prior_expected)
    has_stop_tsv = stop_tsv.is_file() and arm_is_complete(stop_tsv, stop_expected)

    fig_info = _figure_status(run_dir)
    report_path = run_dir / "report.md"
    has_report = report_path.is_file() and report_path.stat().st_size > 0

    rl_fit = run_dir / "rl_fit.tsv"
    epochs = 0
    if rl_fit.is_file():
        try:
            epochs = max(0, len(rl_fit.read_text(errors="replace").splitlines()) - 1)
        except OSError:
            epochs = 0
    summary = manifest.get("summary") or {}
    epochs = int(summary.get("epochs_run") or epochs)

    log_diag = parse_run_log_diagnostics(run_dir / "run.log")
    stage_info = _parse_log_stage(run_dir, log_diag)
    tag = _run_tag(run_dir.name)

    # Classify into the exact pipeline failure tiers.
    # Note: a finished run pulled from MetaCentrum might omit models/*.pkg locally if only
    # results were synced; if complete datasets & state=='complete' exist, model saving succeeded.
    if has_stop_tsv and has_prior_tsv:
        missing_figs = fig_info["count"] < len(EXPECTED_FIGURE_SLOTS) or bool(fig_info["missing_slots"])
        log_path = run_dir / "run.log"
        repaired_after_log = bool(
            has_report
            and not missing_figs
            and log_path.is_file()
            and report_path.stat().st_mtime_ns > log_path.stat().st_mtime_ns
        )
        historical_log_err = bool(
            (log_diag.get("analysis_failures") or log_diag.get("figure_failures"))
            and not repaired_after_log
        )
        late_bug = historical_log_err or state != "complete"
        if not has_report or missing_figs or late_bug:
            category = "missing_figures_or_report"
            problems = []
            if not has_report:
                problems.append("missing report.md")
            if fig_info["missing_slots"]:
                problems.append(f"missing figure slots {','.join(fig_info['missing_slots'])}")
            if historical_log_err and log_diag.get("figure_failures"):
                problems.append(f"figure errors in run.log ({','.join(log_diag['figure_failures'])})")
            if historical_log_err and log_diag.get("analysis_failures"):
                problems.append("analysis failed in run.log")
            if state != "complete":
                problems.append(f"manifest state={state}")
            reason = "; ".join(problems)
            action = "re_report"
            command = f"python -m _david.code.analysis {shlex.quote(str(run_dir))}"
        else:
            category = "healthy"
            reason = "ok"
            action = "none"
            command = ""
    elif has_stop_pkg or (stage_info["training_summary_seen"] and stop_expected > 0 and has_prior_tsv):
        # Model exists (or training finished cleanly before stop-arm scoring/sync).
        category = "no_eval_dataset"
        missing_arms = []
        if not has_prior_tsv:
            missing_arms.append("prior.tsv")
        if not has_stop_tsv:
            missing_arms.append("final_desired_ratio_at_stop.tsv")
        reason = (
            f"missing/incomplete {', '.join(missing_arms)} after stage '{stage_info['stage']}'"
            + ("" if has_stop_pkg else " (model checkpoint absent locally)")
        )
        if has_prior_tsv:
            action = "rebuild"
            command = f"python -m _david.code.metacentrum.submit rebuild --runs {shlex.quote(run_dir.name)}"
        else:
            action = "rerun"
            command = _rerun_command(tag, config)
    else:
        category = "no_model_saved"
        err = manifest.get("error") or stage_info["detail"] or stage_info["reason"]
        dur = (
            f", {stage_info['elapsed_log_hours']:.1f}h"
            if stage_info.get("elapsed_log_hours") is not None
            else ""
        )
        reason = f"{stage_info['reason']} (epoch {epochs}{dur}): {err}"
        action = "rerun"
        command = _rerun_command(tag, config)

    hours = (
        round(float(manifest["elapsed_seconds"]) / 3600, 2)
        if manifest.get("elapsed_seconds") is not None
        else stage_info.get("elapsed_log_hours")
    )
    record: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
        "tag": tag,
        "state": state,
        "category": category,
        "reason": reason,
        "detail": manifest.get("error") or stage_info["detail"],
        "stage": stage_info["stage"],
        "action": action,
        "command": command,
        "epochs": epochs,
        "hours": hours,
        "pkgs": pkgs,
        "has_stop_pkg": has_stop_pkg,
        "has_prior_tsv": has_prior_tsv,
        "has_stop_tsv": has_stop_tsv,
        "figures_count": fig_info["count"],
        "missing_figure_slots": fig_info["missing_slots"],
        "has_report": has_report,
        "log_diagnostics": log_diag,
    }
    if use_cache:
        try:
            cache_dir = run_dir / "cache_stats"
            cache_dir.mkdir(exist_ok=True)
            tmp_cache = cache_dir / f".health.{os.getpid()}.tmp"
            tmp_cache.write_text(
                json.dumps(
                    {
                        "version": HEALTH_CACHE_VERSION,
                        "sources": sources,
                        "record": record,
                    },
                    indent=2,
                )
            )
            os.replace(tmp_cache, cache_path)
        except OSError:
            pass
    return record


def _fetch_remote_state() -> dict[str, Any]:
    """Query active MetaCentrum jobs, rebuild specs, and remote output checkpoints."""
    from _david.code.metacentrum.submit import STORAGE, list_jobs, remote

    jobs = list_jobs()
    active_by_tag: dict[str, list[str]] = {}
    for job in jobs:
        active_by_tag.setdefault(job.name, []).append(f"{job.short_id}({job.state}:{job.queue})")

    active_by_run: dict[str, list[str]] = {}
    if jobs:
        spec_script = (
            f"cd {shlex.quote(STORAGE + '/jobs')} 2>/dev/null || exit 0\n"
            "for j in "
            + " ".join(shlex.quote(j.name) for j in jobs)
            + '; do [ -f "$j/spec.json" ] && echo "### $j" && cat "$j/spec.json" && echo ""; done'
        )
        raw_specs = remote(spec_script)
        cur_name = None
        buf: list[str] = []
        for line in raw_specs.splitlines():
            if line.startswith("### "):
                if cur_name is not None:
                    _collect_rebuild_runs(cur_name, "\n".join(buf), active_by_tag, active_by_run)
                cur_name = line[4:].strip()
                buf = []
            elif cur_name is not None:
                buf.append(line)
        if cur_name is not None:
            _collect_rebuild_runs(cur_name, "\n".join(buf), active_by_tag, active_by_run)

    remote_runs: dict[str, dict[str, Any]] = {}
    probe = (
        f"cd {shlex.quote(STORAGE + '/_david/output')} 2>/dev/null || exit 0\n"
        'for d in */; do d=${d%/}; [ -d "$d" ] || continue; '
        'sp=$([ -s "$d/models/desired_ratio_at_stop.pkg" ] && echo 1 || echo 0); '
        'st=$(awk "END{exit !(NR>1)}" "$d/final_desired_ratio_at_stop.tsv" 2>/dev/null && echo 1 || echo 0); '
        'rp=$([ -s "$d/report.md" ] && echo 1 || echo 0); '
        'echo "$d|$sp|$st|$rp"; done'
    )
    for line in remote(probe).splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4:
            rname, sp, st, rp = parts
            remote_runs[rname] = {
                "remote_stop_pkg": sp == "1",
                "remote_stop_tsv": st == "1",
                "remote_report": rp == "1",
            }
    return {
        "active_by_tag": active_by_tag,
        "active_by_run": active_by_run,
        "remote_runs": remote_runs,
    }


def _collect_rebuild_runs(
    job_name: str,
    raw_json: str,
    active_by_tag: dict[str, list[str]],
    active_by_run: dict[str, list[str]],
) -> None:
    try:
        spec = json.loads(raw_json)
    except json.JSONDecodeError:
        return
    if spec.get("mode") == "rebuild":
        labels = active_by_tag.get(job_name, [job_name])
        for run_name in spec.get("runs") or []:
            active_by_run.setdefault(run_name, []).extend(f"{job_name}:{lbl}" for lbl in labels)


def _run_date(name: str) -> str | None:
    match = re.match(r"^(\d{8})_\d{4}_", name)
    return match.group(1) if match else None


def check_health(
    root: Path,
    out_dir: Path | None = None,
    *,
    since: str | None = None,
    remote_check: bool = False,
    fix_reports: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Audit every run under ``root``, optionally repair reports, and write MD/TSV reports."""
    runs = iter_run_dirs(root)
    if since:
        runs = [r for r in runs if (_run_date(r.name) or "") >= since]

    records: list[dict[str, Any]] = []
    fixed: list[str] = []
    for run_dir in runs:
        rec = inspect_run(run_dir, use_cache=use_cache)
        if rec is None:
            continue
        if fix_reports and rec["category"] == "missing_figures_or_report":
            try:
                analyse(run_dir)
                fixed.append(run_dir.name)
                refreshed = inspect_run(run_dir, use_cache=False)
                if refreshed is not None:
                    rec = refreshed
            except Exception as error:
                logger.exception("auto-repair failed for %s: %s", run_dir.name, error)
        records.append(rec)

    remote_info = _fetch_remote_state() if remote_check else None
    if remote_info:
        active_by_tag = remote_info["active_by_tag"]
        active_by_run = remote_info["active_by_run"]
        remote_runs = remote_info["remote_runs"]
        for rec in records:
            live = list(active_by_run.get(rec["run"], [])) + list(active_by_tag.get(rec["tag"], []))
            r_meta = remote_runs.get(rec["run"], {})
            rec["live_jobs"] = sorted(set(live))
            rec["remote_stop_pkg"] = r_meta.get("remote_stop_pkg")
            rec["remote_stop_tsv"] = r_meta.get("remote_stop_tsv")
            # If the model exists remotely and this run has no local model yet, it is
            # rebuildable from MetaCentrum storage rather than requiring a full retrain.
            if (
                rec["category"] == "no_model_saved"
                and r_meta.get("remote_stop_pkg")
                and not rec["has_stop_pkg"]
            ):
                rec["category"] = "no_eval_dataset"
                rec["action"] = "rebuild"
                rec["reason"] = "stop model exists on MetaCentrum storage; missing stop arm locally"
                rec["command"] = (
                    f"python -m _david.code.metacentrum.submit rebuild --runs {shlex.quote(rec['run'])}"
                )
    else:
        for rec in records:
            rec["live_jobs"] = []

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["category"]] = counts.get(rec["category"], 0) + 1

    target = out_dir or root
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(" ", "seconds"),
        "root": str(root),
        "total": len(records),
        "counts": counts,
        "fixed_reports": fixed,
        "remote_checked": remote_check,
        "runs": records,
    }
    _write_tsv(records, target / "health_report.tsv")
    _write_markdown(payload, target / "health_report.md")
    return payload


def _write_tsv(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for r in records:
        diag = r.get("log_diagnostics") or {}
        rows.append(
            {
                "run": r["run"],
                "tag": r["tag"],
                "category": r["category"],
                "state": r["state"],
                "stage": r.get("stage"),
                "epochs": r["epochs"],
                "hours": r["hours"],
                "has_stop_pkg": r["has_stop_pkg"],
                "has_prior_tsv": r["has_prior_tsv"],
                "has_stop_tsv": r["has_stop_tsv"],
                "figures_count": r["figures_count"],
                "missing_figure_slots": ",".join(r.get("missing_figure_slots") or []),
                "has_report": r["has_report"],
                "scoring_failures": diag.get("scoring_failures"),
                "conformers_dropped": diag.get("conformers_dropped"),
                "zero_conformer_molecules": diag.get("zero_conformer_molecules"),
                "live_jobs": ",".join(r.get("live_jobs") or []),
                "action": r["action"],
                "reason": r["reason"],
                "command": r["command"],
                "path": r["path"],
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    counts = payload["counts"]
    runs = payload["runs"]
    lines = [
        "# DrugEx Run Health Report",
        "",
        f"Generated: `{payload['generated']}` · Root: `{payload['root']}` · Total runs: **{payload['total']}**",
        "",
        "## Summary",
        "",
        "| category | count | meaning |",
        "|---|---:|---|",
        f"| `healthy` | {counts.get('healthy', 0)} | model + complete TSVs + 9 figures + report.md |",
        f"| `no_model_saved` | {counts.get('no_model_saved', 0)} | missing `desired_ratio_at_stop.pkg` (needs diagnosis / re-run) |",
        f"| `no_eval_dataset` | {counts.get('no_eval_dataset', 0)} | model saved or training finished, missing/incomplete evaluation TSVs (rebuild) |",
        f"| `missing_figures_or_report` | {counts.get('missing_figures_or_report', 0)} | datasets complete, missing/partial PNGs or `report.md` or late error |",
        "",
    ]
    if payload["fixed_reports"]:
        lines += [
            f"Auto-repaired reports/figures in this pass ({len(payload['fixed_reports'])}): "
            + ", ".join(f"`{n}`" for n in payload["fixed_reports"]),
            "",
        ]

    for category, heading in (
        ("no_model_saved", "No Model Saved (Failed / Unfinished Training)"),
        ("no_eval_dataset", "No Evaluation Dataset (Model Saved, Needs Rebuild)"),
        ("missing_figures_or_report", "Missing or Partial Figures / Report"),
    ):
        bucket = [r for r in runs if r["category"] == category]
        lines += [f"## {heading} ({len(bucket)})", ""]
        if not bucket:
            lines += ["_none_", ""]
            continue
        lines += [
            "| run | state | ep | figs | live job | diagnosis |",
            "|---|---|---:|---:|---|---|",
        ]
        for r in bucket:
            live = ", ".join(r.get("live_jobs") or []) or "—"
            reason = str(r["reason"]).replace("|", "/")
            lines.append(
                f"| `{r['run']}` | {r['state']} | {r['epochs']} | {r['figures_count']}/9 | {live} | {reason} |"
            )
        lines.append("")

        need_action = [r for r in bucket if not r.get("live_jobs") and r.get("command")]
        if need_action:
            lines += ["### Ready Remediation Commands (not currently active in PBS)", "", "```bash"]
            for r in need_action:
                lines.append(f"# {r['run']}: {r['reason']}")
                lines.append(r["command"])
            lines += ["```", ""]

    # Highlight runs with notable relaxation drops or scoring timeouts even when healthy
    noisy = [
        r
        for r in runs
        if (r.get("log_diagnostics") or {}).get("scoring_failures", 0) > 0
        or (r.get("log_diagnostics") or {}).get("zero_conformer_molecules", 0) >= 10
    ]
    if noisy:
        lines += [
            f"## Runs With Scoring Timeouts or High Zero-Conformer Drop Counts ({len(noisy)})",
            "",
            "| run | scoring_timeouts | zero_conf_mols | dropped/total |",
            "|---|---:|---:|---:|",
        ]
        for r in sorted(
            noisy,
            key=lambda x: (
                (x.get("log_diagnostics") or {}).get("scoring_failures", 0),
                (x.get("log_diagnostics") or {}).get("zero_conformer_molecules", 0),
            ),
            reverse=True,
        )[:25]:
            d = r.get("log_diagnostics") or {}
            lines.append(
                f"| `{r['run']}` | {d.get('scoring_failures', 0)} | "
                f"{d.get('zero_conformer_molecules', 0)} | "
                f"{d.get('conformers_dropped', 0)}/{d.get('conformers_total', 0)} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "_david" / "output")
    parser.add_argument("--out", type=Path, default=None, help="directory for health_report.{md,tsv}")
    parser.add_argument("--since", default=None, metavar="YYYYMMDD")
    parser.add_argument("--remote", action="store_true", help="cross-check MetaCentrum queue and storage")
    parser.add_argument(
        "--fix-reports",
        action="store_true",
        help="locally re-run analysis/figures for runs missing figures or report.md",
    )
    parser.add_argument("--no-cache", action="store_true", help="bypass cache_stats/health.json")
    parser.add_argument("--json", action="store_true", help="print JSON summary to stdout")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payload = check_health(
        args.root,
        args.out,
        since=args.since,
        remote_check=args.remote,
        fix_reports=args.fix_reports,
        use_cache=not args.no_cache,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        c = payload["counts"]
        print(
            f"health audit ({payload['total']} runs): "
            f"healthy={c.get('healthy', 0)}, "
            f"no_model_saved={c.get('no_model_saved', 0)}, "
            f"no_eval_dataset={c.get('no_eval_dataset', 0)}, "
            f"missing_figures_or_report={c.get('missing_figures_or_report', 0)}"
        )
        out_dir = args.out or args.root
        print(f"report: {out_dir / 'health_report.md'}")
        print(f"table:  {out_dir / 'health_report.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
