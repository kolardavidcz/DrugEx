"""Audit and correct the ROCS scale of stored evaluation arms.

The ROCS score moved through three regimes, and a stored ``ROCS`` column means
different things in each:

``raw``
    Raw TanimotoCombo in [0, 2]. The scorer had no modifier (the default since
    commit ``986d63a``), so the arm stores the physical score.
``normalized``
    ``min(raw, clip) / clip`` in [0, 1]. The scorer had a ``ClippedScore``
    modifier. Downstream code converted it back by multiplying with ``clip``;
    ``Raw_ROCS`` records that conversion when the pipeline wrote it.
``unknown``
    Neither fingerprint matches. The run is excluded from ROCS rankings instead
    of being silently trusted.

The defect this module exists for: before commit ``9aa5d88`` the clip was
``1.8``, but the visualization layer reconstructed the physical score by
multiplying any [0, 1]-looking arm by **2.0** — the ratio of the full
TanimotoCombo range, not of the configured clip. The recovered value is then
``2.0 / 1.8 = 1.111`` times too large, so a molecule at raw 1.81 is drawn at
2.01 and everything at or above the clip saturates at 2.0.

Detection never guesses from a directory name. It reads the evidence:

* ``Raw_ROCS`` present and equal to ``ROCS`` -> raw arm;
* ``Raw_ROCS`` present and equal to ``clip * ROCS`` -> normalized arm, and the
  clip is recoverable from the ratio itself;
* no ``Raw_ROCS``: the arm's value range plus the code era recorded in the
  manifest config (or the file's timestamp for runs that predate manifests).

The era rule is deliberately blunt and documented: the config field
``rocs_use_modifier`` only exists from ``986d63a`` onward, so its absence means
the modifier was always applied; the clip default was 1.8 before ``9aa5d88`` and
2.0 after it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _david.code.config import OUTPUT_ROOT
from _david.code.schema import std

LOGGER = logging.getLogger(__name__)

SCALE_NAME: str = "rocs_scale.json"
REPORT_NAME: str = "ROCS_SCALE_AUDIT.md"
ARM_NAME: str = "reinforced_raw.csv.gz"

RAW: str = "raw"
NORMALIZED: str = "normalized"
UNKNOWN: str = "unknown"
VERDICTS: tuple[str, ...] = (RAW, NORMALIZED, UNKNOWN)

#: TanimotoCombo cannot exceed 2.0; a normalized arm can never store more than 1.
COMBO_MAX: float = 2.0
NORMALIZED_MAX: float = 1.0 + 1e-6

#: Commit dates that changed the default scoring regime. Used only when a run has
#: no manifest config to read; a run with a config is classified from the config.
_MODIFIER_INTRODUCED = datetime(2026, 10, 31, 4, 12, tzinfo=timezone.utc)
_CLIP_WIDENED = datetime(2026, 11, 4, 4, 41, tzinfo=timezone.utc)
LEGACY_CLIP: float = 1.8
CURRENT_CLIP: float = 2.0

#: How closely a stored ratio must match a candidate factor to be accepted. The
#: arm-side check is exact (the pipeline multiplied by the clip itself), so it is
#: tight. A fresh rescoring re-embeds conformers and can drift by a few percent,
#: so the empirical tie-breaker is deliberately looser.
_FACTOR_TOLERANCE: float = 0.02
_RESCORE_TOLERANCE: float = 0.10


@dataclass
class ScaleVerdict:
    """What the evidence says about one stored arm."""

    run: str
    verdict: str
    clip: float | None
    factor: float
    n_rows: int = 0
    n_saturated: int = 0
    ratio_median: float | None = None
    ratio_max: float | None = None
    evidence: str = ""
    config_modifier: bool | None = None
    config_clip: float | None = None

    @property
    def is_raw(self) -> bool:
        return self.verdict == RAW

    @property
    def is_inflated(self) -> bool:
        """True when a [0,1] arm was being read with the wrong (2.0) factor."""
        return self.verdict == NORMALIZED and self.clip is not None and self.clip < COMBO_MAX

    @property
    def saturated_fraction(self) -> float:
        return (self.n_saturated / self.n_rows) if self.n_rows else 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "verdict": self.verdict,
            "clip": self.clip,
            "factor": round(self.factor, 6),
            "n_rows": self.n_rows,
            "n_saturated": self.n_saturated,
            "saturated_fraction": round(self.saturated_fraction, 6),
            "ratio_median": self.ratio_median,
            "ratio_max": self.ratio_max,
            "config_modifier": self.config_modifier,
            "config_clip": self.config_clip,
            "evidence": self.evidence,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> ScaleVerdict:
        return cls(
            run=str(data.get("run", "")),
            verdict=str(data.get("verdict", UNKNOWN)),
            clip=data.get("clip"),
            factor=float(data.get("factor", 1.0)),
            n_rows=int(data.get("n_rows", 0)),
            n_saturated=int(data.get("n_saturated", 0)),
            ratio_median=data.get("ratio_median"),
            ratio_max=data.get("ratio_max"),
            evidence=str(data.get("evidence", "")),
            config_modifier=data.get("config_modifier"),
            config_clip=data.get("config_clip"),
        )


def arm_path(run: Path | str) -> Path:
    return Path(run) / "raw_data" / ARM_NAME


def read_config(run: Path | str) -> dict[str, Any]:
    manifest = Path(run) / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    config = payload.get("config")
    return config if isinstance(config, dict) else {}


def era_scoring(
    config: dict[str, Any] | None, started: datetime | None = None
) -> tuple[bool, float]:
    """Return ``(modifier_applied, clip)`` for the code that scored the arm."""
    config = config or {}
    if "rocs_use_modifier" in config:
        clip = config.get("rocs_clip_upper")
        return bool(config["rocs_use_modifier"]), float(clip or CURRENT_CLIP)
    if config:
        # The field did not exist yet, so the modifier was unconditional.
        clip = config.get("rocs_clip_upper")
        return True, float(clip or LEGACY_CLIP)
    if started is not None and started < _MODIFIER_INTRODUCED:
        return True, LEGACY_CLIP
    if started is not None and started < _CLIP_WIDENED:
        return False, LEGACY_CLIP
    return False, CURRENT_CLIP


def _started_at(run: Path) -> datetime | None:
    """When the run started, from its ``YYYYmmdd_HHMM`` name.

    The directory mtime is useless here: runs are copied between hosts and
    renamed by the layout normalizer, so it records the move, not the experiment.
    """
    from _david.code.dashboard import started_at

    stamped = started_at(run.name)
    if stamped is not None:
        return stamped.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(run.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _ratio(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """``ROCS / Raw_ROCS`` over rows where both are positive."""
    if "Raw_ROCS" not in frame.columns or std.ROCS not in frame.columns:
        return None, None
    raw = pd.to_numeric(frame["Raw_ROCS"], errors="coerce").to_numpy(dtype=float)
    stored = pd.to_numeric(frame[std.ROCS], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(raw) & np.isfinite(stored) & (raw > 1e-6)
    if not mask.any():
        return None, None
    ratio = stored[mask] / raw[mask]
    return float(np.median(ratio)), float(np.max(np.abs(ratio)))


def classify(frame: pd.DataFrame, config: dict[str, Any] | None, started: datetime | None = None) -> ScaleVerdict:
    """Decide how a stored arm's ``ROCS`` column relates to raw TanimotoCombo."""
    modifier, clip = era_scoring(config, started)
    n_rows = int(len(frame))
    stored = pd.to_numeric(frame.get(std.ROCS, pd.Series(dtype=float)), errors="coerce")
    stored_max = float(np.nanmax(stored)) if stored.notna().any() else 0.0

    if "Raw_ROCS" in frame.columns:
        ratio_median, ratio_max = _ratio(frame)
        if ratio_median is None:
            # A collapsed arm can be all zeros, so no ratio exists. If the two
            # columns still agree row by row, the arm is raw rather than unknown.
            raw_values = pd.to_numeric(frame["Raw_ROCS"], errors="coerce")
            same = raw_values.fillna(-1.0).round(6).equals(
                stored.fillna(-1.0).round(6)
            )
            return ScaleVerdict(
                run="",
                verdict=RAW if same else UNKNOWN,
                clip=None,
                factor=1.0,
                n_rows=n_rows,
                evidence=(
                    "Raw_ROCS equals ROCS (all values zero), so the arm stores raw "
                    "TanimotoCombo"
                    if same
                    else "Raw_ROCS present but carries no positive value and "
                    "disagrees with ROCS"
                ),
                config_modifier=modifier,
                config_clip=clip,
            )
        if abs(ratio_median - 1.0) <= _FACTOR_TOLERANCE:
            return ScaleVerdict(
                run="",
                verdict=RAW,
                clip=None,
                factor=1.0,
                n_rows=n_rows,
                ratio_median=ratio_median,
                ratio_max=ratio_max,
                evidence="Raw_ROCS equals ROCS, so the arm stores raw TanimotoCombo",
                config_modifier=modifier,
                config_clip=clip,
            )
        implied = 1.0 / ratio_median if ratio_median > 0 else None
        if implied is not None and abs(implied - clip) / clip <= _FACTOR_TOLERANCE:
            raw = pd.to_numeric(frame["Raw_ROCS"], errors="coerce").to_numpy(dtype=float)
            saturated = int(np.count_nonzero(np.isfinite(raw) & (raw >= clip - 1e-6)))
            return ScaleVerdict(
                run="",
                verdict=NORMALIZED,
                clip=clip,
                factor=clip,
                n_rows=n_rows,
                n_saturated=saturated,
                ratio_median=ratio_median,
                ratio_max=ratio_max,
                evidence=(
                    f"Raw_ROCS = {implied:.3f} x ROCS, matching the configured clip {clip}"
                ),
                config_modifier=modifier,
                config_clip=clip,
            )
        return ScaleVerdict(
            run="",
            verdict=UNKNOWN,
            clip=None,
            factor=1.0,
            n_rows=n_rows,
            ratio_median=ratio_median,
            ratio_max=ratio_max,
            evidence=(
                f"Raw_ROCS / ROCS is neither 1 nor the clip {clip} "
                f"(median ratio {ratio_median:.3f})"
            ),
            config_modifier=modifier,
            config_clip=clip,
        )

    # No Raw_ROCS: decide from the value range the scoring regime produced.
    if stored_max <= NORMALIZED_MAX and modifier:
        saturated = int(np.count_nonzero(stored.to_numpy(dtype=float) >= 1.0 - 1e-9))
        return ScaleVerdict(
            run="",
            verdict=NORMALIZED,
            clip=clip,
            factor=clip,
            n_rows=n_rows,
            n_saturated=saturated,
            evidence=(
                f"no Raw_ROCS, maximum ROCS {stored_max:.3f} <= 1 and the scoring "
                f"era applied the ClippedScore modifier (clip {clip})"
            ),
            config_modifier=modifier,
            config_clip=clip,
        )
    if stored_max > NORMALIZED_MAX:
        return ScaleVerdict(
            run="",
            verdict=RAW,
            clip=None,
            factor=1.0,
            n_rows=n_rows,
            evidence=(
                f"no Raw_ROCS and maximum ROCS {stored_max:.3f} > 1, so the arm "
                "cannot be on the normalized [0, 1] scale"
            ),
            config_modifier=modifier,
            config_clip=clip,
        )
    return ScaleVerdict(
        run="",
        verdict=UNKNOWN,
        clip=clip,
        factor=1.0,
        n_rows=n_rows,
        evidence=(
            f"no Raw_ROCS, maximum ROCS {stored_max:.3f} <= 1 and the scoring era "
            f"kept raw scores (clip {clip}), so the scale cannot be decided from values"
        ),
        config_modifier=modifier,
        config_clip=clip,
    )


def inspect(run: Path | str) -> ScaleVerdict:
    """Classify one run directory from its stored arm and config."""
    run = Path(run)
    arm = arm_path(run)
    if not arm.is_file():
        return ScaleVerdict(
            run=run.name, verdict=UNKNOWN, clip=None, factor=1.0,
            evidence="no reinforced arm to inspect",
        )
    frame = pd.read_csv(arm)
    verdict = classify(frame, read_config(run), _started_at(run))
    verdict.run = run.name
    return verdict


def correct_frame(frame: pd.DataFrame, verdict: ScaleVerdict) -> pd.DataFrame:
    """Return the frame with ``ROCS`` on the raw [0, 2] scale.

    ``Raw_ROCS`` is attached when it is missing and ``ROCS_Saturated`` marks the
    molecules that hit the clip, so a best-of table can prefer an unsaturated
    molecule at an equal recorded score.
    """
    out = frame.copy()
    if out.empty or std.ROCS not in out.columns:
        return out
    stored = pd.to_numeric(out[std.ROCS], errors="coerce")
    if "Raw_ROCS" in out.columns:
        raw = pd.to_numeric(out["Raw_ROCS"], errors="coerce")
        saturated = (
            raw.notna() & (raw >= (verdict.clip - 1e-6))
            if verdict.verdict == NORMALIZED and verdict.clip
            else pd.Series(False, index=out.index)
        )
    elif verdict.verdict == NORMALIZED:
        raw = stored * verdict.factor
        saturated = stored >= 1.0 - 1e-9
    else:
        raw = stored.copy()
        saturated = pd.Series(False, index=out.index)
    out["Raw_ROCS"] = raw
    out[std.ROCS] = raw
    out["ROCS_Saturated"] = saturated.astype(int)
    return out


def scale_file(run: Path | str) -> Path:
    return Path(run) / "raw_data" / SCALE_NAME


def read_verdict(run: Path | str) -> ScaleVerdict | None:
    path = scale_file(run)
    if not path.is_file():
        return None
    try:
        return ScaleVerdict.from_payload(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None


def ensure_verdict(
    run: Path | str, *, refresh: bool = False, settle_unknown: bool = False
) -> ScaleVerdict:
    """Cached verdict for a run; writes ``raw_data/rocs_scale.json`` on a miss."""
    run = Path(run)
    if not refresh:
        cached = read_verdict(run)
        if cached is not None:
            return cached
    verdict = inspect(run)
    if verdict.verdict == UNKNOWN and settle_unknown:
        verdict = settle(run)
    path = scale_file(run)
    if path.parent.is_dir():
        path.write_text(json.dumps(verdict.payload(), indent=2) + "\n")
    return verdict


def correct_run_frame(run: Path | str, frame: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    """Correct an arm loaded from ``run`` before it reaches any figure."""
    return correct_frame(frame, ensure_verdict(run, refresh=refresh))


#: Conformer knobs that change the geometry and therefore the ROCS value itself.
#: They are copied from the run's config when present; the reward knobs are not,
#: because the rescoring is always raw.
_GEOMETRY_KEYS: tuple[str, ...] = (
    "max_conformers",
    "max_isomers",
    "max_heavy_atoms",
    "max_rotatable_bonds",
    "minimize_candidates",
    "reference_attempts",
    "reference_rmsd_cutoff",
    "max_reference_conformers",
    "reference_selection_mode",
    "energy_window_kcal",
    "timeout_seconds",
    "conformer_threads",
)


def sample_rows(frame: pd.DataFrame, sample: int = 10) -> pd.DataFrame:
    """Pick molecules spread over the recorded ROCS range, for a scale check."""
    valid = frame[frame[std.VALID] == 1] if std.VALID in frame.columns else frame
    if valid.empty:
        valid = frame
    stored = pd.to_numeric(valid[std.ROCS], errors="coerce")
    ordered = stored.sort_values(ascending=False, na_position="last")
    half = max(1, sample // 2)
    top = list(ordered.index[:half])
    middle = list(ordered.index[len(ordered) // 2 : len(ordered) // 2 + (sample - half)])
    picked = list(dict.fromkeys(top + middle))
    return valid.loc[picked]


def raw_scores(smiles: list[str], reference: Path, config: dict[str, Any] | None) -> np.ndarray:
    """Score ``smiles`` with the raw TanimotoCombo scorer, never a modifier."""
    from _david.code.config import PipelineConfig
    from _david.code.environment import create_environment

    # PipelineConfig is frozen, so a value is applied through object.__setattr__;
    # the geometry knobs only steer the conformer search, which does change the
    # score slightly, but never its scale.
    cfg = PipelineConfig()
    for key in _GEOMETRY_KEYS:
        value = (config or {}).get(key)
        if value is None or value in ("auto",):
            continue
        try:
            object.__setattr__(cfg, key, value)
        except (AttributeError, TypeError, ValueError):
            LOGGER.warning("ignoring geometry knob %s=%r", key, value)
    object.__setattr__(cfg, "rocs_use_modifier", False)
    env = create_environment(reference, config=cfg)
    scores = env.getScores(smiles)
    return np.asarray(scores[std.ROCS], dtype=float)


def settle(run: Path | str, *, sample: int = 10) -> ScaleVerdict:
    """Decide an ambiguous arm by rescoring a few of its molecules from scratch.

    This is the tie-breaker for arms that predate both the manifest config and
    ``Raw_ROCS``: a handful of molecules are re-embedded and rescored with the raw
    scorer, and the stored value is divided by the fresh one. A ratio near 1 means
    the arm is raw; a ratio near ``1 / clip`` means it was normalized with that clip.
    """
    run = Path(run)
    verdict = inspect(run)
    if verdict.verdict != UNKNOWN:
        return verdict
    arm = arm_path(run)
    reference = run / "raw_data" / "query_molecule_3d.sdf"
    if not arm.is_file() or not reference.is_file():
        return verdict
    frame = pd.read_csv(arm)
    picked = sample_rows(frame, sample)
    smiles = picked[std.SMILES].astype(str).tolist()
    if not smiles:
        return verdict
    LOGGER.info("%s: rescoring %d molecules to settle the ROCS scale", run.name, len(smiles))
    fresh = raw_scores(smiles, reference, read_config(run))
    stored = pd.to_numeric(picked[std.ROCS], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(fresh) & np.isfinite(stored) & (fresh > 1e-6)
    if not mask.any():
        verdict.evidence = "rescoring produced no usable pair of values"
        return verdict
    ratio = float(np.median(stored[mask] / fresh[mask]))
    verdict.ratio_median = ratio
    if abs(ratio - 1.0) <= _RESCORE_TOLERANCE:
        verdict.verdict = RAW
        verdict.clip = None
        verdict.factor = 1.0
        verdict.n_saturated = 0
        verdict.evidence = (
            f"rescored {int(mask.sum())} molecules; stored ROCS / fresh raw ROCS = "
            f"{ratio:.3f}"
        )
        return verdict
    if ratio > 0:
        implied = 1.0 / ratio
        plausible = [clip for clip in (LEGACY_CLIP, CURRENT_CLIP) if abs(implied - clip) / clip <= _RESCORE_TOLERANCE]
        if not plausible:
            verdict.evidence = (
                f"rescored {int(mask.sum())} molecules; stored ROCS / fresh raw ROCS = "
                f"{ratio:.3f} implies a clip of {implied:.2f}, which matches neither "
                f"{LEGACY_CLIP} nor {CURRENT_CLIP}"
            )
            return verdict
        verdict.verdict = NORMALIZED
        verdict.clip = float(min(plausible, key=lambda clip: abs(implied - clip)))
        verdict.factor = verdict.clip
        verdict.n_saturated = int(np.count_nonzero(stored[mask] >= 1.0 - 1e-9))
        verdict.evidence = (
            f"rescored {int(mask.sum())} molecules; stored ROCS / fresh raw ROCS = "
            f"{ratio:.3f}, implying a clip of {implied:.2f}"
        )
        return verdict
    verdict.evidence = f"rescoring gave an unusable ratio {ratio:.3f}"
    return verdict


def audit(
    root: Path | str = OUTPUT_ROOT,
    *,
    refresh: bool = False,
    settle_unknown: bool = False,
) -> list[ScaleVerdict]:
    """Classify every RL arm under ``root`` (archive included, for the record)."""
    from _david.code.dashboard import discover_runs

    root = Path(root)
    runs = discover_runs(root)
    verdicts = [
        ensure_verdict(run, refresh=refresh, settle_unknown=settle_unknown)
        for run in runs
    ]
    return sorted(verdicts, key=lambda v: v.run)


def render_report(verdicts: list[ScaleVerdict]) -> str:
    lines = [
        "# ROCS scale audit",
        "",
        "Every stored `reinforced_raw.csv.gz` is checked for the scale of its `ROCS`",
        "column. `raw` means [0, 2] TanimotoCombo. `normalized` means the scorer applied",
        "a `ClippedScore` modifier, so the arm holds `min(raw, clip) / clip` and must be",
        "recovered with that clip. The pre-2026-11-04 clip was 1.8 while the figure code",
        "recovered with 2.0, which inflated ROCS by 1.111 and saturated everything above",
        "the clip at 2.0.",
        "",
        "| Run | Verdict | Clip | Factor | Satur. | Ratio | Evidence |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for verdict in verdicts:
        ratio = "—" if verdict.ratio_median is None else f"{verdict.ratio_median:.3f}"
        clip = "—" if verdict.clip is None else f"{verdict.clip:.2f}"
        satur = f"{100 * verdict.saturated_fraction:.1f}%" if verdict.n_rows else "—"
        lines.append(
            f"| `{verdict.run}` | {verdict.verdict} | {clip} | {verdict.factor:.3f} "
            f"| {satur} | {ratio} | {verdict.evidence} |"
        )
    if not verdicts:
        lines.append("| — | — | — | — | — | — | no arms found |")
    flagged = [v for v in verdicts if v.is_inflated]
    lines.extend(
        [
            "",
            f"Arms inflated by the 2.0-vs-{LEGACY_CLIP} clip mismatch: **{len(flagged)}**.",
        ]
    )
    for verdict in flagged:
        lines.append(
            f"- `{verdict.run}`: stored normalized with clip {verdict.clip}, "
            f"{verdict.n_saturated}/{verdict.n_rows} molecules at the clip."
        )
    unknown = [v for v in verdicts if v.verdict == UNKNOWN]
    if unknown:
        lines.append("")
        lines.append("Arms whose scale could not be decided (excluded from ROCS rankings):")
        for verdict in unknown:
            lines.append(f"- `{verdict.run}`: {verdict.evidence}")
    return "\n".join(lines) + "\n"


def write_report(
    root: Path | str = OUTPUT_ROOT,
    *,
    refresh: bool = False,
    settle_unknown: bool = False,
) -> Path:
    root = Path(root)
    verdicts = audit(root, refresh=refresh, settle_unknown=settle_unknown)
    report = root / REPORT_NAME
    report.write_text(render_report(verdicts))
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "refresh", "settle"))
    parser.add_argument("--root", default=str(OUTPUT_ROOT))
    parser.add_argument("--run", default=None)
    args = parser.parse_args(argv)

    refresh = args.action == "refresh" or args.action == "settle"
    settle_unknown = args.action == "settle"
    if args.run:
        verdict = ensure_verdict(
            Path(args.root) / args.run, refresh=refresh, settle_unknown=settle_unknown
        )
        print(json.dumps(verdict.payload(), indent=2))
        return 0

    report = write_report(
        args.root, refresh=refresh, settle_unknown=settle_unknown
    )
    print(report)
    print(report.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
