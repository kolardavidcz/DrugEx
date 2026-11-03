"""Command line entry point for the conformer package.

Examples
--------
::

    python -m _david.code.conformer.cli migrate
    python -m _david.code.conformer.cli registry
    python -m _david.code.conformer.cli figures
    python -m _david.code.conformer.cli cache-stats

    # run a real study on the EGCG stress panel
    python -m _david.code.conformer.cli study methods \
        --panel egcg --output _david/output/conformer/runs/egcg_methods --workers 8

Every command is idempotent and uses the content-addressed cache, so a rerun
after a crash resumes instead of restarting.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import pandas as pd

from _david.code.conformer.analysis import loaders, reproduce
from _david.code.conformer.analysis.preparation import build_registry, write_component_tables
from _david.code.conformer.core import datasets, storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import MoleculeSpec
from _david.code.conformer.core.paths import (
    CACHE_ROOT,
    COMPONENT_DIRS,
    DATA_ROOT,
    OUTPUT_ROOT,
    cache_root,
    component_dir,
)
from _david.code.conformer.prepare.panels import egcg_panel
from _david.code.conformer.studies import clustering as clustering_study
from _david.code.conformer.studies import color_shape as color_shape_study
from _david.code.conformer.studies import optimization_methods as method_study
from _david.code.conformer.studies import preparation as preparation_study
from _david.code.conformer.studies.base import run_study
from _david.code.conformer.studies.measure import MeasureExecutor


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def _panel_from_legacy(raw_dir: Path, split: str, name: str) -> Path:
    """Convert a legacy calibration panel CSV into the canonical schema."""
    source = raw_dir / split / "panel.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    out = pd.DataFrame(
        {
            "name": frame["name"],
            "smiles": frame["smiles"],
            "source": "drugcentral_calibration",
            "tags": "",
            "split": frame["split"] if "split" in frame.columns else split,
            "molecule_id": frame["molecule_id"] if "molecule_id" in frame.columns else "",
        }
    )
    return datasets.write_panel(name, _specs_from_frame(out))


def _specs_from_frame(frame: pd.DataFrame) -> tuple[MoleculeSpec, ...]:
    specs: list[MoleculeSpec] = []
    for _, row in frame.iterrows():
        specs.append(
            MoleculeSpec(
                name=str(row["name"]),
                smiles=str(row["smiles"]),
                source=str(row.get("source", "legacy")),
                tags=(f"split:{row.get('split', '')}",) if row.get("split", "") else (),
            )
        )
    return tuple(specs)


def migrate() -> dict[str, Path]:
    """Copy validated legacy tables and panels into canonical storage."""
    written: dict[str, Path] = {}
    sources: dict[str, dict[str, object]] = {}
    for name in datasets.TABLES:
        legacy = loaders.LEGACY_TABLES.get(name)
        if legacy is None or not legacy.is_file():
            continue
        frame = datasets.load_table(name, legacy=loaders.LEGACY_TABLES)
        written[f"table:{name}"] = datasets.write_table(name, frame)
        sources[name] = {
            "legacy_path": legacy.as_posix(),
            "canonical_path": written[f"table:{name}"].as_posix(),
            "bytes": legacy.stat().st_size,
            "sha256": loaders.load(name).digest,
        }

    calibration_raw = loaders.LEGACY_CALIBRATION
    for split, panel_name in (
        ("development", "drug_panel_development"),
        ("confirmation", "drug_panel_confirmation"),
    ):
        source = calibration_raw / split / "panel.csv"
        if source.is_file():
            written[f"panel:{panel_name}"] = _panel_from_legacy(
                calibration_raw, split, panel_name
            )

    written["panel:egcg"] = datasets.write_panel("egcg", egcg_panel())
    written["sources"] = storage.write_json(
        DATA_ROOT / "canonical" / "migration_sources.json", {"tables": sources}
    )
    return written


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def registry(panels: Sequence[str] | None = None) -> dict[str, Path]:
    """Build the canonical molecule registry from the requested panels."""
    names = list(panels) if panels else list(datasets.PANELS)
    specs: list[MoleculeSpec] = []
    for name in names:
        specs.extend(datasets.load_panel(name))
    frame, errors = build_registry(
        datasets.deduplicate_specs(specs), cache=ArtifactCache()
    )
    written = write_component_tables(frame, output_dir=component_dir("preparation"))
    if errors:
        written["errors"] = storage.write_csv(
            component_dir("preparation") / "registry_errors.csv",
            pd.DataFrame(errors),
        )
    return written


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures(*, canonical_only: bool = False, output_root: Path | None = None) -> dict[str, Path]:
    """Regenerate every component deliverable."""
    return reproduce.regenerate(output_root=output_root, canonical_only=canonical_only)


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------


def _molecules(panel: str | None, panel_file: Path | None) -> tuple[MoleculeSpec, ...]:
    if panel_file is not None:
        frame = pd.read_csv(panel_file, keep_default_na=False)
        return _specs_from_frame(frame)
    if panel == "egcg-panel":
        return egcg_panel()
    return datasets.load_panel(panel or "egcg")


def study_methods(args: argparse.Namespace) -> dict[str, Path]:
    molecules = _molecules(args.panel, args.panel_file)
    study = method_study.build_study(
        molecules,
        args.output,
        selections=tuple(args.selections),
        caps=tuple(args.caps),
        candidate_attempts=tuple(args.candidate_attempts),
        name=args.name,
    )
    result = run_study(
        study,
        MeasureExecutor(cache_root=cache_root().as_posix()),
        cache=ArtifactCache(),
        workers=args.workers,
    )
    return dict(result.artifacts)


def study_clustering(args: argparse.Namespace) -> dict[str, Path]:
    molecules = _molecules(args.panel, args.panel_file)
    study = clustering_study.build_study(
        molecules,
        args.output,
        selections=tuple(args.selections),
        caps=tuple(args.caps),
        candidate_attempts=tuple(args.candidate_attempts),
        name=args.name,
    )
    result = run_study(
        study,
        MeasureExecutor(cache_root=cache_root().as_posix()),
        cache=ArtifactCache(),
        workers=args.workers,
    )
    return dict(result.artifacts)


def study_color_shape(args: argparse.Namespace) -> dict[str, Path]:
    molecules = _molecules(args.panel, args.panel_file)
    color_shape_study.run_tversky_grid(
        molecules,
        args.output,
        cache=ArtifactCache(),
        workers=args.workers,
    )
    return {"tversky_grid": Path(args.output) / "tversky_color_grid.csv"}


def study_preparation(args: argparse.Namespace) -> dict[str, Path]:
    molecules = _molecules(args.panel, args.panel_file)
    frame, errors = preparation_study.audit_molecules(
        molecules,
        methods=tuple(args.methods),
        cache=ArtifactCache(),
    )
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    written = {"audit": storage.write_csv(target / "preparation_audit.csv", frame)}
    if errors:
        written["errors"] = storage.write_csv(target / "preparation_errors.csv", pd.DataFrame(errors))
    written["summary"] = storage.write_csv(
        target / "survival_summary.csv",
        preparation_study.survival_summary(frame),
    )
    return written


# ---------------------------------------------------------------------------
# cache stats
# ---------------------------------------------------------------------------


def cache_stats() -> int:
    """Print cache occupancy per namespace and return a shell exit code."""
    root = CACHE_ROOT
    if not root.exists():
        print(f"cache root {root.as_posix()} does not exist")
        return 1
    rows = []
    for namespace in sorted(path for path in root.iterdir() if path.is_dir()):
        # Metadata sidecars are bookkeeping, not payload; every other file is
        # one cached artifact and must be counted.
        payloads = [
            path
            for path in namespace.rglob("*")
            if path.is_file() and not path.name.endswith(".meta.json")
        ]
        size = sum(path.stat().st_size for path in payloads)
        rows.append({"namespace": namespace.name, "entries": len(payloads), "bytes": size})
    frame = pd.DataFrame(rows)
    if frame.empty:
        print(f"cache root {root.as_posix()} is empty")
        return 0
    frame["megabytes"] = (frame["bytes"] / 1e6).round(2)
    print(frame.to_string(index=False))
    print(f"total: {frame['megabytes'].sum():.2f} MB")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _add_study_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--panel", default="egcg", help="canonical panel name")
    parser.add_argument("--panel-file", type=Path, default=None, help="CSV with name,smiles columns")
    parser.add_argument("--output", type=Path, required=True, help="output directory for the run")
    parser.add_argument("--name", default=None, help="study name (defaults to the command)")
    parser.add_argument("--workers", type=int, default=1, help="process workers for study cells")
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 2, 4, 8, 12, 0])
    parser.add_argument("--candidate-attempts", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument(
        "--selections",
        nargs="+",
        default=["diverse"],
        choices=["diverse", "rmsd_0.35", "rmsd_0.50", "rmsd_1.25"],
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m _david.code.conformer.cli",
        description="Conformer ensembles, shape/color scoring, and reproducible studies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate", help="copy legacy tables and panels into canonical storage")
    registry_parser = subparsers.add_parser("registry", help="build the canonical molecule registry")
    registry_parser.add_argument("--panels", nargs="+", default=None)

    figures_parser = subparsers.add_parser("figures", help="regenerate component figures and tables")
    figures_parser.add_argument("--canonical-only", action="store_true")
    figures_parser.add_argument("--output-root", type=Path, default=None)

    subparsers.add_parser("cache-stats", help="print cache occupancy")

    methods_parser = subparsers.add_parser("study", help="run a study matrix")
    study_kinds = methods_parser.add_subparsers(dest="kind", required=True)
    for kind in ("methods", "clustering", "color-shape"):
        study_parser = study_kinds.add_parser(kind)
        _add_study_options(study_parser)
    preparation_parser = study_kinds.add_parser("preparation")
    preparation_parser.add_argument("--panel", default="egcg")
    preparation_parser.add_argument("--panel-file", type=Path, default=None)
    preparation_parser.add_argument("--output", type=Path, required=True)
    preparation_parser.add_argument("--methods", nargs="+", default=list(preparation_study.AUDIT_METHODS))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "migrate":
        written = migrate()
    elif args.command == "registry":
        written = registry(args.panels)
    elif args.command == "figures":
        written = figures(canonical_only=args.canonical_only, output_root=args.output_root)
    elif args.command == "cache-stats":
        return cache_stats()
    elif args.command == "study":
        if args.name is None:
            args.name = args.kind.replace("-", "_")
        handlers = {
            "methods": study_methods,
            "clustering": study_clustering,
            "color-shape": study_color_shape,
            "preparation": study_preparation,
        }
        written = handlers[args.kind](args)
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command!r}")
        return 2

    for name, path in sorted(written.items()):
        print(f"{name}: {Path(path).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
