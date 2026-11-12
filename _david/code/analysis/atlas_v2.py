"""Render the collapse-aware experiment tree atlas: ``python -m _david.code.analysis.atlas_v2``.

Reads the scanner's ``collapse_scan.json`` (running the scanner first when it is missing or
``--rescan`` is given) and renders ``collapse_tree_atlas.html`` from
``atlas_v2_template.html``: every run a leaf, every configuration axis a reorderable floor,
and each edge coloured by the collapsed fraction of the branch it carries — the ε=0 branch
reads red at a glance, which is the whole point of the campaign's current question.

Every judgement the tree encodes — verdict pills, the (desired, unique) frontier stars, the
per-family medals — reads the **stop arm** (``final_desired_ratio_at_stop.tsv``), the live
final policy, never the gmean checkpoint's arm. The gmean numbers are evidence in a run's
modal, not a basis for ranking.

    python -m _david.code.analysis.atlas_v2 [--scan FILE.json] [--out FILE.html] [--rescan]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from _david.code.analysis.collapse import _depict, build as scan_build
from _david.code.analysis.diversity import load_profile
from _david.code.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

OUTPUT_ROOT = PROJECT_ROOT / "_david" / "output"
TEMPLATE = Path(__file__).with_name("atlas_v2_template.html")

#: The tree's floors, most to least influential as a default question. Epsilon leads
#: because the scan's own finding is that exploration is the collapse axis; the buttons
#: reorder the whole tree, so this is only where the argument starts.
LAYERS: tuple[dict[str, str], ...] = (
    {"id": "epsilon", "label": "exploration ε"},
    {"id": "reward", "label": "reward scheme"},
    {"id": "geometry", "label": "geometry (refs · minimization)"},
    {"id": "budget", "label": "conformer budget"},
    {"id": "color", "label": "colour force field"},
    {"id": "shape_color", "label": "shape/colour weight"},
)
DEFAULT_LAYERS = [layer["id"] for layer in LAYERS]

#: A family per objective set: the campaign's one split that changes what "desired" means.
FAMILIES = (
    {"key": "rocs", "label": "SA + ROCS"},
    {"key": "qed", "label": "SA + ROCS + QED"},
)

#: Verdicts that cannot win a medal: a collapsed arm's desired ratio describes one repeated
#: series, and pilot/incomplete arms have not been measured at full scale.
UNMEDALABLE = ("collapsed", "pilot", "incomplete")


def _factors(axes: dict[str, Any]) -> dict[str, str]:
    """The campaign's axes, folded onto the tree's six floors as short strings."""
    epsilon = axes.get("epsilon")
    refs, cons = axes.get("reference_conformers"), axes.get("max_conformers")
    weight = axes.get("alignment_shape_weight")
    return {
        "reward": str(axes.get("scheme") or "—"),
        "epsilon": "—" if epsilon is None else f"{epsilon:g}",
        "geometry": "—" if refs is None else f"K{refs:g} · {'opt' if axes.get('minimize') else 'raw'}",
        "budget": "—" if cons is None else f"C{cons:g}",
        "color": str(axes.get("color_ff") or "—"),
        "shape_color": "—" if weight is None else f"w {weight:g}",
    }


def _constants(family_runs: list[dict[str, Any]], since: str | None) -> str:
    """What a family's runs hold still, spelled out under its tree."""
    pinned = []
    for key in (
        "scheme", "epsilon", "minimize", "reference_conformers",
        "max_conformers", "color_ff", "alignment_shape_weight",
    ):
        values = {str(r["axes"].get(key)) for r in family_runs}
        if len(values) == 1:
            pinned.append(f"{key}={values.pop()}")
    draws = {r.get("n_expected") for r in family_runs}
    scale = f"{max(draws):,}-molecule final arms" if draws else "final arms"
    bits = [f"runs from {since or 'the start'} onwards", scale, "judged on the stop arm (live policy)"]
    if pinned:
        bits.append("pinned: " + ", ".join(pinned))
    return " · ".join(bits)


def _tree_run(record: dict[str, Any]) -> dict[str, Any]:
    """One scan record as the template's run object."""
    axes = record.get("axes") or {}
    tree_run = {
        "run_id": record["run"],
        "label": record["run"],
        "verdict": record["verdict"],
        "repo": record["repo"],
        "legacy": bool(record.get("legacy")),
        "state": record["state"],
        "hours": record.get("hours"),
        "epochs_run": record.get("epochs_run"),
        "n_expected": record.get("n_expected"),
        "family_key": "qed" if axes.get("include_qed") else "rocs",
        "factors": _factors(axes),
        "arms": record.get("arms") or {},
        "chemistry": record.get("chemistry") or {},
        "hac_drift": record.get("hac_drift"),
        "postpeak_decay": record.get("postpeak_decay"),
        "curve": record.get("curve") or {},
        "repeats": record.get("repeats") or [],
        "lead": record.get("lead"),
        "missing": record.get("missing") or [],
        "frontier": bool(record.get("frontier")),
        "reasons": record.get("reasons") or [],
        "flags": record.get("flags") or [],
    }
    # The diversity profile rides along from its own per-run cache, fresh-checked against
    # the arm tables it was computed from — the scan JSON stays a duplication report and
    # the chemical layer stays optional, so a run without a profile still renders.
    profile = load_profile(OUTPUT_ROOT / record["run"])
    if profile:
        tree_run["diversity"] = {
            key: profile[key] for key in ("arms", "cross", "thumb") if profile.get(key)
        }
    return tree_run


def _medals(runs: list[dict[str, Any]]) -> None:
    """Rank 1–3 per family by stop-arm desired, among runs the verdict can stand behind.

    A medal is the atlas's "read this run first" signal, so it must never land on a
    collapsed arm whose desired ratio is one repeated series counted 5000 times, nor on
    an arm that was never scored at full scale.
    """
    for family in {r["family_key"] for r in runs}:
        eligible = [
            r for r in runs
            if r["family_key"] == family
            and r["verdict"] not in UNMEDALABLE
            and (r["arms"].get("stop") or {}).get("desired_ratio") is not None
        ]
        eligible.sort(key=lambda r: -r["arms"]["stop"]["desired_ratio"])
        for rank, run in enumerate(eligible[:3], start=1):
            run["rank"] = rank


def tree_payload(scan: dict[str, Any]) -> dict[str, Any]:
    """The whole template payload from one collapse scan."""
    runs = [_tree_run(record) for record in scan["runs"]]
    _medals(runs)
    counts = dict(scan.get("counts") or {})
    totals = dict(scan.get("totals") or {})
    for verdict in ("collapsed", "concentrated", "clean", "unlearned"):
        totals.setdefault(verdict, counts.get(verdict, 0))
    families = []
    for family in FAMILIES:
        family_records = [
            record for record in scan["runs"]
            if ("qed" if (record.get("axes") or {}).get("include_qed") else "rocs") == family["key"]
        ]
        families.append({**family, "constants": _constants(family_records, scan.get("since"))})
    return {
        "generated": scan.get("generated"),
        "roots": scan.get("roots") or [],
        "since": scan.get("since"),
        "runs": runs,
        "families": families,
        "available_layers": [dict(layer) for layer in LAYERS],
        "default_layers": list(DEFAULT_LAYERS),
        "counts": counts,
        "totals": totals,
    }


def render(payload: dict[str, Any], out: Path) -> None:
    """Attach depictions for the runs worth looking at, then write the atlas."""
    for record in payload["runs"]:
        # Lead structures only for the runs the tree celebrates; every run's repeats,
        # because a collapsed run's evidence is what it repeated.
        if record.get("lead") and (record["frontier"] or record.get("rank")):
            record["lead"]["svg"] = _depict(record["lead"]["smiles"])
        for repeat in record.get("repeats") or []:
            repeat["svg"] = _depict(repeat["smiles"])
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # A "</script>" inside a string would end the payload tag early; SMILES cannot
    # contain it, but the check is one replace and then the class of bug is gone.
    serialized = serialized.replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__PAYLOAD__", serialized),
                   encoding="utf-8")
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scan", type=Path, default=None,
        help="collapse_scan.json to read (default: <output>/collapse_scan.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="atlas path (default: <output>/collapse_tree_atlas.html)",
    )
    parser.add_argument(
        "--rescan", action="store_true",
        help="run the collapse scanner first, even if a scan file exists",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scan_path = args.scan or OUTPUT_ROOT / "collapse_scan.json"
    if args.rescan or not scan_path.is_file():
        scan_build([OUTPUT_ROOT], OUTPUT_ROOT / "collapse_atlas.html")
    scan = json.loads(scan_path.read_text())

    out = args.out or OUTPUT_ROOT / "collapse_tree_atlas.html"
    render(tree_payload(scan), out)
    print(f"tree atlas: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
