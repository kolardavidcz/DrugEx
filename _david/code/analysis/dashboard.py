"""One offline HTML page comparing every run in ``_david/output/``.

``report.md`` answers "what did this run do"; this answers "which run won", "which axis
moved it" and "what was never run". Self-contained: no CDN, no server, no side files.

    python -m _david.code.analysis.dashboard [--root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from _david.code.analysis._helpers import enrich_descriptors, iter_run_dirs, unique
from _david.code.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RUN_NAME = re.compile(r"^\d{8}_\d{4}_.+$")

#: The final policy, not a checkpoint chosen by a criterion: comparing runs on "the best
#: epoch each happened to reach" compares selection rules, not configurations.
ARM = "final_desired_ratio_at_stop.tsv"

#: Settings the campaign varies. A column identical in every row is noise.
AXES = (
    "scheme",
    "minimize",
    "epsilon",
    "reference_conformers",
    "max_conformers",
    "include_qed",
    "color_ff",
    "alignment_shape_weight",
)
GRID_PAIRS = (("minimize", "max_conformers"), ("epsilon", "reference_conformers"))
MOLECULES_SHOWN = 24


def read_run(run: Path) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """One table row and that run's scored molecules, from a single pass over the arm."""
    try:
        manifest = json.loads((run / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("%s: unreadable manifest (%s)", run.name, error)
        return None

    config = manifest.get("config") or {}
    summary = manifest.get("summary") or {}
    row: dict[str, Any] = {
        "run": run.name,
        "state": manifest.get("state", "unknown"),
        "arms": len(list(run.glob("final_*.tsv"))),
        "has_stop_arm": (run / ARM).is_file(),
        "epochs": summary.get("epochs_run"),
        "hours": round((manifest.get("elapsed_seconds") or 0) / 3600, 2) or None,
        "config": {key: config.get(key) for key in AXES},
        "n_unique": None,
        "desired": None,
        "rocs_median": None,
        "rocs_best": None,
        "sa_median": None,
        "scaffolds": None,
    }

    molecules = pd.DataFrame()
    if (run / ARM).is_file():
        try:
            frame = pd.read_csv(run / ARM, sep="\t")
        except (OSError, pd.errors.ParserError) as error:
            logger.warning("%s: %s unreadable (%s)", run.name, ARM, error)
            return row, molecules
        objective = next((c for c in frame.columns if "TanimotoCombo" in c), None)
        # enrich_descriptors is what the figures use, so ROCS comes back as published
        # TanimotoCombo whichever units this run stored it in.
        molecules = enrich_descriptors(unique(frame), objective, config.get("reference_smiles"), config)
    if not molecules.empty:
        if "Desired" in molecules:
            molecules = molecules[pd.to_numeric(molecules["Desired"], errors="coerce").fillna(0) == 1]
        rocs = pd.to_numeric(molecules.get("ROCS"), errors="coerce").dropna()
        row |= {
            "n_unique": int(len(molecules)),
            "desired": round(len(molecules) / max(len(unique(frame)), 1), 4),
            "rocs_median": round(float(rocs.median()), 4) if len(rocs) else None,
            "rocs_best": round(float(rocs.max()), 4) if len(rocs) else None,
            "sa_median": round(float(pd.to_numeric(molecules["Raw_SA"], errors="coerce").median()), 2),
            "scaffolds": int(molecules["Scaffold"].replace("", pd.NA).nunique()),
        }
        molecules = molecules.assign(run=run.name)
    return row, molecules


def best_molecules(frames: list[pd.DataFrame], limit: int = MOLECULES_SHOWN) -> list[dict]:
    """Highest ROCS across all runs, one card per scaffold: a core that wins in six
    configurations is one finding, not six."""
    if not frames:
        return []
    pool = pd.concat(frames, ignore_index=True).nlargest(limit * 20, "ROCS")
    cards, seen = [], set()
    for _, row in pool.iterrows():
        key = str(row.get("Scaffold") or row.get("Canonical") or "")
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            {
                "run": row["run"],
                "svg": _depict(str(row.get("Canonical") or row.get("SMILES") or "")),
                "rocs": round(float(row["ROCS"]), 3),
                "shape": round(float(row.get("ShapeTanimoto", float("nan"))), 2),
                "colour": round(float(row.get("ColorTanimoto", float("nan"))), 2),
                "sa": round(float(row.get("Raw_SA", float("nan"))), 2),
                "qed": round(float(row.get("QED", float("nan"))), 2),
            }
        )
        if len(cards) >= limit:
            break
    return cards


def _depict(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(220, 170)
    drawer.drawOptions().clearBackground = False
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return "—" if value is None else str(value)


def design_grids(rows: list[dict]) -> list[dict]:
    """Which axis combinations exist, and the best median each reached."""
    grids = []
    for row_field, col_field in GRID_PAIRS:
        cells: dict[str, dict] = {}
        for row in rows:
            key = f"{_cell(row['config'][row_field])}|{_cell(row['config'][col_field])}"
            entry = cells.setdefault(key, {"count": 0, "best": None})
            entry["count"] += 1
            if row["rocs_median"] is not None:
                entry["best"] = max(entry["best"] or 0.0, row["rocs_median"])
        grids.append(
            {
                "row_field": row_field,
                "col_field": col_field,
                "rows": sorted({_cell(r["config"][row_field]) for r in rows}),
                "cols": sorted({_cell(r["config"][col_field]) for r in rows}),
                "cells": cells,
            }
        )
    return grids


def build(root: Path, out: Path) -> Path:
    runs = sorted(
        (d for d in iter_run_dirs(root) if RUN_NAME.match(d.name)),
        key=lambda d: d.name,
        reverse=True,
    )
    if not runs:
        raise SystemExit(f"no run directories under {root}")
    logger.info("reading %d runs", len(runs))

    rows, frames = [], []
    for run in runs:
        result = read_run(run)
        if result is None:
            continue
        row, molecules = result
        rows.append(row)
        if row["state"] == "complete" and row["has_stop_arm"] and not molecules.empty:
            frames.append(molecules)

    complete = [r for r in rows if r["state"] == "complete" and r["has_stop_arm"]]
    payload = {
        "generated": datetime.now().isoformat(" ", "seconds"),
        "root": str(root),
        "runs": rows,
        "axes": list(AXES),
        "grids": design_grids(complete),
        "molecules": best_molecules(frames),
        "totals": {
            "runs": len(rows),
            "complete": len(complete),
            "incomplete": len(rows) - len(complete),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DrugEx run dashboard</title>
<style>
:root { color-scheme: light dark; --bg:#f8fafc; --fg:#0f172a; --panel:#fff; --line:#e2e8f0;
  --muted:#64748b; --head:#0f172a; --accent:#4f46e5; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0b1120; --fg:#e2e8f0; --panel:#111827; --line:#1f2937; --muted:#94a3b8; --head:#020617; } }
:root[data-theme="dark"] { --bg:#0b1120; --fg:#e2e8f0; --panel:#111827; --line:#1f2937;
  --muted:#94a3b8; --head:#020617; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
header { padding:20px 16px 14px; background:var(--head); color:#f8fafc; }
h1 { margin:0 0 4px; font-size:20px; } header p { margin:0; color:#cbd5e1; font-size:12.5px; }
.cards { display:flex; gap:12px; padding:16px 16px 4px; flex-wrap:wrap; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:120px; }
.card b { display:block; font-size:22px; } .card span { color:var(--muted); font-size:11.5px; text-transform:uppercase; }
nav { display:flex; gap:4px; padding:14px 16px 0; flex-wrap:wrap; }
nav button { border:1px solid var(--line); background:var(--panel); color:var(--fg); padding:7px 13px; border-radius:7px; cursor:pointer; }
nav button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
main { padding:14px 16px 48px; } section { display:none; } section.active { display:block; }
h2 { font-size:15px; margin:18px 0 8px; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--panel); font-size:12.5px; border:1px solid var(--line); }
th,td { padding:7px 9px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th { background:var(--bg); font-weight:600; cursor:pointer; position:sticky; top:0; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.state { padding:1px 7px; border-radius:999px; font-size:11px; font-weight:600; }
.complete { background:#dcfce7; color:#166534; } .running { background:#fee2e2; color:#991b1b; }
.unknown { background:var(--line); color:var(--muted); }
.grid { border-collapse:separate; border-spacing:4px; }
.grid th { background:transparent; cursor:default; position:static; }
.grid td { border:1px solid var(--line); border-radius:6px; min-width:94px; text-align:center; padding:8px 6px; }
.grid td.done { background:#dcfce7; border-color:#86efac; color:#14532d; font-weight:600; }
.grid td.empty { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
.mols { display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:12px; }
.mol { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:8px; }
.mol svg { width:100%; height:auto; display:block; }
.mol .meta { font-size:11px; color:var(--muted); overflow-wrap:anywhere; }
.mol b { color:var(--fg); }
.note { background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; padding:9px 12px; border-radius:8px; margin:10px 0; }
</style></head><body>
<header><h1>DrugEx run dashboard</h1><p id="sub"></p></header>
<div class="cards" id="cards"></div>
<nav id="nav"></nav>
<main>
  <section id="runs" class="active"><h2>Every run</h2><div class="scroll"><table id="runtable"></table></div></section>
  <section id="design"><div id="grids"></div></section>
  <section id="mols"><h2>Best molecules across all runs</h2>
    <div class="note">One card per Murcko scaffold, desired molecules only.</div>
    <div class="mols" id="molgrid"></div></section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=3) => v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d);
const cell = v => typeof v === 'boolean' ? (v ? 'on' : 'off') : (v ?? '—');

document.getElementById('sub').textContent = `${P.totals.runs} runs under ${P.root} · ${P.generated}`;
document.getElementById('cards').innerHTML = Object.entries(P.totals)
  .map(([k, v]) => `<div class="card"><b>${v}</b><span>${k}</span></div>`).join('');

document.getElementById('nav').innerHTML = [['runs','Runs'],['design','Design grid'],['mols','Molecules']]
  .map(([id, label], i) => `<button data-tab="${id}" class="${i ? '' : 'active'}">${label}</button>`).join('');
document.getElementById('nav').onclick = e => {
  const id = e.target.dataset.tab; if (!id) return;
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b === e.target));
  document.querySelectorAll('section').forEach(s => s.classList.toggle('active', s.id === id));
};

const COLS = [
  ['run', 'run', r => esc(r.run)],
  ['state', 'state', r => `<span class="state ${['complete','running'].includes(r.state) ? r.state : 'unknown'}">${esc(r.state)}</span>`],
  ...P.axes.map(a => [a, a, r => esc(cell(r.config[a]))]),
  ['rocs_median', 'ROCS med', r => num(r.rocs_median), 1],
  ['rocs_best', 'ROCS best', r => num(r.rocs_best), 1],
  ['desired', 'desired', r => num(r.desired), 1],
  ['scaffolds', 'scaffolds', r => r.scaffolds ?? '—', 1],
  ['sa_median', 'SA med', r => num(r.sa_median, 2), 1],
  ['epochs', 'epochs', r => r.epochs ?? '—', 1],
  ['hours', 'hours', r => num(r.hours, 2), 1],
];
let key = 'rocs_median', desc = true;
const draw = () => {
  const rows = [...P.runs].sort((a, b) => {
    const x = a[key] ?? a.config[key], y = b[key] ?? b.config[key];
    if (x === y) return a.run < b.run ? -1 : 1;
    if (x == null) return 1;
    if (y == null) return -1;
    return (x > y ? 1 : -1) * (desc ? -1 : 1);
  });
  document.getElementById('runtable').innerHTML =
    `<thead><tr>${COLS.map(([k, l]) => `<th data-k="${k}">${l}${k === key ? (desc ? ' ▾' : ' ▴') : ''}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr>${COLS.map(([, , f, n]) => `<td class="${n ? 'num' : ''}">${f(r)}</td>`).join('')}</tr>`).join('')}</tbody>`;
};
document.getElementById('runtable').onclick = e => {
  const k = e.target.dataset?.k; if (!k) return;
  desc = k === key ? !desc : true; key = k; draw();
};
draw();

document.getElementById('grids').innerHTML = P.grids.map(g => `
  <h2>${esc(g.row_field)} × ${esc(g.col_field)}</h2>
  <div class="scroll"><table class="grid"><thead><tr><th></th>
  ${g.cols.map(c => `<th>${esc(g.col_field)}=${esc(c)}</th>`).join('')}</tr></thead><tbody>
  ${g.rows.map(r => `<tr><th>${esc(g.row_field)}=${esc(r)}</th>${g.cols.map(c => {
      const e = g.cells[`${r}|${c}`];
      return e ? `<td class="done">${e.count} run${e.count > 1 ? 's' : ''}<br><small>best ${num(e.best)}</small></td>`
               : '<td class="empty">not run</td>';
    }).join('')}</tr>`).join('')}</tbody></table></div>`).join('')
  || '<div class="note">No complete run covers these axes.</div>';

document.getElementById('molgrid').innerHTML = P.molecules.map(m => `
  <div class="mol">${m.svg}<div class="meta"><b>ROCS ${num(m.rocs)}</b>
  · shape ${num(m.shape, 2)} · colour ${num(m.colour, 2)}<br>
  SA ${num(m.sa, 2)} · QED ${num(m.qed, 2)}<br>${esc(m.run)}</div></div>`).join('')
  || '<div class="note">No desired molecule in any complete run.</div>';
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "_david" / "output")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build(args.root, args.out or args.root / "dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
