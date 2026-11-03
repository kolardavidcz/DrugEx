# Archived Conformer Package (v1, `_david/code/archive/conformer_v1/`)

This is the pre-refactor `_david/code/conformer/` package, frozen as-is on
2026-11-03 when the suite was rewritten around four components.

**Nothing here is imported by production code.** It is kept only so that every
number in `_david/code/conformer/results.md` from before 2026-11-03 can still be
traced back to the script that produced it.

## What moved where

| Archived path | v1 responsibility | v2 location |
|---|---|---|
| `core/sampling.py`, `core/utils.py` | shared sampling/IO helpers | `conformer/core/` (`storage`, `provenance`, `cache`) |
| `reference/reference_bank.py` | diversity-ordered reference banks (R50/I8) | `conformer/ensemble/select.py` |
| `reference/drug_panel.py` | DrugCentral curation for the 128-drug panel | `conformer/prepare/registry.py` + `data/conformers/canonical/panels/` |
| `reference/diagnostics.py` | singleton/protomer diagnostics | `conformer/prepare/registry.py` |
| `benchmarks/mega.py` | cross-method conformer/ROCS mega-analysis | `conformer/studies/optimization_methods.py` |
| `benchmarks/batch_probe.py` | multi-core batch scoring benchmark | `conformer/studies/measure.py` |
| `benchmarks/reward_calibration.py` | historical singleton Pareto benchmark | `conformer/studies/optimization_methods.py` |
| `benchmarks/self_recovery.py`, `benchmarks/decoy_benchmark.py` | self-recovery / decoy selectivity | `conformer/studies/color_shape.py` |
| `benchmarks/rl_batch_calibration.py`, `benchmarks/rl_cost_probe.py` | live-RL cost probes | **retired** (thesis §RL cost lives in `_david/output/20261028_*` reports) |
| `analysis/panel_figures.py`, `analysis/render_6panel_synthesis.py` | 128-drug + synthesis figures | `conformer/analysis/*` and `conformer/analysis/reproduce.py` |
| `analysis/analyze.py`, `analysis/visualization.py` | mega-study analysis and renderers | `conformer/analysis/` |
| `orchestrator.py`, `settings.py` | ad-hoc run orchestration and per-study settings | `conformer/studies/base.py` (`StudySpec`, `run_study`) and `conformer/core/contracts.py` |

## Canonical results are *not* archived here

The tables and figures that these scripts produced were migrated into
`_david/data/conformers/canonical/` (with SHA-256 provenance in
`migration_sources.json`) and re-rendered by
`python -m _david.code.conformer.cli figures`. Read
`_david/code/conformer/results.md` for the authoritative, still-valid findings.
