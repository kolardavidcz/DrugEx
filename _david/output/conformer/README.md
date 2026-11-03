# Conformer deliverables

Four components, one question each. Every figure here is regenerated from
registered tables; nothing is hand-edited.

```bash
python -m _david.code.conformer.cli figures --canonical-only
```

```
conformer/
├── 01_optimization_methods/     reference/candidate physics, K, C, energy window
├── 02_color_shape_scoring/     the production scorer and its failure modes
├── 03_clustering_benchmark/    how a 50–100 pose pool becomes a K-pose bank
└── 04_molecule_preparation/    what the inputs are and what survives each stage
```

| Component | Figure | One-line answer |
|---|---|---|
| 1 | `egcg_omnibus_diagnostic.png` | For EGCG, `K=4, C=20` minimized reaches 0.999 self-recovery in 2.39 s; unminimized ETKDG reaches 0.980 in 1.29 s |
| 1 | `drug_panel_stratified_vs_trimmed.png` | Across 128 drugs, MMFF94s reaches 0.910 vs ETKDG 0.862 (+0.048) for +0.57 s per molecule |
| 1 | `isomer_stratification.png` | The minimization gain is not universal: +0.030 at 1–2 isomers, +0.118 at 33+ |
| 1 | `reference_counts_vs_energy_window.png` | The energy window is a resource dial: which embedded poses survive 3/5/8/12 kcal·mol⁻¹ |
| 2 | `tversky_color_162grid_selectivity.png` | Symmetric TanimotoCombo is the defensible default; every asymmetric dial raises a specific false-positive floor |
| 3 | `drug128_definitive_synthesis.png` | Farthest-first diversity ≡ Butina 0.35 Å at `K=4` (0.817 vs 0.816); 1.25 Å loses 0.012 by merging distinct basins |
| 4 | *(no figure — tables only)* | 527 inputs prepared, 0 failures; 73 tautomer/protomer normalizations and 5 symmetry collapses disclosed |

## Reading conventions

**Score scale.** Every `combo`/`score` column is normalized: `0.5 × (shape +
color)` on `[0, 1]`. The physical ROCS TanimotoCombo on `[0, 2]` is `2 × value`.
The RL reward consumes the normalized axis, so a table never needs a mental
doubling.

**Isomer strata.** Bins are `1–2, 3–4, 5–8, 9–15, 16–32, 33+`, assigned from the
*true* stereoisomer upper bound, not from how many isomers a study sampled. The
`9–15` bin is kept even when empty so a gap is visible.

**Provenance.** Each PNG has a sibling `.manifest.json` naming every input table
with its SHA-256, plus the git revision, dirty flag and package versions. The
canonical tables record where they were migrated from in
`_david/data/conformers/canonical/migration_sources.json`.

**Denominators.** Success rates are reported against *scored* inputs, and
`drug_panel_development_summary.csv` carries both `inputs` (128) and
`scored_inputs` (126). Two molecules are excluded by the colour-feature gate, not
by failure — that distinction is in the table, not in prose.

## What is not here

The multi-thousand-cell `mega` conformer sweeps and the old ad-hoc calibration
scripts are retired. Their code is frozen in `_david/code/archive/conformer_v1/`;
the sweeps are archived evidence, not a maintained path. If a new question is
worth asking, it becomes an arm in one of the four component studies.
