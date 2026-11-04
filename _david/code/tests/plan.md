# Test suite pruning — meaningful tests only

Goal: the suite must fail only when **behaviour or a contract breaks**, never when the
owner tunes a research constant, renames a figure, reformats a label, or refactors prose.

Rule used to classify every test:

- **Keep** — computes something (math, I/O, multiprocessing, RDKit, filesystem), enforces a
  schema/boundary, or pins a real past bug.
- **Rewrite** — the behaviour matters but the assertion pins a chosen default, a figure name,
  a label string, or a count that legitimately changes during research.
- **Delete** — echoes a dataclass constructor, greps for absent legacy names, duplicates
  another test, tests only that matplotlib wrote a file, or is misnamed/broken.

Must remain true: `ruff check _david` clean; `python -m unittest discover -s _david/code/tests`
green; no production code touched (`_david/code/tests/molecular_dynamics/` untouched).

Status 2026-11-04: executed — 149 → 143 tests, all green, plus a new regression test for
`verify_disjoint_conformer_seeds` (the `42 * 43 == 43 * 42` coordinate-clone bug).

## Delete (pointless / misleading)

- [x] `test_conformer_analysis`: taxonomy re-export identity (`common.ISOMER_BIN_LABELS is ...`).
- [x] `test_conformer_analysis`: `test_energy_window_figure_renders_from_canonical_data` —
      asserts one figure name renders; the visualization rework will delete/rename it.
- [x] `test_conformer_studies`: `test_zero_arm_matrix_is_rejected_by_run_study` — name lies;
      builds a valid 1-repeat study and duplicates the cartesian-product test.
- [x] `test_pipeline_architecture`: `test_pipeline_config_direct_access` — echoes constructor
      args back and asserts removed kwargs-getters are gone (duplicated in schema test).
- [x] `test_schema_standards`: `test_architectural_boundaries_no_kwargs_getters` — absence of
      long-removed attributes is not behaviour.
- [x] `test_conformer_scoring`: `test_grid_is_eighteen_by_nine` — grid size is a calibration
      artifact; `test_grid_scores_shape_and_symmetry` already checks tensor shape.
- [x] `test_conformer_scoring`: `TestScoreSpecScales` — duplicate of the `ScoreSpec`
      validation in `test_conformer_core`.
- [x] `test_metacentrum`: `test_submit_parser_defaults_to_cpu_rocs_resources` — echoes argparse
      defaults; resource behaviour is covered by `JobSpec`/`qsub_command`.
- [x] `test_metacentrum`: `test_launcher_keeps_only_the_three_required_environment_checks` —
      negatives on shell text (`trap`, `tee`, `rsync`) that a better launcher may re-add.
- [x] `test_metacentrum`: `test_pbs_fallbacks_match_the_routed_default` — pins walltime/PBS
      header text, breaks on a resource change.

## Rewrite (behaviour kept, research constant released)

- [x] `test_conformer_core.reference_spec_default_matches_production_bank` → seed streams of
      the production reference/candidate specs are disjoint, and `verify_disjoint_conformer_seeds`
      rejects the historical `42/43` collision (production code, currently untested).
- [x] `test_conformer_core.minimize_spec_label_is_stable` → enabled/raw are distinguishable and
      identify the force field, no exact string.
- [x] `test_conformer_core.panel_loads_into_specs_with_split_tag` → non-empty, unique names,
      split tag present; drop `len == 128`.
- [x] `test_conformer_core.component_directories_are_distinct` → distinctness only; drop `== 4`.
- [x] `test_conformer_prepare.registry_columns_and_flags` → `enumeration_cap` equals
      `REGISTRY_ISOMER_CAP`, not the literal `32`.
- [x] `test_conformer_analysis.default_scorer_neighbourhood_covers_the_axes` → synthetic grid:
      one row per `NEIGHBOURHOOD` entry in order, unmatched setting skipped; drop production-value
      bounds and the canonical-data dependency.
- [x] `test_conformer_scoring.cache_round_trip...` → keep hit/indices/zero-on-hit, drop
      `first.seconds > 0` (timer resolution flake).
- [x] `test_conformer_studies.deployable_pairs_share_one_reference` → pass the reference
      explicitly and assert every method is covered; drop the `mmff94s_water` default pin.
- [x] `test_conformer_studies.arm_names_are_unique_and_descriptive` → unique + physics
      identifiable; drop the exact `ref-…__cand-…` format and `normalized` scale pins.
- [x] `test_pipeline_architecture.create_rocs_scorer_and_environment` → `include_qed` adds one
      QED scorer and one threshold; drop `color_ff`, `rocs_clip_upper`, `len == 3` default pins.
- [x] `test_schema_standards.pipeline_config_canonical_validation` → defaults are *valid*
      canonical values (`resolve_reward_scheme` accepts, criteria in `VALID_CRITERIA`,
      `n_eval_samples > 0`, `0 < cpu_ratio <= 1`); keep every rejection case.
- [x] `test_schema_standards.production_scoring...` → keep variant rejection in CLI/config/factory;
      drop "legacy fields/flags absent" lists.
- [x] `test_metacentrum.JobSpec.test_defaults...` → CPU-only invariants, positive resources,
      queue derives from walltime via the router; drop `16/32/15/24 h/spec 8.0` literals.
- [x] `test_metacentrum.qsub_command...` → assert each env var (`RUN_SPEC`, `STORAGE_DIR`,
      `RELEASE_DIR`) appears; drop one exact joined string so a new env var is allowed.

## Keep (already meaningful)

- Conformer math (`tversky`/`grid_scores`/self-recovery), selection modes and cache keys,
  contracts validation, study execution/serial-vs-parallel/provenance, dataset resolution,
  taxonomy bins, figure provenance manifest, RL explorer construction regression, ROCS
  colour-typing wiring, run-directory naming/collisions, MetaCentrum routing/spec/parser/
  job-body behaviour, schema single-name rule and strict reward names.

# Visualization rework — figures 03, 06, 07, 11, drop 08 & reference

Owner request (2026-11-04), figure by figure. Every renderer below is judged by "does
this panel change a decision?", not by "does it look busy". Palette rule: **full
saturation only for the values the checkpoint decision is made on** (`desired_ratio`,
`avg_amean`, `avg_gmean`); every other tracked statistic is pastel/muted.

- [x] **03 A** — arithmetic mean leaves panel A (it is normalized). Panel A keeps
      `loss_train` only (plus a muted rolling median for trend).
- [x] **03 B** — becomes the normalized panel: `avg_amean`, `avg_gmean`, `desired_ratio`
      in full colour; `valid_ratio`, `unique_ratio` in pastel. `unique_ratio` must
      actually be drawn (it exists in `_fit.tsv`; the old code silently skipped it for
      runs without the column and nobody noticed). Best-checkpoint marker reads the
      recorded `best_epoch` column instead of guessing `avg_amean`.
- [x] **03 C** — objective desirability (ROCS mean, QED mean pastel) on the left axis and
      **recomputed raw RDKit SA (~3–4)** on a dedicated right axis, median in solid grey
      with a grey IQR band. The old panel plotted the `[0,1]` SA *desirability* under the
      label "SA Desirability Mean", i.e. the physical quantity was never on the figure.
      `_smiles.tsv` stores only desirability, so raw SA is recomputed from SMILES with
      `drugex...sascorer.calculateScore` on a deterministic ≤256-molecule/epoch subsample
      (~1000 mol/s, memoised per SMILES) — cheap and unbiased for a median.
- [x] **03 D** — the epsilon panel was a hard-coded horizontal 0.05 line (no run records
      an `epsilon` column), so it carried no data. Replace with the per-epoch ROCS
      distribution: median, IQR band, p95, cumulative-best p95 ceiling. This shows
      whether the policy converged onto a narrow mode or kept exploring.
- [x] **06** — delete the Δ% tornado (it restates `comparison_scorecard.csv`) and replace
      with **objective-conflict correlation matrices**: Spearman ρ over ROCS, raw SA, QED,
      MW, HAC, LogP, TPSA, RotBonds, Fsp3, rings for baseline, reinforced, and
      reinforced − baseline. This is the reward-design diagnostic the tornado could not
      give: it shows which objectives the policy coupled or decoupled.
- [x] **07** — stop comparing `unique/100`, entropy (bits) and Gini-Simpson on one mixed
      unit axis. New panels: (A) cumulative scaffold coverage curve (rank → % of library
      covered) with unique-scaffold/entropy annotations, (B) scaffold novelty partition
      (shared with prior / novel / lost) plus Jaccard, (C) top-6 scaffold gallery tagged
      `new` vs `shared`. This answers "did RL invent chemistry or collapse onto one core?".
- [x] **08** — delete. Desired properties are already in 01/02/11; a ROCS-vs-MW/QED scatter
      adds no tracked metric.
- [x] **11** — absorb 08's intent by tracking *more* properties, not fewer panels: 3×3 with
      ROCS desirability, QED, raw SA, MW, HAC, LogP, TPSA, Fsp3, RotBonds (median + IQR +
      p95 + seed reference). SA is recomputed raw (same helper as 03 C) — the old panel
      plotted desirability on a 1–6 axis, so it was effectively broken. Axis limits come
      from the data, not pinned research constants.
- [x] **reference_3d_superposition.png** — delete the renderer and its call sites (`pipeline`,
      `run_audit`, package exports, `summary.md` link). Figure 10 is the better 3D view.
      Stage 0 still writes `query_molecule_3d.sdf`; only the redundant picture goes.

Constraints: no figure-render tests are added (see policy above); `ruff check _david`
clean; suite green; verify by rendering from `_david/output/20261104_0447_egcg_eps0/`
(a real baseline+reinforced arm with a full monitor TSV).

Status 2026-11-05: executed. `visualization.py` now renders 8 figures; the retired three
(06 tornado, 08 shape/colour, reference ensemble) are removed from the pipeline, the audit
expectation, the package exports and `summary.md`. `20261104_0445_egcg_gold_oe` and
`20261104_0447_egcg_eps0` were re-rendered in place from stored arms (≈80 s per run), and
`ruff check _david` plus 143/143 tests pass.
