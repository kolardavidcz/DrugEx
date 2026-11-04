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
