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

## Round 2 (2026-11-05, user figure-by-figure review)

- [x] **03 A** — drop the "loss 15-epoch median" overlay (it duplicated the loss trend and
      shared its axis). Add the exploration noise ε on its **own right axis**, read from
      `manifest.json → config` (`epsilon`, `epsilon_end`, `decay_start_epoch`, `decay_epochs`,
      `decay_type`, `use_epsilon_decay`). No run records an epsilon column, so the schedule is
      reconstructed by the single source of truth `models.decayed_epsilon()`, which
      `DecayingSequenceExplorer.get_decayed_epsilon` now delegates to — the plotted curve is
      exactly the schedule training followed. Flat schedules are labelled with their value
      (`ε = 0.00`), runs without a manifest say so explicitly.
- [x] **03 B** — the criterion is self-inspected: `config.criteria` when it names a tracked
      column, otherwise the metric whose argmax epoch equals the monitor's `best_epoch`. The
      criterion keeps full colour + ★, every other tracked statistic goes pastel/dashed, and
      the panel title names the detected criterion. The unique-ratio "no data" bug is real
      but not missing data: `unique_ratio` is computed over valid molecules, so on runs where
      every valid molecule is unique it is *bit-identical* to `valid_ratio` and the second
      line hides underneath. When `np.allclose` holds the two collapse into one honest line
      ("Valid = Unique Ratio (no duplicates)") instead of silently plotting an invisible series.
- [x] **03 C** — mean (solid, full colour) and median (dashed, lighter) for ROCS and QED, plus
      raw SA mean + median + IQR band on the dedicated right axis. Removes the mean/median
      ambiguity that made the old single line unreadable.
- [x] **03 D** — kept as-is and extended: the monitor stores only the combined TanimotoCombo,
      so shape and colour are recovered by re-scoring a deterministic ≤20-molecule/epoch
      subsample (one ETKDG conformer, seed `0xC0FFEE`, ≈0.05 s per molecule, 2.7× faster with
      an 8-worker fork pool and bit-identical) and drawn on a third axis. Labelled as a
      diagnostic re-scoring, not the training reward.
- [x] **05** — two-row gallery. Row 1: seed + top-5 by desirability. Row 2: top-5 by
      TanimotoCombo **excluding** every SMILES already shown, with an explanatory caption cell
      in column 0. Makes "what the reward selected" vs. "what ROCS alone would select" a
      direct visual comparison (e.g. ROCS picks 0.995/0.939/0.931/0.887/0.886 at much worse SA).
- [x] `ResultAnalyzer.analyze` reads `manifest.json` and passes `run_config` +
      `query_molecule_3d.sdf` to figure 03 and a ROCS-ranked frame to figure 05.

Status 2026-11-05: executed. 03 renders in ≈1m45s per run (7 axes, ε on axis 4, raw SA on
axis 5, shape/color on axis 6; shape 0.42–0.53, colour 0.05–0.09). 05 renders 12 axes
(seed, LEAD #1–#5, caption, ROCS #1–#5). Both runs re-rendered in place, `ruff check _david`
clean, 143/143 tests pass.

## Round 3 (2026-11-05, five new diagnostics + alignment fixes)

User asked for extra figures ("feel free to add some extra images as i might better decide
later"), better 2D alignment in figure 05, and a way to align the exported SDF molecules.
Figure set grows from 8 to 13 PNGs (plus the aligned SDF).

### New figures (feed from one shared per-molecule table)

`epoch_molecule_table` parses `_smiles.tsv` once, recomputes QED, physical SA, ECFP4
similarity to the seed and the analyzer's `SelectionScore`, and maps SA through the run's own
`sa_score_lower`/`sa_score_upper` clip. Figures 12–16 consume that single frame instead of
re-scoring the monitor five times.

- [x] **12 `12_multiobjective_hypervolume.png`** — exact dominated hypervolume over
      (ROCS TanimotoCombo / 2, QED, clipped SA desirability) with the reference point at the
      origin, computed by a z-sweep + 2D skyline (hand-checked against a union-of-boxes
      calculation and verified monotone cumulative). (A) per-epoch and cumulative hypervolume
      against the seed's single-point volume, (B) best-so-far per objective with last-improvement
      dots, (C) Pareto front at the monitor's `best_epoch` (not the last epoch, which is often
      degraded) coloured by QED. The composite reward assumes weights; the hypervolume does not,
      so this is the weight-free answer to "is the population actually better?".
- [x] **13 `13_exploration_exploitation.png`** — per-epoch centroid path through
      (seed similarity, ROCS/2), (raw SA, ROCS/2) and (internal diversity, ROCS/2), with the
      baseline cloud behind, the seed marked and the best checkpoint starred. Diversity uses a
      vectorised mean pairwise ECFP4 Tanimoto. Panel C's narrow x-range is annotated as a zoom
      so noise is not mistaken for signal.
- [x] **14 `14_sampling_budget.png`** — stack of desired & new / desired duplicate / valid
      undesired / unparsable per epoch, plus cumulative useful output. Explains why
      `desired_ratio` plateaus and how much sampling is spent re-discovering known molecules.
- [x] **15 `15_lead_set_stability.png`** — consecutive top-k Jaccard overlap, age of the best
      molecule and chemical drift (mean max ECFP4 similarity to the previous top-k), for both
      the selection-score and ROCS rankings. Low overlap means the "best epoch" is sampling
      noise rather than a locked-in mode.
- [x] **16 `16_rocs_enrichment.png`** — ROCS ECDF per stage, enrichment factors against the
      baseline's own top 1/5/10 % thresholds, and when the global top-k leads first appeared.
      **Honest result for the eps0 run: the delivered 3000-molecule arm does not enrich the
      ROCS tail** (median 0.321 vs 0.340, p99 0.424 vs 0.430, max 0.498 vs 0.553, delivered-arm
      EF 0.70× / 0.49× / 0.55× at 1/5/10 %). The earlier "tail improvement" reading came from
      comparing single 50-molecule epochs; the figure now shows the full arm instead of hiding it.

### Fixes found while building the above

- [x] **Unparsable rows polluted every arm-level ROCS statistic.** `unscale_to_physical_metrics`
      keeps failed generations with `ROCS = 0.0`, so an unfiltered arm mixed 2000/5000 zeros
      into quantiles (baseline p99 read 0.430 instead of 0.861). Added `_valid_subset`, used by
      `_arm_objectives` (figure 12/13 baselines) and by figure 16's baseline and delivered arms.
- [x] **03 C / 03 D axes** — `_smiles.tsv` stores the raw TanimotoCombo (0–2), not a
      desirability. Both axes are relabelled and their upper limits now come from the data
      (capped at 2.05); the old 0–1.02 axis clipped the gold run's combo up to 1.09.
- [x] **05 alignment** — the old fallback called `GenerateDepictionMatching2DStructure(mol, ref)`
      with no `refPatt`, which does not align at all, and the seed Murcko core was tested
      before the card set existed. The template is now chosen from exactly the molecules that
      will be drawn: the seed Murcko core when every card contains it, otherwise the MCS of the
      displayed set. Cards are built first, then one template is prepared and every card is
      aligned onto it, so the shared core sits at identical coordinates on all of them
      (verified: min matched-atom RMSD 0.0000 for every card). Cards also render at the axes'
      native resolution (760×540 instead of 340×240, previously upscaled ~2.6×).
- [x] **10 / SDF alignment** — `rdShapeAlign.AlignMol(ref_heavy, lead_heavy)` passed the seed as
      the probe, so the seed moved and the leads never did; the exported SDF was therefore not
      aligned at all. New `rocs.align_to_reference(candidate, reference)` keeps the winning pose
      (best of the four canonical axis flips) and returns it in the reference's canonical
      inertial frame; the seed is canonicalised first, so plot, SDF and score share one frame.
      Verified end-to-end: re-scoring each stored SDF pose with `rdShapeAlign.ScoreMol` against
      the stored seed reproduces its recorded `TanimotoCombo` to 1e-5. Panel B now labels the
      drawn pose's combo separately from the pipeline's best-conformer ROCS, and all three 3D
      panels share one bounding box and box aspect so sizes are comparable. SDF records carry
      `Alignment`, `ROCS`, `TanimotoCombo`, `ShapeScore`, `ColorScore` and the card metrics.
- [x] **13 layout** — a figure-level colorbar beside a 1×3 grid cannot be negotiated by
      `tight_layout`; the grid and colorbar are now placed explicitly.

### Tests (behaviour only, no rendering or filename pins)

- [x] `tests/test_diagnostics_analysis.py` (15 tests): `align_to_reference` reproduces
      `score_pair` and its returned pose re-scores to its own reported combo without mutating
      the input; hypervolume matches hand-computed volumes and ignores NaN/below-reference
      points; `_sa_desirability` respects run bounds and degenerate bounds; `_valid_subset`
      drops unparsable rows; `AlignedDepicter` picks a template shared by every displayed
      molecule and puts matched atoms on the template coordinates.

Status 2026-11-05: executed. Both runs re-rendered in place (13 PNGs each, ≈2 min per run),
`ruff check _david` clean, full suite green.

---

## Round 4 — 2026-11-05, batch B (six single-knob ablations on the gold base)

The two completed arms (`20261104_0445_egcg_gold_oe`, `20261104_0447_egcg_eps0`) agree on the
headline: **the delivered arm does not enrich the ROCS tail** (fig. 16: eps0 EF 0.70×/0.49×/0.55×,
gold 1.20×/1.01×/1.00×) while validity (96 %→60/78 %) and size (HAC 26→14/22) degrade. Round 4
therefore tests *why the policy degrades*, not new objectives, and every arm keeps the gold base so
the six arms are mutually comparable.

Measured facts that fixed the design:

- `loss_train` falls 21→3 in the eps0 run (corr with epoch −0.63) while `avg_gmean` is flat
  (0.722→0.709) and validity drops to 60 %. With `beta=0` the REINFORCE loss is
  `-mean(logp·r)` with every `r>0`, i.e. it measures policy sharpening, not objective progress;
  invalid molecules (reward 0) receive a *weak positive* weight rather than a negative one.
- The reward is already ROCS-dominated in log space: Gompertz ROCS anchors
  (0.40→0.05, 1.20→0.80) over an observed combo band of 0.35–0.65 contribute a log-reward spread
  ≈5× the SA term, and the SA objective is a `ClippedScore(5.0, 3.0)` that the baseline already
  saturates (median raw SA 2.63 → desirability 1.0). There is no live SA or QED pressure to trade
  away; a "ROCS-heavy" arm can only mean shrinking the residual SA weight.
- Per-epoch mean reward measured from the stored populations: 0.336 (eps0), 0.378 (gold) → the
  advantage-centering baseline is `beta ≈ 0.35`.
- Anchors are **not** changed: `AGENTS.md` §3.1 records anchor compression as a solved root cause,
  and the supervisor run that did show a large ROCS gain achieved it with
  `ParetoCrowdingDistance` on the raw combo, which has no anchors at all.

Arms (all `q_1d`, 16 cores, 32 GB, 15 GB scratch, `patience=30`, `min_epochs=50`):

| Run tag | Deltas from `PipelineConfig` | Question |
| :--- | :--- | :--- |
| `egcg_p30_epsdecay` | `--epsilon 0.15 --use-decay --decay-start-epoch 5 --decay-epochs 60` | Does early ε→0 annealing hold validity up while still moving combo? |
| `egcg_p30_bs128` | `--batch-size 128` | Is the collapse driven by 8 Adam steps/epoch rather than the objective? |
| `egcg_p30_s1k` | `--samples 1000 --epochs 100` | Is the per-epoch sampling budget (figs 14/15) the binding constraint? |
| `egcg_p30_pareto` | `--reward-scheme pareto_crowding` | Gold-base version of the supervisor's winning scheme (rank/crowding, no Gompertz map). |
| `egcg_p30_beta35` | `--beta 0.35` | Advantage centering so the loss measures contrast instead of entropy collapse. |
| `egcg_p30_rocs95` | `--weights 0.95 0.05` | ROCS-only reward (SA reduced to a 5 % residual), QED already off. |

New code needed by `egcg_p30_rocs95`: `PipelineConfig.weights` existed but had no CLI surface, so
`--weights` was added to `pipeline.py` (`nargs="+"`, default `None` → auto-calibrated pair, so the
default run is byte-identical). Three behaviour tests pin the contract: the flag reaches
`PipelineConfig.weights`, absence keeps the calibrated defaults, and lopsided weights make the
ROCS term dominate the reward ratio.

Known limitation, accepted by the user: no same-config replicate is in this batch, so a difference
of a few percent between arms is not interpretable as an effect. `egcg_p30_bs128` and
`egcg_p30_pareto` are the cheapest candidates for a replicate in the next batch.

Expected wall clock from the measured 2026-11-04 rate (≈7.2 min/epoch including setup and the
5000-molecule arm): 6–9 h for the single-knob arms, up to ~21 h for `bs128` if it keeps improving to
the 200-epoch cap, and 14–18 h for `s1k` under its 100-epoch cap (still inside `q_1d`).

## Round 5 — 2026-11-05, standard baseline, dashboard, cross-run comparison

Three requests arrived together: (1) turn the 5000 prior molecules that every run samples
into one standardized, scored, reusable dataset; (2) make the output tree addressable
(`YYYYmmdd_HHMM_<note>` RL arms + category folders) and replace the static run table with an
interactive overview; (3) visualize all runs together. Task 2.4 asks for a *fair*
re-evaluation of older arms against the standard baseline, which is compute-heavy and belongs
on MetaCentrum. The tree normalization and figure-05 fix are already done (commit `426d92c`,
`edf9e8d`); `ORGANIZATION_LOG.md` is the audit trail.

### 1. Standard prior dataset (`prior_dataset.py`)

Every RL run spends its first stage sampling `n_eval_samples` (now 5000) molecules from the
*same* untrained prior and scoring them with the run's own environment. That arm is therefore
almost identical across runs, yet it is re-sampled and re-scored every time, and because
`threshold_percentile` optionally calibrates from it, the *training* reward baseline drifts
between runs. Freeze it once:

- `build()` samples the prior with a fixed seed, scores with the run's scorers, and writes
  `SMILES`, `CANONICAL_SMILES`, **raw** `Raw_ROCS`/`Raw_SA`/`QED`, 2D descriptors and the
  Murcko scaffold. Raw values are the contract: any later config re-derives the scaled
  `ROCS`/`SA` deterministically (`ROCS = Raw_ROCS × {1,2}`, `SA = clip((upper − Raw_SA) /
  (upper − lower))`), so the dataset is not tied to one `rocs_use_modifier`/SA window.
- `statistics.json` records validity, uniqueness, property quantiles, Murcko-scaffold counts
  and ECFP4 diversity, so "what is the prior like" is a number, not an impression.
- `manifest.json` pins the query, the reference SDF hash, the sampling seed, the software
  versions, the source revision and SHA-256 of the CSV + statistics.
- Location `_david/output/datasets/<molecule>_prior_<n>_s<seed>/` (a category folder, so the
  normalizer leaves it alone), mirrored to `<storage>/datasets/<name>` by checksum.
- `PipelineConfig.prior_dataset` + `--prior-dataset` make Stage 1 load the frozen arm instead
  of re-sampling; the manifest records the dataset name and hash. A reference mismatch (the
  dataset was built for another molecule) is a hard error, not a silent fallback.

### 2. Interactive overview (`dashboard.py`)

`_david/output/dashboard/index.html`, self-contained (embedded JSON, inline SVG depictions,
no CDN, works offline):

- **Runs** — one row per `YYYYmmdd_HHMM_` RL arm (archive and category folders excluded),
  config normalized to canonical field names, legacy keys (`rocs_color_weight`,
  `energy_window_kcal`) reported as retired instead of silently compared.
- **Clustering** — runs with byte-identical effective configs fold into one node; the rest
  split on the knobs that actually vary, so the tree answers "what did we change between
  these runs". The leaf shows the run name, state and whether a standard-eval exists.
- **Design grid** — reward scheme × criterion, and ε × β, drawn as a matrix with done /
  not-done cells. An empty cell is a *proposal*, not a claim about results.
- **Cross-run statistics** — baseline vs reinforced medians per run, validity/eval size, and
  a global best-of-the-best table built from every stored reinforced arm (desirability =
  ROCS + QED − 0.1·SA, the same ranking figure 05 uses), with the molecule's 2D structure.
- Regenerated by `python -m _david.code.dashboard` and at the end of every pipeline run; a
  dashboard failure is logged and never fails the experiment.

### 3. Standard re-evaluation (`standard_eval.py`)

A run is only comparable if its final policy is sampled at the standard size against the
standard baseline. Artifacts decide what is possible:

| Artifact | Action |
| :--- | :--- |
| `models/SequenceRNN_reinforced.pkg` | reload and sample `n_eval_samples` with a fixed seed → true standard arm |
| only `raw_data/reinforced_raw.csv.gz` | re-score the stored molecules, record `resampled: false` and the smaller n |
| neither | `skip`, with the reason; the dashboard shows it rather than hiding it |

The scored arm lands in `<run>/standard_eval/` with its own manifest (dataset hash, seed,
counts, resampled flag). It never overwrites the historical `reinforced_raw.csv.gz`: the
point is to add a comparable measurement beside the original, not to rewrite history.
`render_run.py` (or `audit_rl_runs --regenerate`) is the standard re-render command for
older runs; it releases only figure/label changes, never research constants.

Heavy runs are submitted from MetaCentrum: one `q_1d` job runs `standard_eval run --all`,
processes the plans sequentially, and rsyncs `standard_eval/` back. Local execution is only
for a single run/smoke test. Budget: ≈1–3 h per 5000-molecule arm, so `--all` fits a 24 h
lane only while the checkpointed set stays small; otherwise split by `--runs`.

### Tests (behaviour only)

`test_prior_dataset` (raw→scaled re-derivation, manifest hash, reference mismatch refusal),
`test_dashboard` (stamped runs discovered, archive excluded, identical configs clustered,
legacy keys reported, HTML escapes a hostile run tag), `test_standard_eval` (artifact →
action classification, no-overwrite contract, rescore path keeps the stored count).
