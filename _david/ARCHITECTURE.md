# DrugEx thesis — architecture and current state of the art

**Snapshot:** 2026-11-06. Authoritative for the code under `_david/`.
**Read this first, then `_david/CURRENT_PROBLEMS.md`.** This file describes what exists
and why; the problems file describes what is broken or unexplained.

---

## 1. Scope and hard invariants

- Thesis code lives only under `_david/`. `drugex/` is upstream core and is read-only.
- One run = one self-describing directory under `_david/output/`.
- One configuration object, `_david/code/config.py::PipelineConfig`, is the only source
  of defaults. CLI flags mirror it; they never invent a value.
- The scoring stack is molecule-agnostic. EGCG is the default query, not a code path.
- Scientific comparability is carried by a protocol hash, not by filenames or dates.
- Tests cover behaviour and contracts only. No test pins a chosen hyperparameter, a
  figure filename or a label format.
- Lint gate: `ruff check --no-cache _david` must be clean. Test gate:
  `python -m unittest discover -s _david/code/tests -t .` must be green.
- WSL (`/home/kolar/DrugEx`) is the source of truth. `dichtator` is the lab GPU/CPU
  server; MetaCentrum is ad-hoc PBS compute. Sync only through the private `dichtator`
  git remote; never push GitHub `origin`, never force push.

---

## 2. Execution lanes

| Lane | Use | Entry point |
|---|---|---|
| WSL workstation | editing, lint, unit tests, light analysis, 4 cores / 4 GB | `python -m _david.code.training.pipeline` |
| `dichtator` | interactive training, GPU, regeneration, cross-scoring | `ssh dichtator`, tmux, `~/DrugEx/.venv` |
| MetaCentrum | real cluster arms, CPU-heavy ROCS/conformer batches | `python -m _david.code.metacentrum.submit` |

MetaCentrum jobs are reproducible because `submit.py` streams `git archive HEAD` to
`releases/<sha>/`, and the PBS body copies that release into `$SCRATCHDIR/DrugEx`.
Local edits after `qsub` cannot change a queued or running job.

---

## 3. Repository map (`_david/code/`)

```
config.py            PipelineConfig, PROJECT_ROOT, OUTPUT_ROOT, RunPaths
schema.py            std.* canonical column and constant names
chemistry.py         standardization, ETKDG embedding, MMFF minimization,
                     CandidateConformerGenerator (parallel, identity-checked)
rocs.py              CanonicalROCSScorer, create_rocs_scorer, score_pair,
                     WholeCallScorer (multiprocessing wrapper)
reference.py         PrepareMolecule: reference standardization, isomer
                     enumeration, embedding, energy window, Butina RMSD,
                     diverse/rmsd ordering, reference SDF writer
prior_dataset.py     frozen prior dataset: build, load, raw<->scaled ROCS,
                     descriptors, statistics, resolve_dataset, sha256_file

training/
  pipeline.py        5-stage MoleculePipeline + CLI
  environment.py     HarringtonDesirabilityReward, resolve_reward_scheme,
                     create_environment, PrepareMolecule re-export
  models.py          DecayingSequenceExplorer, LossSelectedSequenceExplorer,
                     SequenceRNNOptimizer, explorer factory,
                     restore_best_checkpoint
  loss_criterion.py  frozen held-out sample, reward-weighted NLL J(theta),
                     loss_criterion.json / loss_holdout.csv.gz

analysis/
  visualization.py   4.3k lines: ResultAnalyzer + every publication figure
  dashboard.py       per-run and root HTML dashboards, discover_runs
  standard_eval.py   standard arm planner/runner/verifier, protocol hash,
                     PBS submit/pull/push
  rocs_scale.py      RAW_ROCS -> TanimotoCombo rescaling verdicts,
                     correct_run_frame, scale record
  output_organization.py  run taxonomy and archiving (plan/apply)

run_audit/
  audit_rl_runs.py   discovers runs, folds duplicates, compares config with
                     PipelineConfig(), checks figure suite, regenerates,
                     writes RL_RUN_AUDIT.md and RL_RUN_AUDIT.csv

conformer/          frozen v2 conformer suite (contracts, cache, prepare,
                     ensemble, scoring, studies, analysis, CLI, results.md)
metacentrum/        submit.py, job.py, pbs_launcher.sh, RUNBOOK.md, help/
tests/              19 test modules, ~278 tests
archive/            historical code, never imported by production
```

`_david/code/__init__.py` re-exports the public names. Direct module imports use the
dotted path (`_david.code.training.pipeline`, `_david.code.analysis.standard_eval`).

---

## 4. Configuration (`PipelineConfig`, frozen dataclass)

Defaults as of 2026-11-06. Every field is read where it is used; there is no
kwargs-projection layer.

### 4.1 RL schedule and evaluation

| Field | Default | Meaning |
|---|---|---|
| `epochs` | **120** | cap; early stopping normally binds first |
| `patience` | 30 | epochs without improvement after `min_epochs` |
| `min_epochs` | 50 | earliest allowed stop |
| `n_samples` | 500 | training molecules per epoch |
| `val_samples` | 100 | per-epoch monitor validation arm |
| `n_eval_samples` | 5000 | Stage 3 evaluation arm (gold standard) |
| `batch_size` | 64 | Adam steps per epoch = ceil(500/64) = 8 |
| `criteria` | `avg_gmean` | checkpoint criterion (`avg_amean`, `desired_ratio`, `valid_ratio`, `unique_ratio`, `val_loss` also valid) |
| `beta` | 0.0 | baseline-policy penalization (unused in current arms) |

### 4.2 Exploration

| Field | Default | Meaning |
|---|---|---|
| `epsilon` | 0.05 | per-token prior mutation rate |
| `epsilon_end` | 0.0 | terminal rate |
| `use_epsilon_decay` | False | cosine/linear decay off by default |
| `decay_start_epoch` | 1 | decay start |
| `decay_epochs` | 50 | decay horizon |

### 4.3 Objectives and reward

| Field | Default | Meaning |
|---|---|---|
| `include_qed` | False | QED is measured by standard eval, not optimized by default |
| `thresholds` | `(0.35, 0.50)` / `(0.35, 0.50, 0.50)` with QED | success thresholds for `desired_ratio` and `avg_gmean` |
| `threshold_percentile` | None | if set, thresholds calibrate from the Stage 1 prior |
| `weights` | `(0.75, 0.25)` / `(0.50, 0.25, 0.25)` | Harrington objective weights |
| `reward_scheme` | `harrington` | `pareto_crowding`, `weighted_sum` also valid |
| `score_variant` | `tanimoto_combo` | only production value; Tversky is calibration-only |
| `color_ff` | `oe_mills_dean` | OpenEye ImplicitMillsDean SMARTS port |
| `rocs_use_modifier` | False | raw TanimotoCombo on [0, 2] |
| `rocs_clip_upper` | 2.0 | physical ceiling if modifier is ever enabled |
| `sa_score_lower` | 5.0 | raw SA ≥ 5 → desirability 0 |
| `sa_score_upper` | 3.0 | raw SA ≤ 3 → flat desirability 1 |

Harrington geometry (`environment.py`):

```
d_j(s) = exp(-exp(-(b0_j + b1_j * s)))
D(x)  = exp( sum_j w_j ln(d_j + 1e-6) / sum_j w_j )
ROCS anchors (raw): (0.40, d=0.05) and (1.20, d=0.80)
SA anchors (modified): (0.20, 0.05) and (0.80, 0.90)
QED anchors: (0.30, 0.05) and (0.70, 0.90)
```

The threshold tuple is **not** used by the Harrington reward. It is used by the
DrugEx monitor to compute `desired_ratio` and the geometric/arithmetic means that
select the checkpoint. This distinction is the subject of `CURRENT_PROBLEMS.md` §2.

### 4.4 3D chemistry

| Field | Default | Meaning |
|---|---|---|
| `max_conformers` | 20 | candidate conformers per molecule |
| `max_isomers` | 4 | independent cap for unresolved stereochemistry |
| `max_heavy_atoms` | 45 | pre-filter |
| `max_rotatable_bonds` | 15 | pre-filter |
| `minimize_candidates` | True | MMFF94s minimization at scoring time |
| `optimization` | `MMFF94s`, dielectric 78.4 | force field profile |
| `max_reference_conformers` | 4 | reference ensemble cap K |
| `max_reference_isomers` | 1 | reference isomer cap |
| `reference_selection_mode` | `diverse` | alternative: `rmsd` |
| `reference_rmsd_cutoff` | 0.35 Å | Butina cutoff |
| `energy_window_kcal` | 8.0 | reference conformer energy window |
| `reference_attempts` | 100 | embedding attempts |
| `timeout_seconds` | 30 | per-molecule ETKDG cap |
| `random_seed` | 42 | scoring/conformer seed |

### 4.5 Loss criterion

| Field | Default | Meaning |
|---|---|---|
| `val_holdout_size` | 512 | frozen prior molecules |
| `val_holdout_seed` | 20261106 | held-out draw seed |
| `val_holdout_dataset` | None | falls back to `prior_dataset` |
| `val_min_valid_ratio` | 0.85 | eligibility floor |
| `val_min_unique_ratio` | 0.50 | eligibility floor |

`criteria="val_loss"` requires `prior_dataset` or `val_holdout_dataset`; the
dataclass raises at construction otherwise.

### 4.6 Hardware

| Field | Default |
|---|---|
| `gpu` | `auto` |
| `cpu_ratio` | 0.99 |
| `conformer_threads` | 0 (all cores inside a worker; set 1 when a process pool already saturates the machine) |

---

## 5. Run lifecycle (5 stages)

`MoleculePipeline.run()` → `_run()` in `training/pipeline.py`.

**Stage 0 — reference ensemble.** `PrepareMolecule.generate_reference_conformers()`
standardizes the query, enumerates unresolved stereoisomers, embeds ETKDGv3
conformers, minimizes with the configured force field, filters by molecular identity
and the energy window, prunes by Butina RMSD, orders by stereo round-robin and
shape/colour diversity, and writes `raw_data/query_molecule_3d.sdf`. Default output:
4 conformers of one EGCG stereoisomer. `manifest.reference_sha256` is recorded.

**Stage 1 — prior arm.** Loads the Papyrus pretrained SMILES-RNN and vocabulary,
samples `n_eval_samples` molecules (`raw_data/baseline_raw.csv.gz`). If
`prior_dataset` is set, the frozen dataset's `prior_molecules.csv.gz` is reused and
only re-scored; the dataset hash and reference hash must match.

**Stage 2 — RL.** `SequenceRNNOptimizer.build_explorer()` creates the agent from the
same pretrained package, a frozen `mutate` prior (a separate model instance; aliasing
it to the agent would make epsilon inert), and the environment. `FileMonitor` writes
`models/SequenceRNN_reinforced_fit.log`, `_fit.tsv`, `_smiles.tsv`, checkpoints and
the best-state package. `patience`, `min_epochs`, `criteria` and
`reload_interval=epochs+1` are passed to `explorer.fit`.

After fitting, the final-epoch package is always written
(`SequenceRNN_reinforced_final_epoch.pkg`). `restore_best_checkpoint()` then loads the
monitor's best package back into the agent, and records
`manifest.evaluation_provenance` (`evaluation_source`, `best_epoch`, hashes).
The Stage 3 arm therefore describes the **selected** model, not the last epoch.

**Stage 3 — reinforced arm.** Samples `n_eval_samples` molecules from the restored
agent and scores them with the same environment:
`raw_data/reinforced_raw.csv.gz`.

**Stage 4 — deliverables.** `ResultAnalyzer.analyze()` renders the figure suite,
scorecards, lead tables, the molecule cache and the run summary. The dashboard is
refreshed best-effort; a reporting failure must not mark a finished experiment failed.

### Run directory

```
_david/output/YYYYmmdd_HHMM[_tag]/
  manifest.json          run_id, state, host, full config, revision, hashes
  submission.json        MetaCentrum only: exact args and release
  execution.log          MetaCentrum only: live pipeline stdout/stderr
  models/                checkpoints, monitor telemetry, per-epoch TSV, loss holdout
  raw_data/              reference SDF, baseline arm, reinforced arm, rocs_scale.json
  figures/               publication figures + RENDER_PROVENANCE.json
  molecules/             training_molecules.csv.gz + JSON provenance
  logs/                  pipeline log
  summary.md, comparison_scorecard.csv, top_prioritized_leads.tsv
  standard_eval/         comparable arm: reinforced_standard.csv.gz + manifest + statistics
```

`RunPaths` (`config.py`) is the only definition of that layout.

---

## 6. Scoring stack

### 6.1 Reference

- Query standardization via `chemistry.standardize_smiles`.
- Stereoisomer enumeration is capped independently of the candidate cap.
- `energy_window_kcal=8.0` keeps low-energy poses; identity changes are rejected.
- `diverse_order` symmetrizes directional shape+colour and maximizes dissimilarity,
  with a stereo round-robin prefix so no isomer is starved.
- The SDF therefore contains K conformers (default 4). EGCG self-score under this
  ensemble with default scoring is **1.9936 / 2.0**; under a single MMFF-minimum
  conformer it is **1.2597 / 2.0**. This is a scale difference, not a bug.

### 6.2 Candidates

`CandidateConformerGenerator` (in `chemistry.py`) subclasses DrugEx's RDKit
generator:

- ETKDGv3 with an explicit per-molecule timeout;
- `max_isomers=4`, `max_heavy_atoms=45`, `max_rotatable_bonds=15` pre-filters;
- chunked `multiprocessing` embedding with `params.numThreads=1` inside workers
  (when `conformer_threads=1`) or RDKit's own threads when `0`;
- optional MMFF94s minimization with the configured dielectric;
- identity check after minimization; rejected conformers are logged and skipped;
- per-molecule SDF cache; `genConformers` returns the SDF path.

### 6.3 ROCS

`CanonicalROCSScorer` wraps DrugEx's `RDKitROCSScorer`:

- `score_type="TanimotoCombo"` → shape + colour, `opt_param=0.5` (align-and-score);
- `use_colors=True`, `color_ff="oe_mills_dean"`;
- `rocs_use_modifier=False` → raw score on [0, 2];
- `getScores` returns an `(N, 1)` numpy array; `DrugExEnvironment.getScores`
  returns a DataFrame with the scorer key;
- `WholeCallScorer` runs one whole call in a worker pool so a large arm does not
  pay per-molecule IPC;
- `score_pair` returns directional shape/colour overlaps for reference curation.

`analysis/rocs_scale.py` corrects legacy arms whose `ROCS` column was a normalized
[0, 1] score. The verdict is recorded in `raw_data/rocs_scale.json`
(`scale`, `clip`, `normalized`, counts at the clip). All current arms are `raw`
scale 1.0; `run18` is the only legacy normalized run.

### 6.4 Reward and checkpoint selection

- `harrington` (default): continuous Gompertz desirability, weighted geometric mean.
- `pareto_crowding`: DrugEx rank/density reward. Used by the supervisor notebook and
  by several historical arms.
- `weighted_sum`: linear blend, kept for replay.

The DrugEx monitor computes per-epoch `valid_ratio`, `unique_ratio`, `desired_ratio`,
`avg_amean`, `avg_gmean`. The explorer keeps the epoch maximizing the configured
criterion. `desired_ratio` and the means are computed from the threshold tuple, so
they are only informative when the thresholds sit in the arm's actual score range.

### 6.5 Loss criterion (`criteria="val_loss"`)

`loss_criterion.py` draws a frozen sample of `val_holdout_size` prior molecules
(seed `val_holdout_seed`), weights them by their prior reward, and evaluates

```
J(theta) = - sum_i w_i log pi_theta(x_i) / sum_i w_i     (lower is better)
```

`LossSelectedSequenceExplorer` records `J` as `val_loss`, its negation as
`val_loss_neg`, and an eligibility flag. The base explorer maximizes
`val_loss_neg`, i.e. minimizes `J`. Ineligible epochs keep a finite `val_loss` but
write `NaN` to `val_loss_neg`, so they can neither win nor be logged as `-inf`.
`best_value` starts at `-inf`.

Artifacts: `models/loss_criterion.json` (dataset and sample provenance),
`models/loss_holdout.csv.gz` (the exact scored sample and weights). Both are written
before the first epoch.

---

## 7. Standard evaluation arm

Purpose: one comparable measurement beside each run's historical arm.

- Dataset: `_david/output/datasets/<name>/` with `prior_molecules.csv.gz`,
  `manifest.json` (hashes, scoring block, build config).
- Default: `egcg_prior_5000_s20261105`.
- `_standard_config(dataset)` fixes `include_qed=True`,
  `rocs_use_modifier=False`, SA window from the dataset, `gpu="auto"`.
- `STANDARD_REWARD_SCHEME="harrington"`,
  `STANDARD_REWARD_WEIGHTS=(0.50, 0.25, 0.25)`,
  `STANDARD_REWARD_THRESHOLDS=(0.35, 0.50, 0.50)`.
- `scoring_protocol()` hashes dataset identity, reward definition and every config
  field except `EXECUTION_ONLY_FIELDS`. Training-only knobs (`epochs`, `patience`,
  `n_samples`, `batch_size`, `epsilon*`, `criteria`, `beta`, `prior_dataset`,
  `val_holdout_*`, `training_seed`, hardware/thread fields) are excluded so the same
  science on different hardware keeps one hash.
- `plan_run(run)` decides `resample` (checkpoint), `rescore` (stored arm only),
  `done`, or `skip`.
- `run` writes `standard_eval/reinforced_standard.csv.gz`, `manifest.json`
  (protocol block, checkpoint hash, arm hash), `statistics.json`.
- `verify` re-hashes every listed file, recomputes the protocol from the *dataset
  manifest* and the arm's *dataset record*, and reports recorded-vs-recomputed
  drift, arm-size mismatches and hash mismatches.
- `submit` stages HEAD, plans `--all` against storage, and submits a PBS job;
  `pull` copies arms back only for runs that exist locally (archived runs are not
  resurrected as manifest-less stubs).

Standard arm columns: `SMILES, Valid, ROCS, SA, QED, Desired, Canonical_SMILES, HAC,
MW, LogP, TPSA, HBD, HBA, RotBonds, Fsp3, Rings, Raw_SA, Scaffold, Raw_ROCS,
StandardReward`.

---

## 8. Audit, figures, dashboard, archive

`run_audit/audit_rl_runs.py`:

- discovers every `manifest.json` and every `SequenceRNN_reinforced_fit.tsv`
  (skips `standard_eval/` subdirectories);
- folds byte-identical copies, preferring live over archived, more figures, has a
  manifest, deeper path, descriptive name;
- compares `DECISION_FIELDS` with `PipelineConfig()` (the gold standard);
- checks `CURRENT_FIGURES` (`01,02,03,05,06,07,10,11,12_training_diagnostics`) and
  support files;
- `--regenerate` rebuilds missing figures and the molecule cache from stored arms;
- writes `_david/code/RL_RUN_AUDIT.md` and `_david/output/RL_RUN_AUDIT.csv`.
- Provenance fields in the CSV: `eval_source` (`standard_arm`, `historical_arm`,
  `missing_arm`), `eval_n`, `eval_arm_sha256`, `eval_protocol_sha256`,
  `eval_checkpoint_sha256`.

`analysis/visualization.py` (`ResultAnalyzer`) produces the figure suite: before/after
distributions, multi-objective trade-offs, training trajectories, scaffold diversity,
3D superposition, property trajectories, merged training diagnostics, ROCS enrichment,
stability and sampling budget. `RENDER_PROVENANCE.json` records what was rendered.

`analysis/dashboard.py` builds a root overview and per-run dashboards.

`analysis/output_organization.py` defines the archive taxonomy. Policy: a run that is
not 100 % to standard and not already archived goes to
`_david/output/archive/un_standardize/`; deliberate off-standard evidence arms stay
in `output/`. Every move is logged in `_david/output/ORGANIZATION_LOG.md`.

---

## 9. MetaCentrum subsystem

`metacentrum/submit.py` + `job.py` + `pbs_launcher.sh`.

- `JobSpec` validates name, cores, memory, walltime, scratch, queue, GPU.
- `MEM_GB_PER_CORE = 2`: `--ncpus 32` requires `--mem 64`; a starved pairing raises
  `ValueError`.
- Queue routing by walltime: ≤2 h `q_2h`, ≤4 h `q_4h`, ≤24 h `q_1d`, ≤48 h `q_2d`,
  ≤1 week `q_1w`. Never shorten the request to reach a shallower queue.
- Defaults: 16 cores, 32 GB, 15 GB scratch, 24 h, `spec=8.0`, no GPU.
- `submit` refuses a dirty tracked tree, streams `git archive HEAD` to
  `releases/<sha>/`, repoints `current`, refreshes `models/`, writes
  `jobs/<name>/spec.json`, and calls `qsub` with the release and template.
- Template owns `--output-dir`, `--agent-path`, `--vocab-path`; passing them is
  rejected.
- `job.py` (stdlib-only PBS body) requires `PBS_JOBID, SCRATCHDIR, STORAGE_DIR,
  RUN_SPEC`, validates the spec's release against the launcher, copies the release
  into scratch, runs the pipeline, mirrors output to the PBS log, copies results back
  on success or SIGTERM, and cleans scratch.
- `PIPELINE_ENTRY = _david/code/training/pipeline.py` is the release-completeness
  check; a test asserts it exists.
- `pull` is MetaCentrum → WSL only. `rsync -az` overwrites locally regenerated files;
  pull only missing run names. Verify with
  `metacentrum/archive/file_folder_tree_comparasion/compare_output_trees.py` before
  deleting anything remote.

Commands:

```bash
python -m _david.code.metacentrum.submit submit --name <name> [--ncpus --mem --walltime --args "..."]
python -m _david.code.metacentrum.submit status
python -m _david.code.metacentrum.submit logs <name> -f
python -m _david.code.metacentrum.submit cancel <name>
python -m _david.code.metacentrum.submit pull [--name <run>]
```

Kerberos tickets need an interactive 2FA refresh: `ssh meta kinit kolar...@META`.
The CLI detects expiry and prints the fix; it cannot refresh non-interactively.

---

## 10. Tests and quality gates

```bash
cd /home/kolar/DrugEx
python -m unittest discover -s _david/code/tests -t .
ruff check --no-cache _david
```

Test modules cover planning, protocol hashes, discovery/deduplication, job specs,
submission payloads, PBS body orchestration, loss criterion, pipeline architecture,
molecule cache, conformer contracts, studies, visualization provenance, ROCS scale.

Rules:

- behaviour and contracts only;
- no pinning of hyperparameter values, figure filenames or label formats;
- every fixed regression gets a test whose docstring says what broke and when.

---

## 11. Commands

```bash
# training (WSL/dichtator)
python -m _david.code.training.pipeline --run-tag <tag> \
  --epochs 120 --criteria avg_gmean --epsilon 0.05 --include-qed \
  --prior-dataset _david/output/datasets/egcg_prior_5000_s20261105 \
  --gpu cpu

# loss-criterion arm
python -m _david.code.training.pipeline --run-tag <tag> \
  --epochs 120 --criteria val_loss --epsilon 0.0 --include-qed \
  --batch-size 128 --patience 20 --min-epochs 50 \
  --val-holdout-dataset <dataset> --val-holdout-size 512 \
  --val-holdout-seed 20261106 --prior-dataset <dataset> --gpu cpu

# standard evaluation
python -m _david.code.analysis.standard_eval plan
python -m _david.code.analysis.standard_eval run --all
python -m _david.code.analysis.standard_eval verify
python -m _david.code.analysis.standard_eval submit --dataset egcg_prior_5000_s20261105
python -m _david.code.analysis.standard_eval pull

# audit and reporting
python -m _david.code.run_audit.audit_rl_runs --write _david/code/RL_RUN_AUDIT.md \
  --csv _david/output/RL_RUN_AUDIT.csv
python -m _david.code.analysis.dashboard
python -m _david.code.analysis.output_organization plan|apply
```

---

## 11.1 Current inventory (2026-11-06)

- 24 run directories discovered; 20 complete and verifiable.
- 21 production/historical RL arms; 18 have a local standard arm.
- Newest arms: `20261105_1013_egcg_p30_rocs95` (best@64, early stop at 95/200),
  `20261106_0326_egcg_loss120_ws` (loss criterion, best@1, negative result),
  `20261106_*_egcg_loss120_meta` (cluster twin, cancelled 2026-11-06).
- Standard protocol: all verifiable arms share one hash after the
  execution-only-field fix; 13 legacy arms carry no recorded protocol block and are
  recomputed from their dataset record.
- ROCS scale: every current arm is raw; only `run18` is legacy-normalized.

---

## 12. What the architecture does **not** yet guarantee

- It does not guarantee that the RL objective actually moves ROCS; see
  `CURRENT_PROBLEMS.md` §2.
- It does not guarantee that `desired_ratio`/`avg_gmean` are informative at the
  configured thresholds; they are saturated in every current arm.
- It does not make historical arms comparable to standard arms without the
  `standard_eval/` directory; the raw `raw_data/` arms used different scorers.
- It does not provide a per-epoch ROCS trajectory in the training telemetry; ROCS is
  measured in the training-molecule cache and the final arms, not in `_fit.tsv`.
- It does not automatically push a finished run to MetaCentrum storage or back.
