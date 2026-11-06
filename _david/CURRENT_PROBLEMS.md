# DrugEx thesis — current problems, ranked

**Snapshot:** 2026-11-06. Companion to `_david/ARCHITECTURE.md`.
This is the honest defect list, not a roadmap. Numbers are measured on the arms in
`_david/output/` unless marked otherwise.

---

## 0. Executive summary

| # | Problem | Severity | Evidence | Status |
|---|---|---|---|---|
| P0-1 | The RL objective barely moves ROCS | critical | every arm median 0.655–0.685; supervisor 0.638 → 0.781 | open |
| P0-2 | Success thresholds saturate before training starts | critical | `desired_ratio` = 0.86 at epoch 1 vs 0.14 in the supervisor run | open |
| P0-3 | The checkpoint criterion is blind to ROCS gains | critical | `avg_gmean` best epochs land at 0.69–0.72 while ROCS is flat | open |
| P0-4 | Two incompatible scoring geometries are being compared | critical | EGCG self-score 1.9936 (ours) vs 1.2597 (supervisor) | partially measured |
| P1-1 | No per-epoch ROCS telemetry | high | `_fit.tsv` has no ROCS column; only final arms and cache | open |
| P1-2 | Long runs collapse validity before they improve shape | high | loss arm 0.96 → 0.70; `beta35` final arm 100 % invalid | known, mitigated |
| P1-3 | Two training recipes differ in steps/epoch and epsilon | high | ours 8 steps at ε=0.05; supervisor 4 steps at ε=0.1 | open |
| P1-4 | Best-checkpoint restoration vs final-epoch evaluation | high | supervisor reports final weights; we report best-by-gmean | design choice |
| P2-1 | Standard arms exist for 18/21 arms; 13 lack a recorded protocol | medium | audit CSV `eval_protocol_sha256` | mostly fixed |
| P2-2 | `beta35` archive decision unresolved | medium | stored arm 5000 × invalid sentinel; checkpoint healthy | blocked on user |
| P2-3 | Historical run hygiene | medium | missing manifests, renamed duplicates, empty dirs | partly fixed |
| P2-4 | Cluster ergonomics | medium | Kerberos 2FA, queue waits, rsync overwrite | known |
| P2-5 | Compute cost | medium | 10.4 min/epoch at 16 cores; 21 h full horizon | known |

The single sentence: **the policy is not being asked to improve ROCS, and the
metric used to choose its checkpoint cannot see ROCS anyway.**

---

## 1. What a successful run looks like (reference point)

Supervisor notebook, `tutorial/advanced/rocs/egcg/final_out/eps0.1/executed_eps0.1.ipynb`
(commit `b06ee1a`), one EGCG conformer, 50 epochs × 500 samples, ε=0.1:

| metric | before | after | delta |
|---|---|---|---|
| validity | 0.946 | 0.948 | +0.002 |
| median TanimotoCombo | 0.6384 | 0.7809 | **+0.1425** |
| p90 | 0.7416 | 0.8771 | +0.1355 |
| p99 | 0.8500 | 0.9356 | +0.0856 |
| desired ratio (ROCS ≥ 0.7246, SA ≥ 0.25) | 0.125 | 0.668 | +0.543 |
| median SA (raw) | 2.661 | 2.717 | +0.056 |
| median heavy atoms | 27 | 29 | +2 |
| median MW | 387.6 | 404.0 | +16.3 |

Their threshold is **calibrated on the prior**: ROCS = 85th percentile = 0.7246;
SA = ClippedScore at raw SA 4.5 = 0.25. Reward = `ParetoCrowdingDistance`. They
evaluate the **final-epoch** weights, with `patience` and `reload_interval` disabled.

---

## 2. P0 — the RL does not optimize ROCS

### 2.1 Measurement

Standard arms (5000 molecules, one protocol, valid molecules only):

| run | median ROCS | p90 | frac ≥ 0.75 | frac ≥ 0.85 | reward | ε | criterion |
|---|---|---|---|---|---|---|---|
| prior / loss arm best@1 | 0.676 | 0.775 | 0.182 | 0.021 | harrington | 0.00 | val_loss |
| ctrl_seed42 | 0.685 | 0.782 | 0.185 | 0.020 | harrington | 0.05 | avg_gmean |
| rocs95 best@64 | 0.681 | 0.781 | 0.180 | 0.017 | harrington | 0.05 | avg_gmean |
| h4 with QED | 0.682 | 0.784 | 0.187 | 0.017 | harrington | 0.05 | avg_amean |
| h3 with QED | 0.681 | 0.783 | 0.188 | 0.017 | pareto | 0.05 | avg_amean |
| supervisor reproduction | 0.677 | 0.778 | 0.167 | 0.015 | — | 0.10 | — |
| supervisor notebook | 0.781 | 0.877 | 0.576 | 0.194 | pareto | 0.10 | final |
| `run18` (legacy ε=0) | 0.663 | 0.765 | 0.135 | 0.010 | — | 0 | — |
| `egcg_eps0` | 0.657 | 0.757 | 0.114 | 0.010 | harrington | 0.00 | avg_gmean |

The spread across every modern arm is 0.03 ROCS. That is the prior distribution
plus noise, not optimization. Even the run that reproduces the supervisor's
hyperparameters (`supervisor_reproduction_eps0.1`) lands at 0.677, so the gap is
not epsilon alone.

### 2.2 Root cause candidate 1 — threshold saturation

`PipelineConfig` defaults `thresholds=(0.35, 0.50)` (plus QED 0.50 when enabled).
The DrugEx monitor uses these to compute `desired_ratio` and the means that select
the checkpoint. At epoch 1 our runs report `desired_ratio ≈ 0.86`; the supervisor
reports 0.14 at epoch 1 with a 0.7246 threshold.

Consequences:

- 86 % of the prior already counts as "desired", so the success metric cannot
  reward the tail that matters.
- `avg_gmean` is dominated by saturated objectives and drifts downward as
  validity/novelty fall, so the selected checkpoint is the least-collapsed epoch,
  not the best-shaped one.
- A run can be "successful" by the logged criterion while ROCS never moves.

The Harrington reward does **not** read the threshold tuple; it has its own anchors
(0.40→0.05, 1.20→0.80). So there is a gradient, but it is weak relative to the
variance of the policy-gradient estimate, and the selection signal does not
reinforce it.

**Fix candidates (need a run):**

1. `threshold_percentile=85.0` so thresholds calibrate from the Stage 1 prior
   (the supervisor's measured choice). Keep `harrington` to preserve the gold
   standard, or run a paired arm with `pareto_crowding`.
2. Log ROCS directly in `_fit.tsv` (see P1-1) so the criterion can be audited.

### 2.3 Root cause candidate 2 — scoring geometry

Measured with the current code on EGCG itself:

| protocol | reference | colour | minimization | conformers | EGCG self-score |
|---|---|---|---|---|---|
| ours | 4-conformer diverse ensemble | `oe_mills_dean` | MMFF94s | 20 | **1.9936** |
| supervisor | single MMFF-minimum | default | none | 30 | **1.2597** |

The same molecule scores 1.99 against our reference ensemble because one of the
four reference conformers is an exact match. The supervisor's single-conformer
reference scores 1.26 even for EGCG, because candidate ETKDG poses do not reproduce
that exact conformer. The two numbers are therefore **not on the same scale**, and
the supervisor's +0.14 cannot be compared directly with our +0.00.

What is still unexplained is why our *relative* improvement is smaller even after
rescoring both arms with one protocol. The cross-scoring experiment (1000 unique
valid molecules per arm, both protocols) is running on `dichtator`; it answers:

- if our protocol scores the supervisor's after-arm near 0.78, the policies differ;
- if it scores it near 0.68, our scorer compresses the tail and the comparison is
  an artefact;
- if the supervisor's protocol scores our arm near 0.78, our arm is fine and only
  the measurement differs.

Until that finishes, treat the +0.14 as **not yet attributable**.

### 2.4 Root cause candidate 3 — the reference ensemble itself

Default `max_reference_conformers=4`, `reference_selection_mode="diverse"`.
The supervisor explicitly measured that a single lowest-energy conformer is the
reference for EGCG shape matching, and that 30 candidate conformers saturate the
self-score (10 → 1.1835, 30 → 1.2597, 60 → 1.2597). Our 4-conformer ensemble
changes both the reward landscape and the evaluation scale. It may also make the
task harder: a policy must match *some* low-energy pose of a 4-pose ensemble, and
the diverse poses may reward generic shapes.

This is a scientific choice, not a bug. It needs an explicit ablation:
reference K ∈ {1, 4} × minimization ∈ {on, off} × colour ∈ {default, oe_mills_dean},
scored on one fixed arm.

### 2.5 Root cause candidate 4 — reward scheme

`harrington` is a smooth scalarization. `pareto_crowding` rewards rank and density
in objective space and is what the supervisor used. Our `pareto` arm (`h3`,
`p30_pareto`) lands at 0.677–0.681, so the scheme alone is not sufficient under our
thresholds and reference. It may still be necessary once thresholds are calibrated.

### 2.6 Root cause candidate 5 — optimization pressure

| knob | ours | supervisor |
|---|---|---|
| samples/epoch | 500 | 500 |
| batch size | 64 | 128 (4 steps/epoch) |
| Adam steps/epoch | 8 | 4 |
| epochs | 120 cap | 50 |
| epsilon | 0.05 | 0.10 |
| checkpoint | best by `avg_gmean` | final epoch |
| early stop | min 50, patience 30 | disabled |

Eight steps per epoch doubles the update count relative to the supervisor. If the
policy is collapsing toward a generic drug-like mode, fewer, larger steps with more
exploration may be the difference. Our `bs128` arm (batch 128 → 4 steps) lands at
0.673, and the loss arm (batch 128) at 0.676, so batch size alone is not the cause
either. The interaction with threshold calibration is untested.

---

## 3. P1 — training telemetry and collapse

### 3.1 No per-epoch ROCS

`models/SequenceRNN_reinforced_fit.tsv` records validity, uniqueness, desired ratio,
arithmetic/geometric means, training loss and (for the loss arm) `val_loss`.
It does **not** record ROCS. The only ROCS measurements are:

- the training-molecule cache (`molecules/training_molecules.csv.gz`), 10 % of each
  epoch's samples;
- the final baseline/reinforced arms.

Consequences: a run can be declared "best at epoch N" without anyone knowing whether
mean shape improved at N; early stopping is driven by a criterion that may be
saturated; the supervisor's headline metric is invisible during training.

**Fix:** add mean/median ROCS of the monitor arm to the fit row. The scorer already
returns it; this is a logging change, not a new metric. Until then, any claim that
training improved ROCS must be made from the final arm, never from the trajectory.

### 3.2 Validity collapse and the invalid sentinel

Loss arm `20261106_0326_egcg_loss120_ws`:

| epoch | valid | unique | val_loss | eligible | best_epoch |
|---|---|---|---|---|---|
| 1 | 0.96 | 0.96 | 20.503 | 1 | 1 |
| 5 | 0.90 | 0.90 | 24.408 | 1 | 1 |
| 20 | 0.80 | 0.80 | 27.5 | 0 | 1 |
| 50 | 0.70 | 0.68 | 31.885 | 0 | 1 |

`J` rose monotonically by 55 %; the policy walked away from the prior and the
eligibility gate rejected the collapsed epochs. `best_epoch=1` is therefore a
correct verdict: the RL never beat the one-step policy on held-out NLL. The sign is
right (`val_loss_neg = -J`, maximized), which is proven by epoch 1 winning; a sign
flip would have selected epoch 50.

`beta35` is worse: its stored final arm is 100 % invalid (5000 × the `999…`
sentinel), because the final epoch collapsed at epoch 50. The checkpoint is
`best@3` and resamples to 84 % valid / median ROCS 0.65. The run is currently
represented by the collapsed arm in the archive.

### 3.3 Long horizons

`rocs95` stopped at 95/200 with `best@64`; `ctrl_seed42` at 92 with `best@61`.
Both show validity decay after ~epoch 60. The new 120-epoch cap and patience 30
bound this, but they do not remove the mechanism. Epsilon 0.05 is not enough to
keep the policy from narrowing; epsilon 0.10 is the supervisor's choice and is
untested under our threshold/reference configuration.

### 3.4 Best checkpoint vs final epoch

The pipeline restores the monitor's best checkpoint before Stage 3. The supervisor
evaluates final-epoch weights. These answer different questions:

- best-by-criterion: "what did the selection rule choose?"
- final epoch: "where did the policy end up?"

Our default is defensible only if the criterion tracks the scientific objective.
Given P0-3, it currently does not. The loss criterion makes the distinction explicit
(`evaluation_source=best_epoch`), but the gmean arms do not log whether their best
epoch is meaningfully better than epoch 1.

---

## 4. P2 — evaluation and archive integrity

### 4.1 Standard arms

- 18 of 21 arms have a local `standard_eval/` directory.
- 13 of 18 carry no recorded `protocol` block. They are recomputed from their
  dataset record; `verify` passes because the recomputation matches the recorded
  arms' hash. The audit CSV used to report a phantom second hash because
  `protocol_from_manifest` hashed the dataset manifest instead of the arm's dataset
  record; fixed 2026-11-06 with a regression test.
- `verify` only sees local arms. A remote arm that was never pulled is invisible.
- A run that appears on storage after a `standard_eval submit` plan is not scored
  by that job; a fresh submission is required (`rocs95` and the loss arm are in
  this state as of 2026-11-06).

### 4.2 Run inventory

- 24 run directories discovered; 20 complete and verifiable.
- Some historical runs have no `manifest.json`; the audit still discovers them via
  `SequenceRNN_reinforced_fit.tsv` and marks them accordingly.
- Renamed/duplicated copies exist in the archive; the deduplicator folds them by
  fingerprint, but the displayed name depends on the preference rules.
- `20261104_0446_egcg_to60` was an empty shadow directory and was archived via
  `output_organization apply`; the move is in `ORGANIZATION_LOG.md`.

### 4.3 Molecule cache

Every run now carries `molecules/training_molecules.csv.gz` + JSON provenance
(schema version, hash, row count, reference, ROCS scale, SA window, timestamp).
It is a **candidate-discovery** artifact (10 % of each epoch), not the distributional
arm. Do not compute thesis statistics from it; use `standard_eval/` or the final
`reinforced_raw.csv.gz`.

### 4.4 `beta35`

The stored arm is invalid; the checkpoint is healthy. Two honest options:

1. label the run as collapse evidence and keep the stored arm;
2. regenerate the arm from `best@3` and record `evaluation_source`.

The standard arm already carries the healthy model (median 0.666), so option 1 does
not lose the measurement. User decision pending.

---

## 5. P2 — infrastructure and cost

### 5.1 MetaCentrum

- Kerberos 2FA expires; every status/cancel/pull/submit then fails with a clear
  message. Work stops until `ssh meta kinit …`. No non-interactive fix.
- `q_1d` drains ~2× faster than `q_2d` (1435 vs 764 running jobs on 2026-11-06);
  a `q_1d` job waited ~6 h, a `q_2d` job started in ~1 min.
- 16 cores reached the 32 GB memory ceiling (`resources_used.mem ≈ 31.2 GB`), hence
  `MEM_GB_PER_CORE=2` and the 32-core/64 GB recipe for full-horizon runs.
- `job.py` validated releases against `_david/code/pipeline.py` after the module
  moved to `training/pipeline.py`; every job died 37 s in with "Release is incomplete
  or gone". Fixed 2026-11-06 (`567add3`) with a path test.
- `pull` uses `rsync -az` and overwrites locally regenerated files. Pull only missing
  run names. The `pull_standard_arms` resurrection bug (archived runs reappearing as
  manifest-less stubs) is fixed with a test.

### 5.2 Compute

- Measured 10.4 min/epoch at 16 cores for a minimised 500-sample epoch.
- Full 120-epoch horizon ≈ 21 h; 150 epochs ≈ 26 h (does not fit 24 h).
- dichtator: 48 cores / 125 GB, shared; WSL: 4 cores / 4 GB.
- ROCS scoring is CPU-bound and dominates epoch time; the RNN itself is tiny. More
  cores help linearly until memory bandwidth and stragglers bind.
- GPU is unused by the scoring stack; `gpu` only affects torch training.

---

## 6. Negative results already established (do not repeat)

1. **QED objective conflicts with shape.** EGCG QED = 0.21; optimizing QED strips
   rings. H4 with QED still lands at 0.682, so it is not catastrophic under
   Harrington, but it spends capacity on the wrong objective. Default stays off.
2. **Smooth SA modifier pays for shrinkage.** The old `SmoothClippedScore(5,3)`
   keeps rewarding lower SA; the 250-epoch run shrank heavy atoms 27 → 20 and MW
   384 → 266. `ClippedScore(5,3)` is flat below raw SA 3 and is the production
   setting.
3. **ε = 0 collapses.** `run18` and `egcg_eps0` are the two worst modern arms
   (0.657–0.663). Epsilon 0 is not a valid production setting.
4. **Val-loss criterion on this task is anti-correlated with progress.** Held-out
   NLL rises monotonically while validity falls; best epoch is 1. The criterion is
   a valid anti-overfit gate, but it cannot select a shape-improving policy.
5. **Larger batch alone does not fix ROCS.** `bs128` (0.673) and the loss arm
   (0.676) are inside the noise of the 0.68 cluster.
6. **Pareto alone does not fix ROCS.** `p30_pareto` (0.677) is inside the cluster.

---

## 7. Hypotheses, ranked, with discriminating experiments

| rank | hypothesis | test | kill criterion |
|---|---|---|---|
| H1 | Calibrated thresholds (85th percentile) unlock the tail | paired 50-epoch runs: fixed 0.35 vs p85, same seed/reference | p85 median ≤ fixed median + 0.02 |
| H2 | Our scoring geometry compresses/reshapes the reward | cross-score 1000 molecules per arm under both protocols | protocol accounts for < 0.02 of the 0.14 gap |
| H3 | Our reference ensemble (K=4) makes the task harder/different | K ∈ {1,4} ablation with fixed arm and scorer | K=1 median ≤ K=4 median |
| H4 | 8 Adam steps/epoch over-update the policy | batch 64 vs 128 at fixed 500 samples, 50 epochs | no difference in median or validity decay |
| H5 | ε=0.10 + final-epoch evaluation is the supervisor's real advantage | replicate exactly under our scorer, then swap one knob at a time | replication still ≤ 0.70 |
| H6 | Harrington's weak tail gradient is the limiter | paired harrington vs pareto with calibrated thresholds | both within 0.02 |

Experiment matrix (one MetaCentrum/dichtator job per row, 50 epochs, 500 samples,
one seed, standard 5000-molecule arm):

| id | thresholds | reward | ε | batch | reference | purpose |
|---|---|---|---|---|---|---|
| A | p85 | harrington | 0.05 | 64 | K=4 | our standard + calibration |
| B | p85 | pareto | 0.10 | 128 | K=1 | supervisor-like |
| C | p85 | harrington | 0.10 | 128 | K=1 | isolate protocol |
| D | 0.35 | harrington | 0.05 | 64 | K=1 | isolate reference |
| E | p85 | harrington | 0.05 | 64 | K=4 | duplicate seed for noise |

Acceptance for "we closed the gap": median ROCS ≥ 0.75 on the standard arm with
validity ≥ 0.85 and uniqueness ≥ 0.50, reproduced on two seeds. Anything below
0.72 is not a result.

---

## 8. Decisions needed from the user

1. **Reference policy:** keep the 4-conformer diverse ensemble (current gold) or
   adopt the supervisor's single lowest-energy conformer for EGCG-comparable runs?
2. **Threshold policy:** move the default to a prior-calibrated percentile (85) or
   keep fixed 0.35/0.50 for historical comparability?
3. **Checkpoint policy:** continue best-by-gmean, switch to `desired_ratio` with
   calibrated thresholds, or evaluate final weights as the supervisor does?
4. **`beta35`:** label as collapse evidence or regenerate from `best@3`?
5. **Run budget:** the matrix above is five 50-epoch arms. On 16 cores that is
   ~9 h each; on 32 cores ~4.5 h each. Queue them on MetaCentrum or run on
   dichtator overnight?

Until 1–3 are answered, new arms are not comparable to each other and the +0.14
question cannot be closed.

---

## 9. Reproduction pointers

- Standard-arm table: `_david/output/RL_RUN_AUDIT.csv` (one row per run, config vs
  gold, provenance hashes).
- Per-run gold conformance: `_david/code/RL_RUN_AUDIT.md`.
- Loss-criterion verdict: `_david/output/20261106_0326_egcg_loss120_ws/models/`
  (`loss_criterion.json`, `loss_holdout.csv.gz`, `SequenceRNN_reinforced_fit.tsv`).
- Supervisor evidence: notebook `executed_eps0.1.ipynb` and its
  `provenance.json`, `comparison.csv`, `before_raw.csv.gz`, `after_raw.csv.gz`
  (commit `b06ee1a`).
- Cross-scoring outputs: `/home/kolarv/xdata/crossscore/summary.csv` on dichtator
  (when the run finishes).
- Conformer findings and open decisions: `_david/code/conformer/results.md`.
- MetaCentrum recipes: `_david/code/metacentrum/RUNBOOK.md`, `help/05-resources.md`.

