# Conformer study: status and evidence

**2026-10-29: full drug-panel calibration completed and verified.**
Read the [single maintained report](../../output/20261028_2145_joint_calibration/REPORT.md)
and [final retained-reference figure](../../output/20261028_2145_joint_calibration/reference_caps.png).
All 512 inputs are accounted for: 509 scored, three native-colourless exclusions,
13 preparation failures retained as zero. Verified 18,432 measurement rows,
92,160 retained-cap reductions, 68,347 saved-geometry identity checks, 12 public
replays and 48 complete factory calls; maximum replay difference 1.12e−16.
Main-checkout validation passed 36 tests, scoped Ruff and shell syntax checks.

**Main methodological finding:** 0.5 Å reference RMSD pruning can lose important
ROCS colour matches. Fosfomycin scores 0.7085 with one pruned reference versus
1.0000 with its three saved unpruned references, using the same candidates.
Increasing candidate requests 10→40 did not repair this example. Twenty-four
scored inputs at C10 lose more than 0.01 through this reference pruning.
Recommend evaluating reference-only shape/colour-aware selection before freezing
an RL preset; that method is proposed, not silently implemented.

Best tested cost/mean baseline above 0.95: MMFF94s ε78.4, reference preparation
100 requests/stereoisomer and stereo cap8, all post-filter references at RMSD0.5;
candidate C10/I4, additional pruning off. Mean0.9551, 442/509 at least0.95;
component mean3.460s versus R50/C20/all mean0.9552 at4.780s. Among496 prepared
inputs its mean is0.9802; primary results still include13 zero-score failures.
No universal small fixed reference bank or optimal stereo cap4 is established.

User-selected validity policy is now shared by reference/RL paths: reject only
invalid conformers, preserve valid siblings and the input, count identity rejects
separately (−2). Seven reference and five candidate identity rejections occurred
across budgets. Eight affected inputs were recomputed;504 unchanged checkpoints
were retained with explicit source/provenance. Hydrogen normalization preserves
E/Z and isotope checks. Pentazocine's failed stereo embeddings caused the long
tail (41/85min reference preparation; C20 generation94s); it was not dropped.

All active code is in the main checkout's `_david/code/`. Run artifacts are under
`_david/output/20261028_2145_joint_calibration/`, with report/figure/progress at
root, data/provenance in `raw_data/`, logs in `logs/`. All17 cached worktrees were
removed. Native user edits were preserved. Candidate minimization remains OFF
by default; no RL training has been launched. Historical evidence below is not
an instruction to restore old methods or removed execution paths.

**2026-10-28 follow-up: RL launch on hold; pipeline refactoring under review.**
Read the dated [pipeline/reference plan](../../output/20261028_rl_preparation/PLAN.md)
for the scientific interpretation, exact preset proposal, adaptive-planner limits,
code-review order and MetaCentrum access blocker. No RL/PBS job has been submitted.
At MMFF94s ε78.4/C20/I4/0.5 Å/all retained references, 135/139 selected inputs
reach 0.95; four fall below it (minimum 0.777589). This is not a universal
guarantee or evidence for exactly three references. The user's latest steering
keeps epoch runs on hold. Defaults remain unchanged; the new explicit profile
configuration is optional. Historical measurements remain bound to 95d0d60.
Clean refactored source **0dca8cc** passed **27 tests** (3.906 s), scoped Ruff
and PBS shell syntax checks. Its production EGCG diagnostic retained three
references and seven candidates from 20 requests, scoring **0.99871485**
(shape 0.99979687, color 0.99763283), exactly matching component replay.
The adaptive-planner audit covers 507 saved prepared inputs: 151 hit the 250
per-isomer ceiling; nine exceed the stereo ceiling 25; total requests can reach
6,250. These are planning diagnostics, not validated coverage or new defaults.

**2026-10-28: native-scoring mini-study completed and verified.**

Read the compact [current report](../../output/20261028_1753_native_self_recovery/REPORT.md)
for selected settings, measured EGCG controls, limitations, reproduction and
code-review pointers. The old cross-molecule run and the later color-override
panel are cancelled/superseded; their saved data must not be mixed with this run.

Selected protocol: each input against its own independent reference ensemble;
no minimization, MMFF94 ε1 and MMFF94s ε78.4; reference RMSD selections 1.25/0.5 Å
from the same saved pools; reference requests 50 per stereoisomer, stereo cap 4;
candidate requests 1/5/10/15/20/ 25/30/50 and stereo caps 1/2/4/8. Stereo seed 42 on
both sides, candidate geometry seed 42 and independent reference geometry seed 43.
One maximum over permitted pairs per input, then summaries across inputs; no
AQME budgets, seed averaging, formal hypothesis test or RL training.

The native color/combo score is unchanged. Colorless prepared reference inputs
are excluded by feature presence **before any ROCS alignment**, with reasons
recorded in cohort.csv. This is an operational scope restriction, not proof of
drug-likeness. Candidate generation failures remain zero-score observations for
eligible inputs. Do not select by high scores or silently omit failures.

Groups count distinct retained reference structures (conformers and permitted
stereo identities), not merely stereoisomers. Use the maximum count across the
two optimized methods and BOTH cutoffs; exclude the unminimized profile from
this grouping calculation. Fixed ranges 1–4/5–8/9–12; more than 12 is outside this
mini-study and must be reported. Fixed membership applies across all figures.

Historical measurement source: Git revision **95d0d60**. Its cached execution checkout was removed during workspace consolidation; reproduce from Git, not an old execution path.
All **22** regression tests passed (3.644 s) and scoped Ruff passed there. The new
EGCG pilot completed 960 rows; six representative native public-scorer replays
matched within 1.2e-16. Exact settings, source hashes, test command and replay
values are in [its manifest](../../output/20261028_1753_native_self_recovery/egcg_pilot/manifest.json).
EGCG at 50 reference / 25 candidate requests retained 8 usable candidates and scored:
none 0.8838/0.9332, MMFF94 0.9571/0.9571, MMFF94s ε78.4 0.9989/0.9989 for cutoffs
1.25/0.5 Å. These are single-input controls, not panel-wide conclusions.

The panel finished successfully with **eight** one-thread workers; it was already
complete when the requested CPU increase was checked. No restart or allocation
change was needed. All 512 reference preparations were accounted for; selected
groups contain **61/40/38 inputs** (1–4/5–8/9–12), **139 total**. Exclusions:
345 above the selected reference-count range and 28 preparation failures
(14 embedding, five domain-filter, nine identity-check failures). No colorless
exclusions occurred in this panel; no candidate failures occurred in the selected
grid. The scope is 139/512 inputs, not a claim about the excluded inputs.

Output on both hosts: _david/output/20261028_1753_native_self_recovery/panel.
All **133,440** rows form complete unique 960-cell grids. Independent checks
passed reference masks/counts, same-pose component and timing reductions,
2,880 summary rows, source/output hashes and 1,454 reference SDF hashes.
Eighteen representative native PUBLIC-scorer replays spanning three groups,
three methods and two cutoffs matched within 1.12e-16. Evidence and inherited
22-test/pilot validation are appended to the existing panel manifest. Native
numeric overshoot is retained: combo max 1.00000361, shape max 1.00000899;
verification tolerance 1e-5. All 24 PNGs are mirrored locally with root data and
cohort; full reference SDFs remain on dichtator, with replay examples local.

At candidate C25/I4/all retained references, equal-input panel mean ROCS is
**0.9692/0.9890/0.9893** (none/MMFF94/MMFF94s ε78.4) at 1.25 Å, versus
**0.9872/0.9953/0.9961** at 0.5 Å. For MMFF94/0.5 Å, C5/10/25/50 means are
0.9925/0.9946/0.9953/0.9963; corresponding mean component seconds are
0.258/0.349/0.547/0.806. These are descriptive best-pair results for the restricted
cohort, not full ensemble coverage, an established optimal budget or RL benefit.
Read the report for all figures, denominators, caveats and review order.

Timing clarification: at the current C30/I4 budgets, candidate MMFF94 helper
time averages 0.0940 s/input (median 0.0532, p95 0.2189); MMFF94s ε78.4 averages
0.0933 s. Generation averages 0.3247 s. These are component measurements on
the selected 139 inputs, **not measured RL epoch slowdowns**. Total-time method
comparisons also change reference ensembles and cannot isolate the candidate
switch. See the report's RL timing section; no new experiment was launched.

The completion monitor can now be paused. The proposed existing-drug panel is
a later complementary study, not a replacement or extra experiment authorized
for this run. Production candidate minimization remains OFF by default. The
native user's environment.py argument-order edit must remain preserved.

## Superseded cross-molecule experiment — retained evidence

The user previously selected a
fixed generated-molecule scoring test, not RNN training. The new matrix compares
seven matched reference/candidate preparation methods on 512 randomly selected
valid, unique molecules from saved RNN evaluations, with three embedding seeds.
Candidate attempts: 1/5/10/15/20/30/50/100; isomer caps: 1/3/5/10/all≤30;
reference caps: 1/2/4/8/all≤30. Missing reference counts remain unavailable.
Four images will show mean, median, tail score/time, and failure/ranking stability.
The selected training switch remains OFF by default. The earlier minimized
in-memory trial failed a public-scorer parity check at C=15 (0.034 raw difference);
the shared helper now round-trips through SDF before either path scores it.

Evidence consulted for this extension: full-text Methods and Discussion of
[McNutt et al., 2023, DOI 10.1021/acs.jcim.3c01245](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647020/).
Their task-specific comparisons support measuring minimization, ensemble size
and diversity separately; larger ensembles can also introduce false positives.
Their docking/pharmacophore results do not establish DrugEx RL benefit or the
best parameters here. The new study keeps specified input stereo and labels
best-over-isomer scoring as a search over prepared forms, not activity evidence.

## Superseded execution checkpoint — do not resume

- **2026-10-28 provisional preview:** [mean score/time](../../output/conformer/mega_20261028/preview_508/mega_mean.png),
  [median](../../output/conformer/mega_20261028/preview_508/mega_median.png),
  [tails](../../output/conformer/mega_20261028/preview_508/mega_tails.png), and
  [failure/ranking stability](../../output/conformer/mega_20261028/preview_508/mega_reliability.png).
  These use 508 complete parent grids (2,133,600 rows), averaging three seeds
  per parent before aggregation. Missing parents: 51, 61, 226, 446. Their unusually
  slow generation makes this an incomplete subset; timing summaries may be biased
  downward. Do not treat the preview as the completed study or select production
  settings from it. Saved [summary](../../output/conformer/mega_20261028/preview_508/summary.csv)
  and [preview manifest](../../output/conformer/mega_20261028/preview_508/manifest.json)
  record included IDs, checkpoint hashes and rendering source hashes. Local copies
  match all preview artifact hashes. Preview rendering passed scoped Ruff checks
  on dichtator at `98afd83`, checked unique grid keys and 4,200 rows per parent,
  and left the running measurement source/data unchanged. Reproduce from existing
  checkpoints with `.venv/bin/python -B -m _david.code.conformer.mega --render-partial
  --output /home/kolarv/DrugEx/_david/output/conformer/mega_20261028`;
  this writes a separate `preview_<completed-count>/` directory. It does not run
  conformer generation or validate final public-scorer replay.
- Measurement source: clean committed `ff96921`, isolated at
  `/home/kolarv/.cache/drugex-mega-ff96921` on dichtator.
- Persistent job: tmux `drugex-mega-512`; 16 one-thread workers, no RL epochs.
- Output: `/home/kolarv/DrugEx/_david/output/conformer/mega_20261028`.
  Read its `manifest.json` and `exit_code` for the next bounded completion check.
  A successful full grid contains 2,150,400 parent/configuration/seed rows.
- Query preparation completed. Retained reference counts by method are:
  none=4; MMFF94=3; MMFF94s ε1=3; ε6.5=2; ε78.4=3; distance-dependent=3;
  UFF=2. Unsupported reference requests are unavailable, not clipped repeats.
- Eleven regression tests passed on the lab source (2.124 s), scoped Ruff
  passed, and all seven pilot public-scorer checks matched within 1e-10.
  Exact validation records are `/tmp/drugex-mega-ff96921-validation.json` on
  dichtator; copy them into the completed experiment's manifest verification.
- Measurements must finish in the frozen source above. The current renderer
  improves contrast and records rendering provenance; use `--render-only` after
  completion to generate final images without changing raw data or summaries.
- Remaining delivery: validate grid/denominators and representative public-scorer
  replay, render the four images, mirror evidence locally, replace this checkpoint
  with findings, and synchronize through private `dichtator` preserving user edits.
- The failed first minimized trial is preserved under
  `output/conformer/archive/mmff94_candidates_7039298_failed_parity` on dichtator,
  with its manifest explicitly marking the 0.034 raw-score parity failure.

## Selected stratified visualization — 2026-10-28

The user selected disjoint ranges **{1}, {2}, {3–4}, {5–8}, {9+}**, containing
293, 128, 55, 18 and 14 completed parents. These group the estimated isomer upper
bound after preparation, not physically realizable or successfully embedded
isomer counts; see [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html).
The 508-parent snapshot is still provisional; four unusually slow parents are
missing, so time summaries may be biased downward. No new conformers or scores
were generated for this analysis. The training switch remains OFF.

Artifacts on both hosts: `output/20261028_1405_conformer_analysis/`:
[group 1](../../output/20261028_1405_conformer_analysis/isomers_1.png),
[group 2](../../output/20261028_1405_conformer_analysis/isomers_2.png),
[group 3–4](../../output/20261028_1405_conformer_analysis/isomers_3_4.png),
[group 5–8](../../output/20261028_1405_conformer_analysis/isomers_5_8.png),
[group 9+](../../output/20261028_1405_conformer_analysis/isomers_ge9.png),
[score/time](../../output/20261028_1405_conformer_analysis/score_time.png),
[distributions/fixed cohort](../../output/20261028_1405_conformer_analysis/score_distributions.png),
[sampling diagnostics](../../output/20261028_1405_conformer_analysis/sampling_diagnostics.png),
[reference counts](../../output/20261028_1405_conformer_analysis/reference_effects.png).
The folder also contains `molecules.csv.gz` (seed-averaged parent measurements),
`stratified.csv` (plotted summaries), `panel.csv`, `fixed_top_cohort.csv`,
`identity_count_warnings.csv`, and the manifest with source/output hashes,
definitions, verification and separate rendering provenance. No extra report MD.

The user also selected median, tail and reliability panels for every range.
Use each `isomers_<range>_median.png`, `_tails.png`, and `_reliability.png` beside
the original mean/gain image (24 images total including four companion figures).
`additional_metrics.csv` holds the derived lower-tail and ranking values; the
original five data tables remain unchanged. Tails are 10th-percentile ROCS and
95th-percentile time across parent seed averages. Reliability compares each
method's ranking with its C100/I30 anchor within that range, with
k=ceil(0.05*n): 15/7/3/1/1 for ranges 1/2/3–4/5–8/9+. Anchor overlap is 1 by
construction. The last two figures measure top-1 stability, not ROCS similarity
or statistically established enrichment. Insufficient positive anchor scores
produce unavailable overlap, not a fabricated perfect score.

Heatmaps use all available references and shared scales. Paired isomer gains
compare the same parent/method/C against cap 1. Group 1 has zero such gain and
omits redundant gain panels. Score/time frontiers compare budgets within each
method; methods have different query geometries. MMFF94 time bars span median
to 95th percentile across parent seed-average times, not confidence intervals.
Molecule categories depend only on the recorded isomer bound. Conformer attempts
remain a varied budget (1/5/10/15/20/30/50/100) in every category, including group 1.
For each parent and seed, the score is the maximum over usable candidate
conformers/isomers and selected reference conformers, matching training's
best-match aggregation. It is never the mean of the candidate conformer scores.
Only afterward are the three seed maxima averaged for that parent, followed by
category mean/median/quantiles across parents. Requested attempts may yield fewer
conformers because of pruning or failures; realized counts remain separate data.
Preparation failures score zero in overall summaries. The success-only diagnostic
requires all three seeds to succeed at both caps and reports its smaller denominator.
At MMFF94/C100 in group 5–8, cap-30 minus cap-1 gain is 0.096466 mean, 0.024601
median, and about 0.020 among 12 fully successful pairs: overall gains partly
reflect failure recovery. Small strata remain descriptive, without significance claims.

Two inspectable examples (same prepared parent, MMFF94/C100/all references,
three-seed averages): parent 100 scores 0 at cap 1 with failure in every seed,
then 0.392378 at cap 30 with no failures. Parent 138 succeeds at both and scores
0.393194 vs 0.437888. Both inputs have unassigned stereo (bound 8), so changing
the cap can search different stereoisomers; these are not guaranteed comparisons
of one fully specified chemical identity. Different generated candidates versus
EGCG are not EGCG self-recovery. All 20 saved exact-geometry controls score 1.0;
ROCS here is `(shape Tanimoto + color Tanimoto) / 2`. Separate winning-pose
components were not saved, so this dataset alone cannot establish which component
limits an individual low score. Top-5% overlap is ranking stability, not similarity;
the fixed 26-parent MMFF94 anchor cohort has mean 0.421813 and maximum 0.441972.
It is selected on these measurements and is exploratory, not held-out validation.

**Diagnostic exception:** recorded `actual_isomers` exceeds its requested cap in
seven generation settings for parent 46. The benchmark counts distinct isomeric
SMILES after SDF readback; this is not a validated enumeration count. The cause
is unresolved. Preserve the measured values, flag them in the warnings CSV and
label that curve as SDF-readback identities; do not silently clip it to the cap.

Validation on dichtator: scoped Ruff and two targeted regression tests passed
(seed averaging, paired failure handling, range boundaries, lower-tail quantile,
fixed within-stratum ranking anchor and unavailable ranking cases). Every included
parent passed complete/unique grid, reference availability/count, score range,
failure/count and decomposed-time checks, with the identity-count exception above.
Independent Python statistics calculations checked all 1,400 summary cells and
12 raw-parent score/SD reductions for IDs 0/1/11/22/41/46. This is reduction replay,
not public molecular-scorer replay. Saved tables were unchanged during rendering.

Reproduce on dichtator from the repository root into a new directory:
```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m _david.code.conformer.analyze \
  --source /home/kolarv/DrugEx/_david/output/conformer/mega_20261028 \
  --output /home/kolarv/DrugEx/_david/output/YYYYMMDD_HHMM_conformer_analysis
```
Use `--render-only --output <existing-analysis-directory>` without `--source`
to regenerate figures from saved tables. Review [analyze.py](analyze.py) for
validation/aggregation and [visualization.py](../visualization.py) for the
`_analysis_*` renderers, then [the test](../../tests/test_conformer_analysis.py).

Evidence reused/refreshed from free Europe PMC full text: McNutt et al. 2023
(linked above), conformer retrieval and pharmacophore ensemble-size results.
They report heterogeneous responses hidden by average trends and effects of
ligand flexibility; their task does not select our ranges or budgets. Search:
ensemble size, rotatable bonds, flexibility, false positives. Measurements remain
bound to `ff96921`; this selected change only analyzes and renders saved data.

## One-isomer-category audit — 2026-10-28

Read-only analysis of the existing 293 upper-bound-one parents, with no new
generation/scoring. Every parent is compared with the shared EGCG reference;
**zero of the 293 has EGCG connectivity**. The 1,000-attempt deep preparation
created the EGCG ensemble, not an independent reference for each parent.
MMFF94 has three retained query conformers. All isomer-cap score aliases in this
category agree exactly; candidate conformers are still maximized before seed
averaging. The score is divided by two once, before storage, and plotted directly.

MMFF94/all references, means over all parents after averaging three seeds:

| Requested conformers | Mean score | Median score | Mean embedded count | Mean seconds |
| --- | --- | --- | --- | --- |
| 1 | 0.312815 | 0.317233 | 0.993 | 0.143 |
| 10 | 0.332728 | 0.337417 | 7.412 | 0.508 |
| 30 | 0.337885 | 0.341747 | 17.507 | 1.164 |
| 100 | 0.341244 | 0.344548 | 41.896 | 3.024 |

At C100 the 10th/90th percentiles are 0.297435/0.384890; maximum is 0.438530.
Only parents 8 and 72 score zero, in all seeds. Their input sizes are 46 and 50
heavy atoms, above the existing generator limit of 45: these are size-filter
exclusions before embedding, not evidence of minimization nonconvergence. Raw
rows record zero candidates and a subsequent empty-SDF file error. Excluding
these two changes the mean only from 0.341244 to 0.343590, so failures do not
explain the category's low average.

The paired C30→C100 mean gain is 0.003360 (median 0.000748); 33/293 parents gain
more than 0.01. This is diminishing average benefit for this different-molecule
versus-EGCG task, not a universal conformer-budget recommendation. For context,
the median parent has 25 heavy atoms and three oxygens, versus EGCG's 33/11.
Parent 473 is the 9-heavy-atom triamine `NCCCNCCCN` and scores 0.198657; the
median parent 10 scores 0.344548; parent 373 scores 0.438530. These are different
chemical structures, not independent preparations of EGCG. Morgan radius-2/2048
fingerprint Tanimoto to EGCG has median 0.090909 and maximum 0.169811; its
Spearman association with C100 ROCS is only 0.028, so do not use this 2D measure
as a quantitative explanation or validation of the 3D score.

Conclusion: neither averaging conformers nor a large failed-parent fraction
explains 0.34. This panel does not measure the same-identity self-recovery the
user is questioning. Its low means alone neither prove a scorer bug nor validate
the scorer. Separate winning-pose shape/color components were not saved, and the
parent-46 SDF identity-count exception remains unresolved.

## Filtering and self-recovery proposal — not yet selected

Keep the full attempted-parent dataset and failure ledger. Individual optimized
conformers already require native convergence status 0; this does not establish a
global minimum or biological plausibility. "None" has no force-field convergence
criterion. Report preparation availability separately from conditional score
quality. For a planned paired comparison, use the same successful-parent subset
for both settings and report its denominator plus failure rates on the full panel;
do not compare independently filtered method-specific populations as if matched.
Restricting to supported chemistry by declared rules is defensible, but its scope
and exclusions must be explicit. Selecting only high-scoring or universally easy
cases after inspection would answer a biased, narrower question.

A proposed next study addresses independent molecular self-recovery: a fixed,
diverse panel including EGCG, each with a specified identity and frozen reference
ensemble. Regenerate from isomeric SMILES with independent seeds and no reference
coordinates as candidate input. Use known stereo first; an intentionally masked
candidate-stereo challenge would be a separate labeled arm. Record reference
preparation failures, candidate failures, normalized shape/color combo, its two
components at the same winning pose, symmetry-aware heavy-atom RMSD and timing.
An exact-copy control should score 1; independently regenerated conformers need
not. Fixed targets isolate candidate preparation, while the existing matched
reference/candidate methods compare whole protocols. Computational references
measure recovery of those protocols; experimental conformers would support a
different physical-recovery claim. No new study or dataset filtering is authorized
or implemented by this proposal.

Evidence consulted 2026-10-28: free full text of [High-Quality Conformer Generation
with CONFORGE, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10498443/), sections
Dataset Preparation, Processing Time Measurement and RMSD Calculation, and
molecule-processing-failure tables. It converts reference structures to isomeric
SMILES to avoid supplying reference coordinates, assesses best symmetry-aware
heavy-atom RMSD, and separately reports failed molecules. Those benchmarks do not
prescribe our ROCS thresholds or panel size. [RDKit force-field documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)
defines status 0 as converged and status 1 as requiring more iterations; neither
is an activity label. Search scope: conformer benchmark failures, exclusion,
stereochemistry, independent reference RMSD. Current raw scores are unchanged.

## Historical single-molecule baseline

The following dated sections describe the earlier single-molecule experiment;
they are not the current mega-study result or a production-parameter selection.

Updated 2026-10-27. Current instructions: [AGENTS.md](../../AGENTS.md).
Source lives here; generated evidence belongs under `_david/output/conformer/`.

## Current implementation

The preparation/configuration fix is `3cb45dc`; the initial study implementation
is `51b5562`. The subsequent folder move follows the user's dichtator layout.
All implementation changes are under `_david/`. DrugEx core and the RNN model
implementation are unchanged by this work.

- `../chemistry.py` reproduces DrugEx's RDKit standardization sequence with
  isomeric serialization. Identity checks compare connectivity and specified
  stereo, reassigning stereo from 3D coordinates rather than trusting SDF tags.
- `../environment.py` uses that standardization for training reference
  preparation, rejects mixed-isomer SDFs and aligns atom mappings when combining
  records. It honors constructor seeds/attempts, the configured energy window
  and requested reference cap. Clustering uses an explicit 1.25 Å cutoff.
  Only converged, finite-energy reference conformers are retained; the actual
  force field and embedding/convergence counts are saved in the SDF.
- Scorer construction now forwards timeout and isomer limits. Omitted environment
  arguments defer to PipelineConfig instead of overriding it. The candidate
  generator is the existing core algorithm with a small seed-exposing subclass;
  its default embedding seed remains 0xC0FFEE.
- `study.py` compares training-path ETKDG candidates with the same candidates
  minimized using MMFF94. Candidate minimization is a diagnostic alternative;
  training still uses unminimized candidates. Only MMFF94 references are accepted
  in this condition; a UFF fallback is not silently mixed into its results.
- `conformer_experiment.py` delegates to the current study. The moved legacy
  launcher no longer imports the removed Gemini implementation.

A reference must specify its stereochemistry for this self-recovery study.
Candidate input can be supplied separately; unresolved centers can be enumerated
while preserving specified centers. Reference policy for unresolved stereo
remains a research decision. The study rejects a candidate preparation whose
connectivity or specified stereo differs from the stated input; this is not a
general tautomer/protonation equivalence checker.

## Controls and validation

The single module `_david/tests/test_sequencer_pipeline.py` contains six focused
tests: standardization chemistry/stereo, reference SDF round trips and energy
windows, rejection of a wrong isomer/mixed SDF, exact-geometry raw/modified score
scales and parameter wiring, candidate identity for the specified reference, and
independent storage plus freezing of the RNN mutation prior.

Local WSL, RDKit 2026.09.1: all six passed (3.216 s); Ruff passed.
Dichtator, RDKit 2026.03.6, revision `51b5562`: all six passed (0.984 s);
Ruff passed for the task files. A broader check also found an unrelated E741 in
the user's moved molecular-dynamics code; it was not changed.

Final validation and measurements used clean committed source `a09fe58` in
`/home/kolarv/.cache/drugex-mmff94-a09fe58`. All six tests passed (0.937 s), and
Ruff passed for that checkout's code/tests. Import origins and before/after source
hashes were verified. Full commands, logs, versions and validation records are in
the experiment manifest. Concurrent user reorganization, including deletion of
old molecular-dynamics ligands/output, was preserved in the active checkout.

The earlier weak smoke assertions were replaced, not treated as proof of
conformer correctness. Tests do not substitute for reviewing the saved structures
and measurements. The RNN guard does not constitute an audit of the earlier
SequenceRNN transfer or establish training convergence.

## Selected experiment

The user selected the default MMFF model (RDKit MMFF94) and RDKit Tanimoto
shape-plus-color as the initial baseline, with paired minimized/unminimized
candidates. The earlier five-method/two-score comparison is deferred.

Completed run: 1,000 reference embedding attempts, reference seed 42, MMFF94
minimization up to 1,000 iterations, 8 kcal/mol energy window and 1.25 Å heavy-atom
RMSD clustering. Candidate attempts per isomer: 1, 10, 30, 100, 200; isomer caps:
1, 4, 8; reference caps: 1, 2, 3, all; candidate seeds: 12648430–12648432.
These budgets define this comparison; they are not selected production settings.
No wall-clock embedding timeout; one CPU thread; no RNN training.

Each row records requested and actual counts, convergence failures, winning
isomer, raw combo [0,2], combo/2 [0,1], generation/minimization/overlay times.
Timings exclude one-time reference preparation and multiprocessing startup.
They include the study's identity audits and are not a training throughput claim.
Reference-prefix scores reuse measured pairwise overlays; times sum those
overlays and add candidate preparation. They are not full training latency.

Fully specified EGCG should yield one stereoisomer at every isomer cap. Flat
isomer panels are an expected negative result, not evidence for a universal cap.
A later broader-molecule study must evaluate whole distributions as well as a
predefined top-5% tail. The current lead renderer selects five molecules, not 5%.

## Measured EGCG self-recovery

One specified molecule, three independent candidate embedding seeds. The 360 CSV
rows are configuration/seed combinations, not 360 molecules. Isomer caps 1/4/8
each produced one isomer and identical scores for each otherwise matched case.
Reference preparation took 16.98 s: 1,000 requested attempts → 36 embedded →
36 converged → three retained representatives. All 375 generated candidate
conformers converged in the paired minimization; 750 original/minimized SDF
records were saved and their 3D identities checked.

The following uses isomer cap 4 and all three references. Scores are combo/2,
mean ± sample SD across the three seeds; time is the mean seconds per molecule
under the single-thread measurement definition above.

| Requested C | Actually embedded | ETKDG score | Minimized score | ETKDG seconds | With minimization seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.596 ± 0.018 | 0.877 ± 0.133 | 0.168 | 0.189 |
| 10 | 5–7 | 0.685 ± 0.047 | 0.988 ± 0.015 | 0.429 | 0.543 |
| 30 | 7–10 | 0.714 ± 0.050 | 0.988 ± 0.015 | 0.820 | 1.019 |
| 100 | 11–13 | 0.736 ± 0.015 | 0.998 ± 0.004 | 2.055 | 2.322 |
| 200 | 13–15 | 0.736 ± 0.015 | 0.998 ± 0.004 | 3.670 | 3.980 |

Minimization has a much larger effect here than increasing the unminimized search
from 100 to 200 attempts. This supports a geometry-preparation mismatch for this
specified EGCG comparison. It does not establish bioactive conformations, an
optimal force field, general conformer/isomer budgets or improved RNN training.
In particular, the flat isomer axis cannot select a cap for ambiguous molecules.

All three exact-geometry controls scored 2 raw. A deliberately inverted center
was rejected by the identity check (its raw overlay was 0.621431; no similarity
threshold was used as an identity test). Public-scorer replay at C=30/200,
default seed, cap 4 and all references reproduced both CSV scores exactly.
Replay from saved C=200 SDFs matched the unminimized score exactly and differed
by 1.554e-8 raw for the minimized score, consistent with SDF coordinate rounding.
These are the stated replay cases, not a replay of every measured cell.

![Score and time grids](../../output/conformer/mmff94_20261027/summary.png)

Generated evidence, mirrored from dichtator with matching hashes:
[measurements](../../output/conformer/mmff94_20261027/data.csv),
[controls](../../output/conformer/mmff94_20261027/controls.csv),
[reference structures](../../output/conformer/mmff94_20261027/reference.sdf),
[candidate structures](../../output/conformer/mmff94_20261027/candidates.sdf),
[manifest and verification](../../output/conformer/mmff94_20261027/manifest.json).
These ignored outputs are not included in a fresh Git clone.

From the repository root on dichtator, reproduce into a new output directory:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DRUGEX_SCORE_NJOBS=1 \
  .venv/bin/python -B -m _david.code.conformer.study \
  --output /home/kolarv/DrugEx/_david/output/conformer/mmff94_repeat
```

Use `--render-only --output <existing-run-directory>` to regenerate the PNG from
the CSV. Read [chemistry.py](../chemistry.py), [reference/scorer construction](../environment.py),
[study.py](study.py), [the renderer](../visualization.py) and the focused test
module for code review. No earlier unreviewed RNN implementation is certified by
this study.

## Evidence and limits

Consulted 2026-10-27: full-text Gypsum-DL methods, “Unspecified chiral centers”
and “Controlling the combinatorial explosion”, preserve specified centers and
distinguish variant dimensions. That docking study establishes neither a DrugEx
budget nor an RL benefit: [Ropp et al., 2019, DOI
10.1186/s13321-019-0358-3](https://link.springer.com/article/10.1186/s13321-019-0358-3).
Official [RDKit force-field documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)
defines MMFF94 as default and optimization status 0 as converged.
[RDKit stereo enumeration](https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html)
documents preservation of assigned stereo and seeded subset enumeration.
Online documentation may differ from the installed version; live checks cover
both installed versions above. There is no established optimal isomer/conformer
ratio. Code-based deductions and proposals are not measured training effects.

Remaining limits: core clamps conformer requests above 200; native failed
embedding IDs and multiprocessing coordinate precision require separate
investigation. This study runs sequentially and checks 3D identity, so it does
not validate multiprocessing parity. Legacy training visualization still infers
score scales and can double ROCS twice; the new figure reads explicit CSV scales.

## Historical provenance

The review baseline before this work was `e7035d7`; runtime was restored to
`0f6a30d` in rollback `8a8386c`, preserving config annotations and timeout=0.
Previous conformer recommendations are withdrawn. Removed scripts, tests and
reports remain recoverable at `570fc77` on
`codex/recovery-before-conformer-rollback-20261027T185549Z`.
Patches, copied user work and documentation archives are under
`_david/output/rollback_checkpoints/20261027T185549Z/`. User reorganization and
deletions on dichtator are preserved.

The 2026-10-26 review of historical Run 17/18 found 50 recorded epochs and 1,000
evaluation rows per arm. Validity fell from 955 to 883 rows in Run 17 and 951 to
845 in Run 18; stored all-row ROCS means fell 0.328771→0.303615 and
0.326311→0.276163 respectively. Sources were the runs' baseline/reinforced raw
CSVs and training TSVs. These were not rerun here; scale/provenance issues prevent
using them as calibrated physical scores or causal hyperparameter evidence.
