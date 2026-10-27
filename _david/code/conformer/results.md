# Conformer study: status and evidence

**2026-10-28: mega benchmark implementation in progress.** The user selected a
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

## Active execution checkpoint

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
