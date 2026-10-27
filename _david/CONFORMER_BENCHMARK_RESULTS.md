# Conformer budget results, 2026-10-27

Use 100 reference embedding attempts and 30 candidate requests per stereoisomer
as the working baseline while assessing reference coverage. This experiment
does not justify lowering the training budget to 10 or 20. The most promising
next comparison is retaining more reference basins, rather than increasing
candidate requests from 30 to 50. Five retained references is an exploratory
candidate for that comparison, not an independently validated new default.
No training configuration or scientific objective was changed.

## Completed comparison

The matrix contains 241,920 measurements: 32 molecules, three paired seed
replicates, eight reference protocols, three matching functions, reference
embedding budgets 32/100/250, retained-reference caps 1/2/3/5/8, and candidate
requests 1/5/10/20/30/50. Four protocols also have a matched candidate-minimization
arm at reference budget 100. There are 36 condition figures, each exported as
PNG and PDF: 72 files. Each figure contains score, loss, mean-time, median-time,
and p95-time maps. The normal reference clustering cutoff is 1.25 Å.

The frozen fingerprint-diverse panel has 16 screening molecules, eight held-out
molecules, and eight separate controls. It is a stress panel, not a random
population estimate. Reference seeds are 42/43/44; candidate seeds are
0xC0FFEE/0xC0FFEF/0xC0FFF0. Timing uses eight benchmark workers, one embedding
thread per grid build, and explicit spawned processes. Counts are requested per
stereoisomer; pruning, filtering, failures, and up to four enumerated isomers
change the actual retained counts.

Raw combo scores range from 0 to 2. The training modifier divides these scores
by two. Loss is relative to the best observed for the same molecule, replicate,
reference protocol, candidate method, and matching function in this finite grid.
It is not a comparison with a known bioactive pose or measured affinity.

## Counts and held-out evidence

The predeclared screening rule selects the fastest median latency subject to
mean raw loss <=0.02, p95 loss <=0.05, and preparation failures <=5%. The selected
setting is then evaluated unchanged on the holdout. A secondary mean-latency
analysis selects the same count setting for the current Tanimoto path.

For MMFF94 references and the current unminimized ETKDG/Tanimoto candidate path:

| Reference attempts | Reference cap | Candidate request | Screening mean loss | In-domain holdout mean loss | Holdout p95 loss | Holdout mean-loss 95% bootstrap interval |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 3 | 10 | 0.02236 | 0.02953 | 0.07581 | upper bound 0.04535 |
| 100 | 3 | 20 | 0.01574 | 0.02640 | 0.05303 | 0.01296–0.04189 |
| 100 | 3 | 30 | 0.01362 | 0.02389 | 0.04732 | upper bound 0.04009 |
| 100 | 3 | 50 | 0.01200 | 0.02218 | 0.04732 | upper bound 0.03913 |
| 100 | 5 | 30 | 0.00663 | 0.01769 | 0.04732 | upper bound 0.03137 |
| 100 | 8 | 50 | 0.00233 | 0.00899 | 0.01720 | upper bound 0.02182 |

Only the 100/3/20 setting was selected by the primary screening rule. It failed
held-out score tolerances. The other rows describe the existing baseline or
exploratory comparisons; they must not be relabeled as holdout-selected,
independently validated winners. Raising candidate requests to 50 while retaining
only three references still misses the mean-loss tolerance. Retaining five
references at 30 requests meets the observed in-domain tolerances, but its
uncertainty is too wide to confirm the mean-loss tolerance. Even the 100/8/50
upper confidence bound is slightly above 0.02.

The primary holdout denominator remains all eight molecules. One has 52 heavy
atoms and exceeds the production generator's 45-atom limit, producing a 12.5%
holdout preparation-failure rate at every count. The conditional table above
uses seven in-domain holdout molecules and 21 seed observations; it does not
rescue the failed primary decision. Bootstrap sampling uses molecules after
averaging their seeds. Across all 36 conditions, 26 have a screening-selected
setting and none passes the full primary holdout rule; ten have no acceptable
screening setting. Ten of the 26 selected settings meet the secondary in-domain
observed tolerances, illustrating why the denominator must remain explicit.

Reference search itself is relatively cheap. MMFF94 preparation averages
2.30 seconds for 100 attempts and 4.65 seconds for 250. At reference cap eight
and candidate request 50, increasing attempts from 100 to 250 changes the
paired model-panel mean score by +0.000127, with molecule-bootstrap interval
[-0.000523, +0.000799]; mean absolute change is 0.001055. Both budgets yield
seven to eight reference basins across the seeds. This is evidence of numerical
stability on this panel, not proof of bioactive-conformation coverage.

## Measured production cost and numerical parity

The full production factory scored all 32 molecules with automatic candidate
threading and eight spawned scoring workers. These times include live
generation, SDF I/O, process startup, and scoring, but exclude RNN generation
and the rest of an RL epoch. Both conditions use 100 reference attempts and
three retained references.

| Candidate request | Batch seconds, repeat 1 | Batch seconds, repeat 2 | Mean batch seconds |
|---:|---:|---:|---:|
| 20 | 8.2685 | 8.3249 | 8.2967 |
| 30 | 9.8160 | 10.0410 | 9.9285 |

Twenty requests saves about 16.4% in this small batch comparison, but the
held-out loss above does not justify adopting it. Two timing repeats use the
same chemistry seed and are not independent chemistry replicates. Fixed-worker
per-molecule grid times and these full-batch times answer different timing
questions; neither alone predicts a full RL epoch.

Two real production defects were found and corrected:

1. RDKit can return a negative embedding ID after a timeout. Passing that ID to
   SDWriter could create a 2D depiction despite failed 3D embedding. Negative IDs
   are now rejected and counted. The complete candidate/minimization matrix was
   rebuilt. All 32,354 regenerated SDF records have finite 3D coordinates.
2. Default RDKit process pickling rounds coordinates. A native probe measured
   about 2.3e-7 Å coordinate error and reproduced a 0.01276 change in an overlay
   optimum. Candidate threading was ruled out in the probes. Explicit double
   precision payloads now preserve geometry across the process boundary without
   changing the overlay algorithm or global pickle settings. After correction,
   full production scores match the grid within 9.44e-16 at both confirmed count
   settings, for all 32 molecules. The original precision-losing confirmation
   is archived as `production_confirmation_default_precision`.

## Mechanics, matching functions, and limitations

MMFF94s dielectric 1/6.5/78.4, distance-dependent electrostatics, UFF, unminimized
ETKDG, and tighter 0.75 Å clustering were measured alongside current MMFF94.
Dielectric parameters are mechanical sensitivity conditions, not explicit
solvent or validated protein environments. Different matching functions also
change the score scale and size preference; a higher raw score under one
function is not evidence of better ligand ranking or activity.

Additional candidate minimization has no demonstrated average Tanimoto benefit
for the baseline count setting. Among pairs successful under both methods,
MMFF94 changes the mean score by -0.00241 with bootstrap interval
[-0.00761, +0.00195]. MMFF lacks parameters for one phosphorus-containing
screening molecule, creating an additional coverage failure. That makes its
matched-minimization conditions fail the screening failure tolerance. No silent
force-field substitution is used. UFF's corresponding valid-pair mean change
is +0.00374 with interval [-0.00416, +0.01287], also inconclusive.

The measured environment has no Dimorphite-DL, so its existing protonation
helper leaves input structures unchanged. The results reproduce that behavior;
they do not establish a pH-7.4 protomer ensemble. Source samples also contain
45/1,000 and 155/1,000 invalid SMILES, respectively; nine valid duplicate rows
occur in the reinforced source. Those failures are audited before selecting
the valid unique stress panel. No prepared molecular identity overlaps between
screening and holdout after production tautomer/stereoisomer handling.

The literature consultation and access levels are recorded in
[the method note](CONFORMER_BENCHMARK.md#literature-consultation-2026-10-27).
It provides no universal conformer-count constant or justification for choosing
a water-like dielectric automatically. A future scientific comparison should
test reference coverage on a new untouched panel and, when relevant, experimental
pose or active/decoy endpoints. This experiment does not measure RL learning
equivalence, enrichment, affinity, or biological activity.

Before applying reference caps above three, the existing production preparation
rule and configuration wiring need deliberate changes: the EGCG path currently
hard-caps the reference ensemble at three, and some factory defaults override
config values. Merely editing config would not implement the proposed comparison.

## Evidence and handoff

The authoritative complete run is on dichtator at
`/home/kolarv/DrugEx/_david/output/conformer_benchmark_20261027/`.
It contains `REPORT.md`, 72 figures, compressed measurements, individual build
records and SDFs, source snapshots, input hashes, screening/holdout summaries,
selection tables, preparation-failure diagnostics, reference/seed stability,
paired-minimization effects, and both production confirmations. Superseded
baseline and failed-embedding artifacts are retained separately and excluded
from the final matrix. See [reproduction instructions](CONFORMER_BENCHMARK.md).

Measurement CSV SHA-256:
`d68eb3913086f3797e58ca4d9c72caebac60c3a415b67859cc0c60f6646a58d6`.
The 241,920 measurement identities are unique, all raw scores are finite and
bounded, raw/modified scaling is consistent, and every nested reference prefix
has nondecreasing maximum score. Exact geometry self-overlay returns 2.0 for
all three matching functions. Grid matching took 3,138.9 seconds; the manifest
wall time also includes resumed builds and should not be used as per-candidate
throughput.

The benchmark implementation is commit `e2252c0`; the precision correction and
this interpretation are in the subsequent task commit. Launch revision in the
manifest is historical; source snapshots and per-segment hashes identify the
actual measured implementation. Final analysis and confirmation provenance are
recorded separately. Training config annotations and unrelated remote file
reorganization are preserved.

Validation after the precision correction: 19 thesis tests pass, including all
three sequential/parallel metric comparisons and lossless process payloads.
Targeted core tests have seven passes and one Dimorphite-DL availability skip.
Thesis Ruff passes. Modified core files have no new Ruff diagnostics relative
to their baselines (existing counts: 21 in the generator, 86 in the scorer).
The existing legacy fork warning is recorded; this benchmark uses spawn.
