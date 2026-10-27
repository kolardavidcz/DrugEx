# Conformer study: status and evidence

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

The earlier weak smoke assertions were replaced, not treated as proof of
conformer correctness. Tests do not substitute for reviewing the saved structures
and measurements. The RNN guard does not constitute an audit of the earlier
SequenceRNN transfer or establish training convergence.

## Selected experiment

The user selected the default MMFF model (RDKit MMFF94) and RDKit Tanimoto
shape-plus-color as the initial baseline, with paired minimized/unminimized
candidates. The earlier five-method/two-score comparison is deferred.

Planned run: 1,000 reference embedding attempts, reference seed 42, MMFF94
minimization up to 1,000 iterations, 8 kcal/mol energy window and 1.25 Å heavy-atom
RMSD clustering. Candidate attempts per isomer: 1, 10, 30, 100, 200; isomer caps:
1, 4, 8; reference caps: 1, 2, 3, all; candidate seeds: 12648430–12648432.
These budgets define this comparison; they are not selected production settings.
No wall-clock embedding timeout; one CPU thread; no RNN training.

Each row records requested and actual counts, convergence failures, winning
isomer, raw combo [0,2], combo/2 [0,1], generation/minimization/overlay times.
Timings exclude one-time reference preparation and multiprocessing startup.
Reference-prefix scores reuse measured pairwise overlays; times sum those
overlays and add candidate preparation. They are not full training latency.

Fully specified EGCG should yield one stereoisomer at every isomer cap. Flat
isomer panels are an expected negative result, not evidence for a universal cap.
A later broader-molecule study must evaluate whole distributions as well as a
predefined top-5% tail. The current lead renderer selects five molecules, not 5%.

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
