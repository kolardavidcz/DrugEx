# Reference and training conformer benchmark

> **Withdrawn for EGCG recommendations.** An audit found that reference
> standardization discarded the specified stereochemistry. Use the corrected,
> compact [EGCG self-recovery study](conformer_self_recovery/report.md).
> This document and its measurements remain historical evidence.

This experiment asks how much conformational search is needed to preserve the
production scorer's results at an acceptable CPU cost. It does not establish
binding affinity or identify a bioactive EGCG conformation.

The completed experiment and qualified conclusions are in
[CONFORMER_BENCHMARK_RESULTS.md](CONFORMER_BENCHMARK_RESULTS.md).

Three separate quantities matter:

| Quantity | Meaning | Tested values |
|---|---|---|
| Reference embedding budget | Conformers requested before minimization, energy filtering, and clustering | 32, 100, 250 |
| Retained reference cap | Lowest-energy representatives of distinct heavy-atom RMSD clusters used for scoring | 1, 2, 3, 5, 8 |
| Candidate budget | ETKDG conformers requested per enumerated stereoisomer during training scoring | 1, 5, 10, 20, 30, 50 |

Requested counts differ from actual retained counts. The candidate pipeline can
enumerate up to four unassigned stereoisomers, so its total can also exceed the
per-isomer request. All actual counts are recorded.

## Reproduce

Run from the repository root using the prepared environment on dichtator:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DRUGEX_SCORE_NJOBS=1 MPLBACKEND=Agg \
  .venv/bin/python -B -m _david.code.conformer.benchmark_cli \
  --workers 8 --output _david/output/conformer_benchmark_NEW
```

Use `--smoke` for a small integration check. Use `--analyze-only --output PATH`
to regenerate reports without repeating chemistry. An interrupted run can be
continued with `--resume --output PATH`; configuration and input-file hashes
must match. Completed build records and complete scoring tasks are reused.
Source snapshots and per-segment source hashes preserve the implementation
used before and after a resumed run. Do not edit result records to manufacture
completeness.

After completing the grid, `--confirm-production --output PATH` checks current
and selected Tanimoto counts through the full scoring factory. It uses the
production candidate thread setting (`0`), eight spawned scoring workers, the
frozen panel, and two timing repeats. Its times include generation, SDF I/O and
pool overhead; numerical differences from the one-thread grid are recorded.
Run this after the matrix so the two workloads do not compete for CPUs.

Default panel sources are saved prior samples from Run 17 and reinforced samples
from Run 18, selected using Morgan-fingerprint max-min diversity without looking
at saved rewards. Each source contributes 12 unique molecules: eight screening
and four holdout. Eight controls, including EGCG, analogues, and small unrelated
molecules, are reported separately. Three paired seed replicates are used.
Holdouts are later max-min selections, so this panel is a diversity stress test,
not a random sample for estimating population-wide RNN performance.
Across-source deduplication excludes canonical structures encountered anywhere
in an earlier source pool, so the reinforced panel emphasizes structures not
present in that prior sample. This is another reason to avoid treating these
panel averages as population estimates.

## What constitutes evidence

For each molecule and seed, the score is the maximum production overlay score
over candidate conformers and the selected reference ensemble. Within a fixed
reference ensemble, adding reference conformers cannot lower that maximum.
Independent embedding budgets generate and cluster different ensembles, so
increasing an embedding budget need not improve a particular score.

The comparison target is the best observed score across the finite grid for the
same molecule, seed, protocol, matching function, and candidate method. Loss is
that score minus the score of a smaller setting. The selection rule was fixed
before the full result was available: choose the fastest screening setting with
mean raw-combo loss at most 0.02, 95th-percentile loss at most 0.05, and preparation
failures at most 5%. Check that exact setting on held-out molecules without
retuning. The tolerances are engineering choices; the raw combo ranges from zero
to two, and its training modifier divides it by two. Bootstrap intervals resample
molecules after averaging seeds. Repeated seeds are not independent molecules.

Reference conditions compare MMFF94, MMFF94s, UFF, unminimized ETKDG, MMFF94s
dielectric changes, and a tighter RMSD clustering cutoff. An additional arm
minimizes candidates with the corresponding mechanics at reference budget 100.
Current training candidates are unminimized. Dielectric constants are mechanical
sensitivity parameters, not explicit water or a validated protein environment.
An 8 kcal/mol window is the existing reference energy filter, not proof of a
physiological energy distribution. Energies across different force fields are
not compared.

The normal EGCG cutoff is 1.25 Å, verified against the production flexibility
rule and the first three production reference geometries. An earlier 1.5 Å
benchmark baseline was corrected before drawing conclusions. Its reference
builds and partial matching records remain explicitly marked as superseded;
final matching uses rebuilt 1.25 Å references throughout.

Here 1.25 Å is the project's EGCG clustering rule; it is not presented as a
universal RDKit default.

The wider panel also exposed a production generator bug: RDKit can return
`[-1]` after a native embedding timeout. The previous loop passed that sentinel
to `SDWriter`, which generated a 2D depiction when no 3D conformer existed;
Tversky then raised `Conformer must be 3D`. The corrected generator rejects and
counts negative embedding IDs. A native reproduction and deterministic
regression test verify the fix. The entire candidate/minimization arm and
matching matrix are repeated with the corrected generator; earlier artifacts
are retained separately. Partial successes retain their real conformers and
their explicit rejected-result counts.

The measured environment lacks Dimorphite-DL. Its existing protonation helper
therefore returns input structures unchanged, confirmed with an acetic-acid
probe in `environment_audit.json`. The benchmark reproduces this current
training behavior; it does not establish a pH-7.4 protomer distribution.
Adding a protonation backend changes the chemistry and needs a separate
comparison rather than an installation during this measurement run.

Tanimoto and the two existing Tversky variants call the production overlay
kernel. Its alignments optimize equal-weight shape and color; Tversky variants
convert the overlap components and select their best score over those starts.
This is not a new optimizer for a Tversky-specific pose objective. Higher scores
under a differently weighted metric do not establish a better metric.
The production coefficients `(reference weight, candidate weight)` are
`(0.95, 0.05)` for `tversky_ref` and `(0.30, 0.70)` for
`tversky_subshape`. Their asymmetry changes which unmatched volume is penalized;
the exported molecule-size correlations are a diagnostic, not an activity test.

Timing separates one-time reference preparation, candidate generation/SDF work,
and actual overlay execution under a fixed concurrent worker budget. It excludes
RNN sampling and multiprocessing startup/IPC. Tail latency, failures, optimizer
iteration limits, seed sensitivity, changes between reference budgets, and
matched-minimization effects are exported alongside score/time heatmaps.
Nested reference caps reuse a running maximum over exact production pair
scores; each cap stores the actual cumulative time for the work required by
that prefix. Tests compare these results with independently scored ensembles.
Mean, median, p95 and maximum latency are exported. The original screening rule
minimizes median latency; a separate mean-latency selection uses the same score
tolerances and screening partition. That secondary analysis was added after
the pilot exposed long tails and is labeled separately because average service
cost is relevant to training throughput.

Minimization diagnostics report end-to-end score changes, preparation-failure
fractions, and a separate comparison restricted to pairs successful under both
methods. A missing force-field parametrization is a coverage failure rather
than evidence that minimization produced a worse physical geometry.

## Outputs and checks

The dated experiment is `_david/output/conformer_benchmark_20261027/` on
dichtator. Generated ensembles and detailed measurements remain ignored by Git.
Its `manifest.json`, source snapshots, molecule panel, individual build records,
`measurements.jsonl`, compressed CSV, summary tables, `selected_settings.csv`,
diagnostic tables, `REPORT.md`, and PNG/PDF figures form the evidence package.
Analysis source hashes are recorded separately from measurement source hashes.

Tests compare all three matching functions against the full production scorer,
keep filtered molecules in the failure denominator, verify disjoint panel
identities, check molecule-level bootstrap behavior, and prevent reselection
after a failed holdout. Passing these checks verifies implementation behavior;
the full measurements are still needed for a scientific recommendation.

Before applying a recommendation to RL, inspect configuration wiring in
`environment.py`: existing non-null factory defaults can override config values;
the timeout argument is not forwarded to the candidate generator, the reference
energy window is hardcoded, and reference preparation caps the EGCG ensemble at
three even when a larger cap is requested. This benchmark passes its settings
explicitly and does not change the user's annotated training configuration.

Full-batch confirmation exposed a separate precision issue: default RDKit
process pickling reduced coordinate precision and changed some overlay optima.
The production scorer now serializes its reference and candidate payloads with
`CoordsAsDouble`, retaining the existing property flags without changing global
pickle settings. Native probes isolate the effect from candidate threading.
The corrected full scorer matches the grid within 1e-15 for all 32 molecules
at both confirmed count settings. The overlay kernel and grid scores are
unchanged; earlier production-confirmation results remain archived separately.

## Method sources

- [RDKit conformer-generation documentation](https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation): ETKDG, pruning, seeds, and threading. ETKDG structures can often be used without a subsequent force-field minimization, which motivates measuring that alternative.
- [RDKit force-field helpers](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html): explicit MMFF94/MMFF94s/UFF mechanics and optimization controls.
- [RDKit shape alignment](https://www.rdkit.org/docs/source/rdkit.Chem.rdShapeAlign.html): underlying shape/color alignment implementation.
- [Riniker and Landrum, 2015, DOI 10.1021/acs.jcim.5b00654](https://pubmed.ncbi.nlm.nih.gov/26575315/): original ETKDG method. The abstract was checked; this note does not claim a full-text review or derive a universal conformer count from that paper.

## Literature consultation, 2026-10-27

The decision is a CPU-cost versus production-score-stability budget for an
EGCG-referenced RNN scorer. The inspected baseline is MMFF94, an 8 kcal/mol
relative-energy filter, 1.25 Å clustering, three retained references, and 30
candidate requests. Implementation evidence is the launch revision in the
manifest plus its source snapshots; the generator's failed-ID correction is
documented above. Search terms included "ETKDG conformer generation",
"epigallocatechin gallate conformation solvent", and "freely available conformer
generation methods". Only free sources were used.

[Botten et al. (2015), Structural Properties of Green Tea Catechins,
DOI 10.1021/acs.jpcb.5b08737](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/48751449/acs_2Ejpcb_2E5b08737.pdf)
was inspected in full-text form, especially Methods (journal page 12861) and
Conclusions (12865). The authors fitted GAFF torsions against DFT and studied
explicit-solvent MD/metadynamics. Default GAFF overestimated one torsional barrier
by a factor of four; water only subtly changed the resulting torsional landscape.
This supports checking force-field assumptions and challenges any automatic
claim that a water-like dielectric must improve the reference. It does not
validate MMFF dielectric settings, specify a DrugEx conformer count, or equate
similarity with biological activity.
Its eight optimized structures were used for force-field parametrization;
that number is not evidence that eight ROCS reference conformers suffice.

[Ebejer et al. (2012), DOI 10.1021/ci2004658](https://pubs.acs.org/doi/abs/10.1021/ci2004658)
compared conformation recovery, diversity, and runtime on 708 molecules; its
abstract was inspected. It predates ETKDG and does not justify transferring a
numerical budget into the current pipeline. The RDKit documentation above
supports testing unminimized ETKDG as an alternative rather than assuming
additional minimization improves task performance.

The unresolved scientific gap is bioactive-pose or active/decoy validation.
This experiment measures numerical sufficiency within its finite grid and
stress panel. Its engineering tolerances are explicit choices, not literature
constants. Switching the production objective, solvent treatment, protonation
policy, or RL settings remains a research decision for the user after reviewing
the evidence.
