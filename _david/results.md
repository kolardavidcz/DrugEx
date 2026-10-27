# Thesis status and results

Updated 2026-10-27. Current instructions: [AGENTS.md](AGENTS.md).
This consolidates the retired onboarding, handoff, rollback and report notes.
No new conformer experiment or training run was performed for this update.

## Baseline and recovery

The restored runtime baseline is `0f6a30d`, before the first standalone conformer
experiment (`52b0d2d`). Rollback commit `8a8386c` restored `code/environment.py`,
`code/visualization.py` and six core source/stub files byte-for-byte. The only
active-code exception was `code/config.py`, preserving user annotations and
timeout=0. Follow-up `6eb989d` changed documentation only. This is a review
baseline, not a scientifically validated algorithm.

The removed Gemini/Codex standalone conformer scripts and added tests are
recoverable on local/private branch
`codex/recovery-before-conformer-rollback-20261027T185549Z` at `570fc77`.
Working-tree patches and copied untracked files are under
`output/rollback_checkpoints/20261027T185549Z/{wsl,dichtator}/`.
Datasets, models, checkpoints and raw experiment outputs were preserved.
Retired narrative documents remain recoverable from Git or the documentation
cleanup archive in that checkpoint directory. Frozen source/evidence snapshots
remain historical records, not runnable current instructions.

Dichtator contains user launcher deletions and a moved, untracked
`code/conformer/conformer_experiment.py`. Preserve them. The moved wrapper depends
on removed modules and is outside the validated baseline.

## What the existing tests establish

At `8a8386c`, all six baseline tests and Ruff passed on dichtator in the clean
checkout `/home/kolarv/.cache/drugex-rollback-validation-8a8386c` (tests: 1.782 s).
Import origins were verified. Exact commands, environment and logs are in the
checkpoint's `dichtator/validation.json`, `tests.log` and `ruff.log`.
The existing multithreaded-fork warning remains; short tests do not establish
long-running multiprocessing safety.

Inspection of `tests/test_sequencer_pipeline.py` on 2026-10-27:

| Assertion | Present coverage | Limit |
| --- | --- | --- |
| Reference generation | Aspirin SDF exists, 1–3 records, 3D flag and Z span | No stereochemical identity, energy-window or convergence assertion |
| Scoring | Benzene scores above caffeine | No exact self-score or explicit score-scale assertion |
| Molecular identity | None across preparation/SDF round trip | No deliberate wrong-stereoisomer check |
| Other pipeline behavior | Checkpoint loading, separate frozen prior, reward and exploration schedule | Not conformer-search validation |

Former self-recovery diagnostics used subsequently reverted source. They do not
validate the current implementation. A same-molecule independent-embedding score
is a diagnostic, not an assertion that every flexible molecule must score 1.
A different stereoisomer must fail an identity check; its shape/color similarity
need not fall below an arbitrary threshold.

## Unresolved implementation findings

Verified from the restored source; none is fixed by this documentation update:

- Reference SMILES standardization uses non-isomeric output and can discard
  specified stereochemistry. An inspected historical reference was S,R while the
  input/candidate was R,R. Candidate tautomer/protonation transformations also
  require an identity audit; preserving tags only at input is insufficient.
- `create_rocs_scorer()` hardcodes four isomers and does not forward its timeout
  argument. The core generator defaults to 20 seconds and clamps conformer
  requests above 200. Config timeout=0 does not disable this timeout.
- Reference preparation applies flexibility-based caps of one or three retained
  conformers. Larger requested reference limits therefore do not necessarily
  increase the ensemble. The energy window is hardcoded to 8 kcal/mol; trace
  configurable values into the live method before interpreting a grid.
- A native failed-embedding ID can reach SDF writing. Previous probes also found
  coordinate-precision loss in multiprocessing serialization; those fixes were
  reverted. Reproduce against the selected execution path before claiming parity.
- Legacy reporting can double ROCS scores twice. The stated combo scale is raw
  shape+color in [0,2]; combo/2 is a separate normalized scale. Neither is a
  percentage of biological accuracy. Current Tversky variants evaluate converted
  overlaps on the Tanimoto-driven alignments, not separately optimized poses.

## Decisions and proposed work

User requirements: molecule-agnostic implementation under `_david/`; the study
must reflect candidate preparation and stereoisomer handling used in training;
new scientific settings are proposed before measurement. A rewrite of the
unreviewed conformer study is permitted. No replacement implementation or new
parameter sweep has yet been selected/run after rollback.

Previously selected reference comparison: MMFF94s constant dielectric 1, 6.5 and
78.4; UFF; MMFF94s distance-dependent dielectric. Scores: TanimotoCombo and
TverskySubshape. These are experimental electrostatic settings, not validated
solvent/pocket environments. The training candidate baseline is unminimized
ETKDGv3. No production conformer count has been established.

Proposals awaiting selection:

- First establish preparation/SDF identity and exact-geometry scoring controls;
  inspect their raw structures and outputs before interpreting a parameter map.
- Keep isomer cap I independent of conformers per isomer C. The user's proposed
  I=max(4,C/4) is a possible coupled policy, not an implemented setting. Before
  pruning, requested work scales approximately as actual enumerated isomers × C;
  both axes increasing can obscure the source of a score improvement.
- Preserve best-over-enumerated-isomers scoring as the current baseline. For
  unspecified stereo it means the best tested stereoisomer, not a resolved
  compound or a stereoisomer-mixture average. Record the winning isomer and
  actual search counts. A fixed absolute stereoisomer should not be inverted by
  enumeration. Reference inputs with unresolved stereo need an explicit policy.
- Optional comparison: unminimized candidates versus candidates minimized by the
  matching reference method. It would test geometry-preparation mismatch at fixed
  search budget. It is not yet approved or deployed in training.
- Independent embedding seeds and a panel spanning rigid/flexible, achiral,
  specified and unspecified stereo would test sampling variability and transfer
  beyond one reference. Molecules, seeds and budgets remain to be chosen.

Literature consultation, 2026-10-27: reused and rechecked the prior stereochemistry
review; questions were preserving known stereo, enumerating unknown stereo, and
whether conformer/isomer budgets should be tied. Baseline inspected: `6eb989d`,
with no runtime edits. Full-text Gypsum-DL methods, “Unspecified chiral centers”
and “Controlling the combinatorial explosion”, preserve specified centers and
describe independent variant dimensions. That docking study does not establish
a DrugEx budget or an RL benefit: [Ropp et al., 2019, DOI
10.1186/s13321-019-0358-3](https://link.springer.com/article/10.1186/s13321-019-0358-3).
Official [RDKit stereoisomer enumeration documentation](https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html)
defines onlyUnassigned, maxIsomers and seeded subset enumeration; its online
version can differ from the installed RDKit. No inspected evidence establishes
the proposed ratio C/4 as generally optimal. Budget/objective cautions above are
code-based deductions, not measured training effects.

## Historical training evidence

Retained from the 2026-10-26 CSV review; not rerun for this update. Each arm has
1,000 generated rows, including invalid rows in the stored-column means:

| Run (50 recorded epochs) | Valid baseline → reinforced | Stored ROCS mean baseline → reinforced |
| --- | --- | --- |
| `run17_baseline_50ep` | 955/1000 → 883/1000 | 0.328771 → 0.303615 |
| `run18_epsilon_zero_50ep` | 951/1000 → 845/1000 | 0.326311 → 0.276163 |

Sources: `output/<run>/raw_data/{baseline,reinforced}_raw.csv.gz` and
`models/SequenceRNN_reinforced_fit.tsv`. Stored scales and historical parameter
provenance require reconciliation; affected scorecards/plots are not calibrated
physical results. These observations do not identify a causal hyperparameter
effect. Run 19 was incomplete at the earlier inspection. Legacy GraphTransformer
runs use different conditioning and are not matched SequenceRNN comparisons.
