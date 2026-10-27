# Pre-conformer code baseline, 2026-10-27

The active code is restored to `0f6a30d`, immediately before the first standalone
conformer experiment (`52b0d2d`). This restores a simpler review baseline; it does
not establish scientific correctness or endorse its parameters.

## Scope and preserved work

- Restore `code/environment.py`, `code/visualization.py`, and the six modified
  DrugEx core source/stub files byte-for-byte to the baseline.
- Remove the Gemini and Codex standalone conformer implementations and their
  added tests. Withdraw the two tracked benchmark reports; their history remains
  available through Git.
- Keep `code/config.py` unchanged, including the user's annotations and the
  existing timeout=0 setting. This is the only active-code difference from the
  baseline; it does not prove the effective native timeout is unlimited.
- Keep current agent instructions, literature skills, onboarding/history notes,
  and unrelated legacy review cleanups. They are explicit exceptions to a whole
  repository rollback.
- Keep generated results, datasets, checkpoints and uncommitted user work in
  place. `conformer_self_recovery/` and `output/conformer_benchmark_20261027/`
  contain historical evidence, not a current runnable study. No conformer-count
  recommendation from these studies is adopted.
- Dichtator has uncommitted launcher deletions and a moved, untracked
  `code/conformer/conformer_experiment.py`. Preserve these user edits; the moved
  wrapper is outside the restored/validated tracked implementation.

## Recovery checkpoints

The complete pre-rollback tracked state is retained on local/private remote branch
`codex/recovery-before-conformer-rollback-20261027T185549Z` at `570fc77`.
Working-tree/index patches and copies of untracked files are under
`output/rollback_checkpoints/20261027T185549Z/`, in separate `wsl` and `dichtator`
subdirectories on their respective hosts. Bulk ignored data were not copied or
deleted. Git history was not rewritten.

## Verified issues retained for later review

These findings describe the restored implementation; they are not new fixes or
selected research methods:

- Reference SMILES standardization uses non-isomeric output and can discard
  specified stereochemistry. The inspected historical reference was S,R while
  the input/candidate was R,R. The later study's identity correction was rolled
  back with its other implementation changes.
- The core RDKit generator clamps requests above 200. Factory timeout arguments
  are not forwarded, leaving the generator's default 20-second setting effective.
- Native failed embedding IDs can reach the SDF-writing loop. A previous native
  reproduction showed a failed embedding being written as a 2D record.
- Default RDKit multiprocessing serialization can reduce coordinate precision;
  previous probes showed resulting overlay-score differences.
- Existing reporting infers score scaling from observed maxima. Raw combo and
  modified scores need explicit verification before interpreting its plots.

Future work must be approved, small and implemented under `_david/`. Keep
observations about scoring separate from claims about RNN training outcomes.

## Validation

Revision `8a8386c` was validated on dichtator in the clean isolated checkout
`/home/kolarv/.cache/drugex-rollback-validation-8a8386c`. Core/thesis import paths
were checked to resolve to that checkout, rather than the active working tree.

- `.venv/bin/ruff check --no-cache _david/code _david/tests`: passed.
- `.venv/bin/python -B -m unittest discover -s _david/tests -p 'test_*.py' -v`:
  all six restored baseline tests passed (1.782 seconds).
- Test environment: `MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
  DRUGEX_SCORE_NJOBS=2 DRUGEX_MP_CONTEXT=fork`. Existing Papyrus files were used
  through the resolver's absolute-path fallback; no checkpoint was copied.
- The existing warning about fork from a multithreaded process remains recorded.

Exact commands, logs, revision, import origins and clean before/after status are
in the dichtator checkpoint's `validation.json`, `ruff.log` and `tests.log`.
These checks cover loading and basic component behavior; the scientific issues
above remain unresolved. This documentation-only follow-up does not change the
validated source.
