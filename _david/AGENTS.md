# Thesis implementation guide

## Scope

The current work uses an unprompted Papyrus SMILES SequenceRNN, an independent frozen mutation prior, and RL with ROCS, SA, and QED objectives. EGCG is the default query. Fragment-conditioned GraphTransformer runs are historical comparisons under `legacy/stupid_graph_transformer/`.

Debate methods candidly; implement the user's research choices. Preserve informal annotations in `code/config.py`. Include unfavorable diagnostics such as shrinkage, invalid/duplicate samples, and scaffold concentration.

Keep all thesis implementation changes inside `_david/`. Treat DrugEx core and
other repository paths as read-only. Report core issues and discuss any solution
within `_david/`; do not patch core code. Keep diffs small and refactor only as
needed for the requested change.

## Source map

- `code/config.py`: PipelineConfig, RunPaths, query, checkpoint resolution.
- `code/environment.py`: query preparation, reference conformers, scorers, modifiers, rewards.
- `code/models.py`: agent/frozen-prior loading and exploration schedules.
- `code/pipeline.py`: reference preparation, baseline sampling, threshold calibration, RL, final sampling, analysis.
- `code/schema.py`: column names; distinguish modified ROCS/SA from physical values.
- `code/utils.py`: hardware selection and runtime initialization.
- `code/visualization.py`: analysis, scorecards, lead ranking, diagnostic figures.
- `code/run_overnight_matrix.py`: sequential Run 17-20 launcher. It skips runs when summary.md exists and prints a final completion message even after caught errors. Inspect before reuse.
- `tests/`: current unittest suites; the former tests.py path is absent.
- `metacentrum/`: grid manager and PBS template; live authentication/submission remains unverified.
- `output/`: generated artifacts, mostly ignored by Git.
- `validation/` and `code/tmp/`: ignored research work; Git synchronization does not transfer them.

## Current behavior and open issues

- Keep agent and mutation prior separate; the prior is frozen and in evaluation mode. Model construction passes the selected GPU tuple.
- Generate with evaluator=None, then score explicitly. Use canonical SMILES for uniqueness and state its denominator.
- Thresholds are calibrated once from baseline percentiles, not continuously during RL. A nonzero desired ratio does not establish a useful optimization signal.
- Runtime defaults to DRUGEX_MP_CONTEXT=fork if unset. Python warns about forking multithreaded processes; neither fork nor forkserver is universally safe without workload-specific verification.
- HardwareManager and scorer environment settings determine workers. Old fixed 24/32-worker claims are not implemented guarantees. Check current load/allocation before machine-wide defaults.
- Config timeout_seconds=0 does not prove the core conformer generator has no timeout; trace constructor wiring.
- Reporting infers ROCS scale from maxima; nested conversions can double it twice. Do not use existing scorecards/plots as calibrated physical results until resolved.
- The standalone conformer implementations have been removed. Do not import or launch their historical scripts/results as current code.

See [CODEX_ONBOARDING.md](CODEX_ONBOARDING.md) for dated evidence. Historical handoffs/reports are not standing instructions or approved experiment parameters.
See [CONFORMER_ROLLBACK.md](CONFORMER_ROLLBACK.md) for the restored baseline,
explicit exceptions, retained evidence and known unresolved issues.

## Validation and synchronization

Edit locally with apply_patch and LF endings. Use the existing remote .venv, preserving user work. From WSL:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator 'cd /home/kolarv/DrugEx && .venv/bin/ruff check --no-cache _david/code _david/tests'
ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator 'cd /home/kolarv/DrugEx && MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 DRUGEX_SCORE_NJOBS=2 DRUGEX_MP_CONTEXT=fork .venv/bin/python -B -m unittest discover -s _david/tests -p test_*.py -v'
```

Prefix with wsl from PowerShell. These are validation examples, not production settings. Onboarding passed 10 tests; rerun relevant checks for implementation changes.

Commit task files with a detailed message and synchronize through dichtator after checking both worktrees. Never push to GitHub origin or commit checkpoints/bulky raw output.
