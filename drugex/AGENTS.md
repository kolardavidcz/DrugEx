# DrugEx core guide

Read root AGENTS.md and the relevant source before changing core behavior. Prefer thesis orchestration under _david/ when existing core functionality supports it.

## Source map

- data/: vocabularies, datasets, corpus processing, fragmentation, splitting.
- molecules/: molecular wrappers, converters, suppliers.
- training/generators/ and training/explorers/: generators and RL explorers.
- training/environment.py, rewards.py, monitors.py: scoring, aggregation, training monitoring.
- training/scorers/: property/QSPR scorers, ROCS backends, modifiers, conformer generation.
- training/scorers/parallel.py: multiprocessing context, worker limits, timeouts.
- parallel/ and utils/: shared processing/utilities.
- dataset.py, train.py, generate.py: CLI entry points.

Many modules have adjacent .pyi stubs. Inspect and update the corresponding stub when changing a public runtime signature. Do not assume every module has one. The previously documented scripts/extract_git_stubs.py is absent. This guide makes no blanket mypy/pyright success claim.

## Scoring and concurrency

Trace the selected backend and modifiers before interpreting scores or thresholds. Distinguish raw [0,2] TanimotoCombo from modified [0,1] values; never infer scale solely from an observed maximum.

pool_context() honors DRUGEX_MP_CONTEXT; capped_n_jobs() honors DRUGEX_SCORE_NJOBS. Thesis runtime currently defaults to fork. Warnings and deadlocks depend on workload and initialization order; historical 24/32-worker limits and universal forkserver requirements are not implemented guarantees.

Embedding and scoring timeouts are separate controls. Inspect constructor arguments/environment before claiming timeouts are disabled. Standalone conformer studies and RL use different paths.

## Validation and Git

Use the existing remote Python and Ruff. Core suites are training/tests.py and training/scorers/tests.py; inspect scope/dependencies before running. The prior 10-test onboarding result covers _david/tests, not all core suites.

Stage task files. Follow the user's synchronization rule: private dichtator pushes after checking both checkouts; never GitHub origin. Preserve unrelated changes. Historical blanket bans on all pushes are superseded.
