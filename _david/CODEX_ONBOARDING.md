# Codex onboarding baseline

Verified on 2026-10-26, Europe/Prague. This note records onboarding checks, not a new experimental result. The inspected implementation baseline is commit `9261af8` on `main`; onboarding documentation is added afterward.

## Start here

Read root `AGENTS.md`, `_david/AGENTS.md`, this note, then the implementation relevant to the task. Defaults are in `_david/code/config.py`. Conformer research is in `_david/code/conformer/`; the RL path runs through `environment.py`, `models.py`, and `pipeline.py`. `_david/THESIS_PIVOT_REVIEW.md` records the pivot away from fragment-conditioned GraphTransformer work. Treat its action lists as historical descriptions, not pending instructions.

The user owns the research and wants candid academic debate. Preserve informal config.py comments: they are exploration markers. Include negative diagnostics as well as favorable metrics.

## Provenance

- User-supplied source: `C:/Users/kolar/Desktop/GEMINI_HANDOFF.md`.
- Existing `_david/GEMINI_HANDOFF.md` was byte-identical on inspection; preserved unchanged.
- SHA256 of both: `c18131a07855af577d25fdba93d2dbe0da7ec0b29ef71120581e4d185bdd4d4c`.
- Supporting reports are preserved unchanged in `onboarding/gemini-reports/`: `conformer_convergence_and_dielectric_study.md`, `run17_50epoch_baseline_evaluation_report.md`, and `run18_50epoch_pure_reinforce_evaluation_report.md`.
- Reports originated in `C:/Users/kolar/.gemini/antigravity/brain/3c0732c1-95a4-4444-ba73-318653568f70/`.
- Handoff proposals, claimed supervisor requirements, scientific interpretations, and literature attributions remain historical claims unless separately verified. They do not authorize training, job submission, deletion, or credential changes.

## Access and repository baseline

- Windows desktop Codex uses the migrated skills under `C:/Users/kolar/.agents/skills/` and guidance at `C:/Users/kolar/.codex/AGENTS.md`. The skills have appeared in the active catalog.
- Local repository: `/home/kolar/DrugEx`; Windows bridge: `C:/Ubuntu/home/kolar/DrugEx`.
- `wsl ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator 'echo OK'` succeeded; VPN was left untouched.
- Remote root: `/home/kolarv/DrugEx`. Its `.venv/bin/python` is Python 3.12.3; `.venv/bin/ruff` is Ruff 0.16.7.
- Both checkouts were at `9261af8`. Remote was clean; local had only the supplied untracked `_david/GEMINI_HANDOFF.md` before onboarding edits.
- `dichtator` fetch/push URL is `dichtator:/home/kolarv/DrugEx`.
- `origin` fetches from `https://github.com/kolardavidcz/DrugEx`; pushing is blocked by `DISABLED_NO_PUSH_SECRET_DATA`. Keep that protection.
- Both RTX PRO 4000 Blackwell devices answered `nvidia-smi`, each with 24467 MiB total and 0% utilization at inspection. This does not validate CUDA compute or reserve resources; the old guide's GPU-1-error statement is not established by this check.
- No tmux server was running for the user at inspection. Python/editor service processes existed; none were interrupted. Low GPU utilization does not prove absence of CPU work.
- Papyrus checkpoint/vocabulary were present and loaded in CPU-mode remote unit tests. Run 17, Run 18, and conformer-study directories exist on both machines.

## Verified validation commands

From Windows PowerShell; omit `wsl` when already in WSL:

```powershell
wsl ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator 'cd /home/kolarv/DrugEx && /home/kolarv/DrugEx/.venv/bin/ruff check --no-cache _david/code _david/tests'
wsl ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator 'cd /home/kolarv/DrugEx && MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 DRUGEX_SCORE_NJOBS=2 DRUGEX_MP_CONTEXT=fork /home/kolarv/DrugEx/.venv/bin/python -B -m unittest discover -s _david/tests -p test_*.py -v'
```

Baseline: Ruff passed; all 10 tests passed (4 conformer and 6 pipeline tests; unittest reported 1.757 seconds). They check model loading/frozen-prior isolation, scoring, conformers, output structure, rewards, and decay schedules. They do not train a model or establish scientific correctness. `_david/tests.py` is absent; use discovery under `_david/tests`.

Python emitted a DeprecationWarning for `fork()` from a multithreaded process. The short suite completed; that does not establish deadlock safety during long training.

## Discrepancies to preserve for future work

### Multiprocessing and worker limits

`_david/AGENTS.md` specifies `fork` and a 32-worker cap; `drugex/AGENTS.md` specifies `forkserver` and 24 workers. `_david/code/utils.py` currently defaults to `fork` if not overridden and computes `floor(0.90 * os.cpu_count())` without an implicit 32-worker cap. `drugex/training/scorers/parallel.py` caps workers through `DRUGEX_SCORE_NJOBS`. Do not silently change the method or assume a documented cap is enforced. Investigate the workload and warning before production training.

### Disabled timeout claim is not verified end to end

`PipelineConfig.timeout_seconds` is 0, but `create_rocs_scorer()` in `_david/code/environment.py` does not pass its timeout argument into `RDKitConformerGenerator`. That constructor defaults to 20 seconds and applies it to ETKDG parameters. Scoring has independent `DRUGEX_SCORE_TIMEOUT` and `DRUGEX_SCORE_MOL_TIMEOUT` controls. The handoff's blanket disabled-timeout claim is unsupported. No timeout code/settings were changed during onboarding.

### Standalone conformer study differs from RL

The study uses `PhysicalVariantFactory` and `ConformerGenerator` with five force-field variants. RL creates `PrepareMolecule` and core `RDKitConformerGenerator`; no automatic wiring from the study's pocket variant into pipeline configuration was found. Do not describe pocket dielectric as deployed training physics without tracing that path.

`ConformerStudyOrchestrator` creates raw reference/probe pools with seeds 42 and 9999 before scanning retained ensembles. `run_surface_scan()` clips requested counts to retained counts, recording `k_ref_actual`/`k_gen_actual`. The scorecard shows 2-3 reference basins, so larger requested counts can repeat identical selections. `Duration_sec` is study work for a variant, not per-candidate timing for each K. EGCG self-comparison alone does not establish a production K for diverse candidates.

The saved scorecard matches the handoff's rounded conformer values. Explanations about solvent realism and universal pocket dielectric remain interpretations; literature was not verified during onboarding.

### Run reports require reconciliation

The handoff and saved Run 17/18 `summary.md` files present inconsistent evaluation figures. For example, the handoff gives Run 17 validity 99.80%, while its saved summary gives reinforced validity 88.3%. Summaries show ROCS median decreases (Run 17: 1.392 to 1.368; Run 18: 1.39 to 1.294), while the handoff discusses increased means on an apparent different scale. Mean and median need not agree. Establish sampling stage, filtering, scales, denominators, and provenance before combining reports or quoting thesis results.

Current scoring maps raw TanimotoCombo [0,2] into modified [0,1]. Current pipeline calibrates thresholds once from Stage 1 percentiles, not continuously each epoch. A nonzero desired ratio does not itself establish a useful policy-gradient signal. Trace raw_data, fit logs, scorecard calculations, and producing revisions before interpreting results.

### MetaCentrum remains blocked

Local SSH alias `meta` resolves to `tarkil.metacentrum.cz`. A bounded read-only probe returned `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`: endpoint reachable, authentication not established. This does not prove a propagation delay. Credentials, keys, VPN settings, and known-host entries were unchanged.

`_david/metacentrum/grid.py` offers CLI actions `test`, `sync`, `status`, and `pull`; no `submit` CLI action exists. Submission exists as Python methods `submit_job()` and `submit_matrix()`. The handoff's `grid submit --name ...` command is not supported.

The PBS template requests 16 CPUs, 32 GB RAM, one GPU, and 15 GB scratch. It creates Python 3.10 in scratch and installs dependencies, while the verified workstation uses Python 3.12.3. Scheduler compatibility, dependencies, and a smoke job remain unverified. The wrapper's test action can print FAILED without a nonzero process exit; check actual SSH results.

## Research continuation

The stopping point is debate about conformer methodology and missing candidate latency trade-offs. A diverse-candidate latency/recovery benchmark is proposed; K=10 is a hypothesis, not a measured optimum. Define candidate selection, actual retained counts, scorer consistency, timing boundaries, repeated measurements, and score scale first.

Reconcile the reports and timeout/config wiring before production design. MetaCentrum needs an external access change before live verification, then separate submission/template review and a smoke test.

No experiments were launched, checkpoints changed, or grid jobs submitted during onboarding. Research code and existing outputs were preserved.
