# Codex onboarding baseline

Verified 2026-10-26, Europe/Prague. Implementation baseline: 9261af8; initial onboarding archive: fb1d1e4. This is a dated record, not a new experiment or permission to run jobs.

## Current guidance and provenance

Read root AGENTS.md and the relevant nested guide, then the actual source. Historical guides were shortened to remove contradictory settings, unsupported scientific claims, and obsolete tool instructions.

The original Desktop handoff and three previous-agent reports are preserved in Git at fb1d1e4. Original handoff SHA256: c18131a07855af577d25fdba93d2dbe0da7ec0b29ef71120581e4d185bdd4d4c. Their current Markdown versions are corrected summaries with explicit limitations; they are no longer byte-identical archives.

The active source map is in [AGENTS.md](AGENTS.md). Preserve config.py exploration comments. The user owns research choices and expects evidence, critical discussion, and unfavorable diagnostics.

## Environment

- Local: /home/kolar/DrugEx; Windows bridge C:/Ubuntu/home/kolar/DrugEx.
- Remote: /home/kolarv/DrugEx through WSL SSH alias dichtator. Probe succeeded without touching VPN.
- Remote Python 3.12.3; Ruff 0.16.7; torch 2.14.0+cu130; RDKit 2026.3.6.
- Threadripper 9960X: 24 physical cores / 48 logical threads; 125 GiB RAM; two RTX PRO 4000 Blackwell GPUs, about 24 GB each.
- PyTorch reports CUDA available, two devices, and sm_120 support. No compute benchmark was run. Hardware availability/load must be rechecked before jobs.
- No recognised training process or tmux session was active at the SSH survey.
- origin pushes are blocked by DISABLED_NO_PUSH_SECRET_DATA. Use private dichtator synchronization after checking both worktrees.
- Remote rg and uv were not found on PATH; use scoped git grep and the existing .venv. Locally, uv is /home/kolar/.local/bin/uv; non-login WSL exec may require the absolute path.
- Papyrus SMILES checkpoint/vocabulary loaded in CPU-mode tests. Pretrained models are under tutorial/data/models/pretrained; data is a symlink to tutorial/data.

## Validation

Remote Ruff check of _david/code and _david/tests passed. All 10 unittest cases passed: 4 conformer tests and 6 pipeline tests. Commands are in the thesis AGENTS.md.

A Python warning about fork from a multithreaded process occurred. Short-suite success does not establish long-training deadlock safety. Full core-suite or type-check success was not verified.

## Experiments and ignored work

- Run 16 has 10 recorded epochs; Run 17 and Run 18 each have 50.
- Run 19 has 18 recorded epochs and a best checkpoint, but no completed reinforced evaluation/final checkpoint found. No Run 20 output directory was found.
- Run 17/18 retain best and final-epoch checkpoints; do not assume a training metric refers to the same checkpoint as the final evaluation.
- Conformer study outputs and scorecard exist.
- validation/ contains ignored docking/MD code and results: six ligands across three recorded targets (Pin1, TTR, SIRT6). Saved paths reference an older workspace. These results were not independently validated; Vina, Meeko, and PLIP were absent from the inspected current .venv.
- The eight validation Python files matched local copies. Two ignored remote code/tmp runners were absent locally. Git synchronization alone does not protect or transfer ignored work.
- Completed October 16 orchestration notes from .agents/teamwork/ were archived outside the repository at C:/Users/kolar/.codex/backups/2026-10-26-instruction-cleanup/historical-teamwork. Their forkserver, 60%-CPU, test counts, and dispatch commands described an older GraphTransformer setup and are not current guidance.

## Unresolved implementation issues

1. Reporting scale: ResultAnalyzer.analyze() calls unscale_to_physical_metrics(), then downstream scorecard/renderers can call it again. For these saved evaluations, maxima below 0.5 allow two successive doublings. Run 17/18 scorecard medians match about four times the stored medians. Do not quote their absolute physical scores until corrected.
2. Timeout wiring: config timeout_seconds=0 is not passed to the core RDKitConformerGenerator constructor, whose default is 20 seconds. Scoring has separate timeout environment variables.
3. Concurrency: current HardwareManager defaults to fork and 90% of logical CPUs. It does not impose the historical fixed 24/32-worker limits. Runtime warnings/behavior require workload-specific investigation.
4. Standalone-versus-RL physics: the five-variant conformer study is a different path from RL query preparation and candidate scoring. Pocket/water-labelled variants are not automatically deployed in training.
5. Conformer counts: grids clip requested K to retained basins. The scorecard has only 2-3 reference basins. Duration_sec is variant-study time, not candidate latency at each K. EGCG self-similarity alone does not select a production K.
6. Thresholds: the pipeline calibrates once from Stage 1 percentiles, not continuously. Desired ratio is not itself an optimization-gradient guarantee.

Run reports now distinguish direct stored-column statistics from generated scorecards; see [Run 17](onboarding/gemini-reports/run17_50epoch_baseline_evaluation_report.md) and [Run 18](onboarding/gemini-reports/run18_50epoch_pure_reinforce_evaluation_report.md). The prior handoff's 99.8-100% validity and mean-improvement claims were unsupported.

## MetaCentrum

SSH alias meta resolved to tarkil.metacentrum.cz, but authentication was denied. This does not prove key propagation delay. No credentials or VPN configuration were changed.

grid.py supports test, sync, status, and pull CLI actions. Submission is through Python methods, not a submit CLI action. Its test command can print failure without a nonzero exit.

The PBS template's Python 3.10 environment, dependencies, scheduler resource syntax, and smoke-job behavior remain unverified. No grid jobs were submitted.

## Personal skills and free literature access

The active personal set in C:/Users/kolar/.agents/skills contains drugex-dichtator, wsl-first, wsl-ide-dotfiles, pubmed-database, literature-search-europepmc, literature-search-openalex, literature-search-arxiv, and literature-search-biorxiv.

The 26 legacy personal skills were removed from discovery into C:/Users/kolar/.codex/backups/2026-10-26-instruction-cleanup/legacy-skills. Built-in and plugin-managed skills were not removed.

The five literature skills reuse the user's local Google DeepMind science-skills resources under Apache-2.0, retaining licenses and original helper scripts. Instructions were adapted for Codex, free-only access, optional credentials, and local WSL uv isolation.

Live keyless smoke checks passed for PubMed, Europe PMC, OpenAlex, arXiv, and bioRxiv. Europe PMC full-text retrieval succeeded for PMC13330364 (57,801 bytes). Its direct PDF download was blocked by a Cloudflare response; use available text/XML or another legitimate OA location and report the limitation. No bypass was attempted.

These are API-backed skills, not MCP servers. They provide the needed free discovery/full-text workflow without a paid integration. Scite was not connected; no paid plan, subscription trial, or API key was activated. Optional free API keys do not block ordinary keyless research.

Keep DOI/PMID/PMCID, source URL, access level, version, and relevant paper sections when supporting claims. Discovery metadata is not full-text evidence; preprints are not automatically peer-reviewed.

## Next research discussion

Resolve score scales and timeout/config wiring before production decisions. Define a diverse-candidate latency/recovery benchmark with actual retained counts, consistent scorers, repeated timings, and a fixed comparison protocol. K=10 and a new 100-epoch matrix remain hypotheses/proposals for the user to assess.

No research code, raw data, checkpoints, or running jobs were changed by the documentation/skill cleanup.
