# `metacentrum/` — ad-hoc PBS jobs for DrugEx

MetaCentrum is used occasionally: one job at a time, for CPU-heavy ROCS / conformer work
that neither the workstation nor `dichtator` should absorb. This package contains
everything needed to stage a commit, submit it, watch it and bring the results home.

**Read [`RUNBOOK.md`](RUNBOOK.md) first.** It is the only file an operator or agent needs
to start a run; it also states the three invariants (defaults come from `config.py`, one
entry point, commit before submitting).

| Path | Role |
| :--- | :--- |
| [`RUNBOOK.md`](RUNBOOK.md) | The template: commands, recipes, checklist. Start here. |
| [`submit.py`](submit.py) | The CLI: `check · sync · submit · status · logs · cancel · pull`. |
| [`job_template.sh`](job_template.sh) | PBS body. Runs exactly one pipeline invocation; edited only when the PBS contract changes. |
| [`help/`](help/README.md) | Long-form tutorials: submitting, monitoring, collecting, failures, resources, storage. Includes [`monitor_live_epochs.py`](help/monitor_live_epochs.py). |
| [`scripts/runed/`](scripts/runed/README.md) | Frozen PBS scripts from the grid era (provenance only, never re-run). |
| [`archive/file_folder_tree_comparasion/`](archive/file_folder_tree_comparasion/) | `compare_output_trees.py` + its usage contract, used to verify WSL ↔ storage parity before deleting remote results. |

Pipeline behaviour (hyperparameters, stages, figures) is **not** documented here — see
`_david/code/config.py`, `_david/AGENTS.md` and `_david/code/RL_RUN_AUDIT.md`.
