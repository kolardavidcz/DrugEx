# MetaCentrum help

Tutorial chapters for running DrugEx on MetaCentrum (`ssh meta`, user `kolardavidcz`).
They exist for the things that go wrong *around* a job: queue behaviour, progress
inspection, transfers, quotas, and the failure modes we have actually hit.

The command contract lives in [`../RUNBOOK.md`](../RUNBOOK.md). These chapters assume you
have read it and explain **why** and **what to do when it breaks**.

| Chapter | Use it when |
| :--- | :--- |
| [`01-submit.md`](01-submit.md) | You need to understand releases, specs, or how arguments reach the pipeline. |
| [`02-monitor.md`](02-monitor.md) | A job is queued/running and you want epochs, ETA, or a health check. |
| [`03-collect.md`](03-collect.md) | Results are ready: pull, verify, audit, then free storage. |
| [`04-failures.md`](04-failures.md) | A job failed, is stuck, was killed, or `qsub`/`ssh` refused. |
| [`05-resources.md`](05-resources.md) | Choosing cores, memory, walltime, queue, GPU vs CPU. |
| [`06-storage.md`](06-storage.md) | Quotas, scratch semantics, what may be deleted and what must be kept. |
| [`monitor_live_epochs.py`](monitor_live_epochs.py) | Executable helper used by chapter 02 (stdlib-only, runs on the login node). |

## State at a glance (2026-11-04)

- No jobs queued, no jobs running. Queue checks in the chapters are live commands, not
  recorded output.
- Storage `$STORAGE` holds `.venv/`, `releases/`, `current`, `models/` only. All older runs
  were checksum-verified against WSL and deleted from the cluster; `_david/output` on
  storage is empty until a new job finishes.
- Local `_david/output/` (48 run directories, ~5.6 GB) is therefore the **only** copy of
  historical results. Treat it as unbacked-up data.

## If you only remember one thing

A MetaCentrum job runs an **immutable release**, not your working tree. Editing files after
submitting changes nothing; a fix requires a new commit and a new job. That property is
what makes cluster results trustworthy, and forgetting it is what makes an afternoon
disappear.
