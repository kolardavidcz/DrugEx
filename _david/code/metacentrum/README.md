# `metacentrum/` — ad-hoc PBS jobs for DrugEx

MetaCentrum is the occasional CPU host for this thesis: a run that needs more cores than the
workstation or `dichtator` can spare. This package stages a committed release, submits it,
supervises it and brings the results home. It is not a mirror of the local tree and it is not
a production queue.

**Read [`RUNBOOK.md`](RUNBOOK.md) first.** It is the whole operator contract: commands,
deltas, resource sizing, monitoring, failure triage and the checklist before saying
"submitted".

| Path | Role |
| :--- | :--- |
| [`RUNBOOK.md`](RUNBOOK.md) | The tutorial: recipes, queue routing, failure triage, checklist. |
| [`submit.py`](submit.py) | The CLI: `check · sync · submit · status · watch · logs · cancel · pull`. |
| [`campaign.py`](campaign.py) | The planned sweep: batches of full-factorial runs, planning by default. |
| [`monitor.py`](monitor.py) | Read-only live epoch monitor, run on a login node (stdlib only). |
| [`queues.py`](queues.py) | Queue depth for `q_4h`, `q_1d`, `q_2d`; `q_1d` is the training default. |
| [`pbs_launcher.sh`](pbs_launcher.sh) | PBS entry point: checks the environment and `exec`s the job body. No logic. |
| [`job.py`](job.py) | The job body: release copy, scratch run directory, invocation, copy-back, signals. |
| [`rebuild_arms.sh`](rebuild_arms.sh) | Rebuild only the stop arm of runs whose earlier artifacts are incomplete. |

## Quick reference

```bash
PY="$HOME/.local/share/venvs/global/bin/python"

$PY -m _david.code.metacentrum.submit check
$PY -m _david.code.metacentrum.queues
$PY -m _david.code.metacentrum.submit submit --name egcg_probe epochs=20
$PY -m _david.code.metacentrum.submit status
$PY -m _david.code.metacentrum.submit watch
$PY -m _david.code.metacentrum.submit logs egcg_probe -f
$PY -m _david.code.metacentrum.submit pull --name egcg_probe
```

Deltas are bare `NAME=VALUE` pairs naming a field of `_david/code/config.py::Config`. Nothing
is passed for a default. The default job is 16 cores / **32 GB** / 24 h, which routes to
**`q_1d`**; 32 GB places roughly 3× faster than 64 GB, and the job body caps scoring workers
to fit the request. A job runs `git archive <revision>`, so commit first; `submit` refuses a
dirty tracked tree.
