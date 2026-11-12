# MetaCentrum runbook — staging, submitting, watching, collecting

MetaCentrum is an **ad-hoc** host for this thesis: occasional jobs, used when the local
workstation or `dichtator` cannot deliver the cores. It is not a production queue that
mirrors the local tree.

If you (or an agent) are asked to "run X on meta", this file is the whole contract. Do not
invent a new shell script, do not edit `pbs_launcher.sh` or `job.py` for one experiment, and
do not call `qsub` directly. `submit.py` stages a release, writes the job spec and submits
it; the PBS side is a small launcher plus `job.py`, and neither changes per run.

---

## 0. Four rules

1. **Defaults live in code.** `_david/code/config.py::Config` is the single source of truth.
   A submission carries **only the deltas** you actually want. "Defaults, but no QED and a
   20-epoch cap" is `include_qed=false epochs=20` and nothing else. If a knob is not a field
   of `Config`, it is a PBS resource (`--ncpus`, `--mem`, `--scratch`, `--walltime`,
   `--queue`) — nothing else exists.
2. **One entry point.** `python -m _david.code.metacentrum.submit …` for one job,
   `python -m _david.code.metacentrum.campaign …` for a planned batch. Both stage a release
   through the same code.
3. **Commit before submitting.** A job runs `git archive <revision>`, so a fix travels
   through a commit, never through an edit on storage. `submit` refuses a dirty tracked tree,
   because those changes would silently not be in the job.
4. **Default to `q_1d` at 32 GB.** The 24 h walltime routes to `q_1d`, the only lane used for
   training. 32 GB places roughly 3× faster than 64 GB; the job body caps scoring workers to
   fit it. Ask for 64 GB only for the heavy cell that grows past ~40 GB.

`PY` below means the local project interpreter:

```bash
PY="$HOME/.local/share/venvs/global/bin/python"
```

---

## 1. First contact

```bash
$PY -m _david.code.metacentrum.submit check
```

It prints, in the order things fail:

- the login node it reached,
- whether `$STORAGE/.venv/bin/python` exists,
- whether HEAD is staged as a release,
- which release `$STORAGE/current` points at,
- the default lane (`24:00:00 -> q_1d`, the training lane),
- a reminder for `python -m _david.code.metacentrum.queues`,
- your current queue, or "nothing queued".

An SSH refusal is almost always an expired Kerberos ticket. The ticket needs interactive 2FA,
so `submit` will not try to renew it for you; it tells you the command:

```bash
ssh meta kinit kolardavidcz@META
```

The storage layout is fixed:

```
/storage/brno2/brno2/kolardavidcz/DrugEx/
├── .venv/                     validated environment; every job uses this interpreter
├── models/                    pretrained .pkg/.vocab, staged by sync
├── releases/<sha>/            immutable `git archive` of one commit
├── current -> releases/<sha>  newest release (used by `watch`)
├── jobs/<name>/spec.json      exact argument vector of one submission (provenance)
└── _david/output/<run>/       results copied back from scratch
```

---

## 2. Submitting one run

### 2a. The minimal command

```bash
$PY -m _david.code.metacentrum.submit submit --name egcg_probe
$PY -m _david.code.metacentrum.submit submit --name egcg_probe epochs=20 include_qed=false
```

Deltas are **bare `NAME=VALUE` pairs**, not pipeline flags. Each one becomes
`--set NAME=VALUE` inside the job. Values are coerced exactly as `_david.code.run` coerces
them:

| Field type | Write it as | Example |
| :--- | :--- | :--- |
| boolean | `true` / `false` | `minimize=false` |
| integer | bare integer | `epochs=40` |
| float | bare decimal | `epsilon=0.05` |
| string | bare token | `scheme=WS`, `color_ff=fdef` |
| tuple | comma list in parentheses, quoted for the shell | `'fixed_thresholds=(1.0,1.0)'` |
| empty tuple | `()` , quoted for the shell | `'fixed_thresholds=()'` |

Anything malformed — an unknown field, a typo in a value, a stray `--flag` — is rejected
**locally**, before the release is staged or `qsub` runs. `output_root` and `n_jobs` are
owned by the job body (the scratch directory and the allocated cores) and are refused.

Useful deltas, all fields of `Config`:

```bash
# shorter, cheaper probe
epochs=20 patience=8 min_epochs=10 n_final=1000

# exploration breadth
epsilon=0.1

# no relaxation of candidate conformers (faster, raw ETKDG poses)
minimize=false

# bigger/smaller reference bank and candidate conformers
reference_conformers=4 max_conformers=30

# no QED objective
include_qed=false

# DrugEx weighted sum instead of Pareto crowding distance
scheme=WS
```

There is no preview of the job's effect and no `--dry-run`: a submission that prints
resources but does not queue anything is a rehearsal, and rehearsals hide the failures that
matter (ACL refusals, expired tickets, a bad resource expression). What the command prints
**before** `qsub` is the record of what was actually requested:

```
release : /storage/brno2/brno2/kolardavidcz/DrugEx/releases/<sha>
queue   : q_1d (default router)
resources: select=1:ncpus=16:mem=64gb:scratch_local=15gb:spec=8.0, walltime=24:00:00
deltas  : --set epochs=20 --set include_qed=false
```

If those lines say something you did not intend, the job is already queued: cancel it, fix
the invocation, submit again.

### 2b. Naming

`--name` is both the PBS job name and the run-directory tag. It must match
`[A-Za-z][A-Za-z0-9_-]{0,63}` and should say what the run is: `egcg_gold_gmean`,
`dnmt1_c30_unmin`, `ws_e05_noqed`. The run directory becomes
`$STORAGE/_david/output/YYYYmmdd_HHMM_<name>` (collision → `_2`, `_3`), so two jobs with the
same name remain distinguishable, but `cancel <name>` and `logs <name>` would consider both.

### 2c. Resources and queues

The default is a CPU job: **16 cores, 32 GB RAM, 15 GB scratch, 24 h, no GPU, no `-q`**.
PBS routes it by walltime to **`q_1d`**, the default and only lane used for training. ROCS
scoring and conformer generation are CPU-bound; the RNN itself is tiny, so this is the
scientifically relevant shape.

**Why 32 GB and not 64:** a 32 GB request places on far more nodes and, on our account,
has queued roughly **3× faster** than 64 GB. The trade is headroom: the heaviest measured
cell grew to about **40 GB around epoch 30** at 16 workers. The job body therefore derives
the scoring-worker count from the memory request:

| `--mem` | scoring workers (16 cores requested) | what fits |
| ---: | ---: | :--- |
| 32 GB | 12 | default; keeps the pool below 32 GB |
| 64 GB | 16 (all cores) | the heavy/minimised cells with ~40 GB peaks |
| 128 GB | 16 at 32 cores | wide runs with many workers |

That cap changes epoch time, never a score. It is automatic — do not pass a worker flag.
`Job` still refuses a pairing below ~2 GB per core (`16 cores` → at least `32 GB`), because
raising cores without memory is an OOM kill after the queue wait.

**Occasional extra memory.** PBS reserves the memory you request for the whole run and does
not grow it mid-job; there is no "start at 32 and wait for 64". If a run is known to grow
past 32 GB (the minimised 20-conformer cell is the one we measured), ask for 64 up front:

```bash
$PY -m _david.code.metacentrum.submit submit --name heavy --mem 64
$PY -m _david.code.metacentrum.submit submit --name wide --ncpus 32 --mem 128
```

That job waits longer for a node that can satisfy 64 GB, but it is not killed at 32. This is
the honest version of "wait until more RAM is available": request the amount the run needs
and let PBS queue it. A 32 GB job that reaches 40 GB is an under-request, not a scheduling
bug.

Queue selection is a consequence of the walltime, not a knob:

| `--walltime` up to | Queue through the `default` router |
| :--- | :--- |
| 2 h | `q_2h` |
| 4 h | `q_4h` |
| 24 h | **`q_1d` — the default training lane** |
| 48 h | `q_2d` |
| 96 h | `q_4d` |
| 168 h | `q_1w` |
| 336 h | `q_2w` |

MetaCentrum frontends cannot submit to an execution queue directly: `q_2h`…`q_1d` are
`from_route_only` and `p2e_1d` has an `acl_hosts` restriction. Ask for the compute the run
needs; **do not shorten the walltime to land in a shallower queue**, because a job that needs
20 h but asked for 2 is killed at hour two.

**`q_4h` is not for training.** A training run needs the whole day; four hours is for
probes, inspections and score-only checks. `q_2d` is only for a minimised run whose honest
budget exceeds 24 h. `--queue` is for a route or special queue an administrator grants, and
is otherwise unnecessary. The CLI is CPU-only: GPU queues are deep and this pipeline does
not need one.

### 2d. How deep is the queue?

```bash
$PY -m _david.code.metacentrum.queues
$PY -m _david.code.metacentrum.queues --json
```

It reads `qstat -Qf` and shows only the three lanes that matter (example snapshot from
2026-11-04; the script always prints the live numbers):

```
QUEUE    RUNNING  QUEUED  HELD   TOTAL  MAX WALLTIME  QUEUED/RUNNING
q_4h         471      28     0     499      04:00:00            0.1x
q_1d        1581    3873     0    5454      24:00:00            2.4x
q_2d         267    4986     0    5253      48:00:00           18.7x
```

`q_1d` is the default (24 h routes there and every training run asks for it). `q_4h` is
probes only. `q_2d` is only for a run that honestly needs more than a day. Depth moves
hourly; use it to time a submission, not as an ETA promise. The `QUEUED/RUNNING` ratio is
a rough comparison between lanes; it is not a wait estimate.

### 2e. Rerunning a staged release

Every submission normally stages HEAD. Two flags make that explicit:

```bash
# stage and run an older commit
$PY -m _david.code.metacentrum.submit submit --name rerun --revision <sha>

# use a release that is already on storage; never archive or upload
$PY -m _david.code.metacentrum.submit submit --name rerun --revision <sha> --no-sync
```

`--revision` without `--no-sync` still stages that commit (safe, idempotent). `--no-sync`
refuses to guess: it requires `--revision` and fails if that release directory is absent.
This is the way to repeat a known-good run after later commits, without pretending the newer
tree is the older experiment.

---

## 3. While it runs

```bash
$PY -m _david.code.metacentrum.submit status
$PY -m _david.code.metacentrum.submit watch
$PY -m _david.code.metacentrum.submit watch -i 60 --job egcg_probe
$PY -m _david.code.metacentrum.submit logs egcg_probe -f
```

`status` reads **long-format `qstat -f`**, resolving ids through `qselect` first, because the
short table shifts columns with long job names. It prints state (`R` running, `Q` queued,
`H` held), queue, elapsed/limit, job id and name.

`watch` runs `monitor.py` on the login node. It finds each running job's scratch directory on
its worker, reads `rl_fit.tsv`, and prints the last epochs with a per-epoch rate, an ETA
against the walltime, and two warnings worth acting on:

- `! ETA exceeds the remaining walltime` — the run will be killed before the epoch cap.
  Cancel and resubmit with a longer walltime if the result matters.
- `! no new epoch for 30+ min` — a stuck molecule, a hung scorer, or a dead process; read the
  PBS log.

If scratch is not reachable yet (the job is still in setup) it falls back to the PBS log.
`watch` reads `$STORAGE/current/.../monitor.py`, so it shows the monitor from the newest
staged release, not from your local tree.

`logs` resolves the log of the **live job id** first. It will not show an older attempt's log
just because the name was reused; if a live job has not written its log yet, it says so.
`-f` follows through an interactive SSH session.

---

## 4. Finishing

```bash
$PY -m _david.code.metacentrum.submit pull                    # every run
$PY -m _david.code.metacentrum.submit pull --name egcg_probe  # one run tag
```

A job copies its run directory back on success, failure, `SIGTERM` (walltime) and `SIGINT`,
so partial results survive a kill. `pull` rsyncs `$STORAGE/_david/output/` into
`_david/output/`. It refuses to pretend when nothing is there yet: a run directory appears on
storage only once the job has copied something back.

Result artifacts worth checking first:

| Path | Meaning |
| :--- | :--- |
| `manifest.json` | state, config, source revision, timings, error (if any) |
| `config.json` | fully resolved config actually used |
| `run.log` | the run's own log |
| `rl_fit.tsv` | one row per training epoch; what `watch` reads |
| `models/desired_ratio_at_stop.pkg` | the single checkpoint: the live policy at stop |
| `final_desired_ratio_at_stop.tsv` | 5 000 molecules sampled from that checkpoint |
| `reference.sdf`, `prior.tsv` | the reference bank and the calibrated prior arm |
| `analysis/` | figures and report; a failed analysis leaves the run otherwise intact |

There is no best-epoch checkpoint and no `gmean` arm in new runs. If a run is incomplete,
`analysis` names the missing final arm; `final_desired_ratio_at_stop.tsv` is the only one new
runs require.

After pulling, verify before deleting anything remote (compare local and remote output trees
by name and size; a clean comparison has no one-sided files). Deletion is a deliberate manual
step, not something `pull` does.

---

## 5. Cancelling

```bash
$PY -m _david.code.metacentrum.submit cancel egcg_probe          # all jobs with that name
$PY -m _david.code.metacentrum.submit cancel 25256877            # by id (full id is fine)
```

Names are resolved through `qselect -N`; a name with no queued job is reported, not silently
ignored. Cancellation still lets `job.py` copy partial results back.

---

## 6. Batches: `campaign.py`

`campaign.py` owns the planned sweep. A batch is a full factorial over its declared axes and
nothing else; every setting another batch varies is pinned in `COMMON`, so a spec states the
whole experiment.

```bash
$PY -m _david.code.metacentrum.campaign                 # plan every batch, submit nothing
$PY -m _david.code.metacentrum.campaign parent          # plan one batch
$PY -m _david.code.metacentrum.campaign parent --limit 2 --submit
```

Planning prints each run's name, memory and deltas, plus a worst-case core-hour count,
duplicate configurations (replicates, not independent runs) and every batch's purpose. With
`--submit`:

1. every delta of the batch is validated locally, so one typo stops the batch before the
   first `qsub`;
2. HEAD is staged **once**, and every job in the batch points at that revision;
3. names already present in `$STORAGE/jobs` are skipped (a spec outlives the queue entry).
   `--again` queues them anyway; `--limit N` queues only the first N.

If a `qsub` fails halfway, the command prints `submitted N of M` in a `finally` block; a
resume skips the specs already written and continues from there. Changing a batch's axes
changes the generated names, so a resumed batch is expected to reuse the same name for the
same settings.

---

## 7. Failure triage

| Symptom | Cause and fix |
| :--- | :--- |
| `ssh meta was refused` / `kinit` hint | expired Kerberos ticket: `ssh meta kinit kolardavidcz@META` |
| `tracked files are modified` | commit first; the job would not contain those edits |
| `these files are not in HEAD` | the release would be incomplete; commit the listed paths |
| delta rejected locally | typo in a `Config` field or value; nothing was submitted |
| `--no-sync needs --revision` | it never guesses which release to use |
| `release … is not staged` | run `sync` or drop `--no-sync` |
| `output_root` / `n_jobs` refused | the job owns the scratch directory and core count |
| state `H` (held) | read the `comment` column in `status`; usually an account/ACL issue |
| job vanishes with no result | read `logs <name>` and the pulled `run.log`; the PBS log has the traceback |
| `no PBS log` | the job never started or the name is wrong; `status` decides |
| `scratch not reachable` in `watch` | normal during setup; it resolves once `job.py` creates the run directory |
| killed at the walltime | partial results are copied back; resubmit with a larger `--walltime` |
| `nothing to pull` | storage has no run directory for that tag yet; the job may still be running |
| analysis failed after a successful run | the run is intact; rerun `python -m _david.code.analysis <run-dir>` |

Read the **first** traceback in the PBS log, not the last line: a `SIGTERM` from walltime
leaves a shutdown trace after the real cause.

---

## 8. What the job actually does

1. `sync` runs `git archive <revision>` locally, unpacks it to
   `$STORAGE/releases/<sha>/`, repoints `$STORAGE/current`, and rsyncs the pretrained
   `.pkg`/`.vocab` to the release. An existing release is not mutated; a job can never drift
   from its code.
2. `submit` writes `$STORAGE/jobs/<name>/spec.json` with the exact argument vector, release
   path and revision.
3. `qsub` starts `pbs_launcher.sh` with `RUN_SPEC`, `STORAGE_DIR` and `RELEASE_DIR` in the
   environment. PBS spools the script outside the release, so the release path is passed
   explicitly — never derived from the launcher's own location. The launcher checks the three
   variables and the storage venv, then `exec`s
   `$STORAGE/.venv/bin/python $RELEASE_DIR/_david/code/metacentrum/job.py --release …`.
4. `job.py` validates the spec, copies the release into `$SCRATCHDIR/DrugEx`, creates the run
   directory under `$SCRATCHDIR/output`, pins thread pools, and runs
   `python -u -m _david.code.run --tag <name> --set output_root=… --set n_jobs=<cpus> <deltas>`.
   Temp files go to `$SCRATCHDIR/tmp`, never to home storage.
5. A `finally` block copies the run directory to `$STORAGE/_david/output/<run-id>` on
   success, failure, `SIGTERM` and `SIGINT`, then cleans scratch. A failed transfer keeps
   scratch alive and exits non-zero.

Consequences worth remembering: a job is immune to later local edits; the checkpoint and
vocabulary come from the staged release; and partial results survive a walltime kill.

---

## 9. Checklist before saying "submitted"

- [ ] `git status --porcelain --untracked-files=no` is empty (or `sync` would have passed).
- [ ] The submission block showed the intended release, queue, resources and deltas — and
      nothing else.
- [ ] The job name is unique and descriptive; it becomes the run tag.
- [ ] `status` shows the job (`Q` is fine; `H` means held).
- [ ] You know the queue (`q_1d` by default) and the expected wall-clock cost.
- [ ] `watch` shows epochs once it starts; if the ETA exceeds the walltime, decide now
      whether to cancel and resubmit with a longer budget.
- [ ] After it finishes: `pull`, check `manifest.json.state`, then read `analysis/`.
