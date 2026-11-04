# 04 — Failures and rescue

Every failure has one of five origins: **infrastructure** (SSH/Kerberos/queue), **request**
(resources the scheduler will not grant), **environment** (`.venv`, checkpoint, stage),
**code** (traceback), or **time** (walltime, early stop, stuck molecule). Diagnose in that
order; it is also the order in which the messages are unambiguous.

## 1. First response (60 seconds)

```bash
# 1. What state is it in, and did it ever run?
$PY -m _david.code.metacentrum.submit status
ssh meta 'qstat -x -f <id> | grep -E "job_state|Exit_status|resources_used|comment"'

# 2. What did the log say at the end?
$PY -m _david.code.metacentrum.submit logs <name> -n 120

# 3. Did the run directory make it back to storage?
ssh meta 'ls -la /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ | tail -5'
```

Classify before acting:

| Symptom | Origin | Go to |
| :--- | :--- | :--- |
| Never left `Q`, or `H` with a comment | infrastructure / request | §2 |
| `ssh`/`qsub` refused with ticket text | infrastructure | §3 |
| Died in the first seconds, no pipeline output | environment / request | §4 |
| Python traceback | code | §5 |
| Ran for hours then vanished | time / memory / scratch | §6 |
| Finished, no run directory on storage | transfer | §7 |

## 2. Queued forever, or held

```bash
ssh meta 'qstat -f <id> | grep -E "comment|estimated|Resource_List.select|queue"'
ssh meta 'qstat -q | grep -E "^(Queue|q_2h|q_4h|q_1d|q_2d|gpu_long)"'
```

- `H` with `comment = Job held by ...` — an administrator or the scheduler held it; usually
  the requested resources exceed the queue's per-job limit. Resubmit smaller.
- `Q` with a huge queue depth — normal for `q_1d` (thousands waiting) and `gpu`. You cannot
  pick a different execution queue, and trimming `--walltime` to reach `q_2h` just guarantees
  a killed job. Wait, or kill and resubmit only if the budget was actually wrong. Never ask
  for a GPU unless the workload truly needs one.
- `Q` with `comment = Not Running: Insufficient amount of resource` and no queue depth —
  the request is unsatisfiable: `--mem` above the node class or `--ncpus` above the queue
  cap. Chapter 05 has the limits.
- `spec=10.0` shrinks the pool to two nodes; we default to `8.0` for that reason.

Submission-time refusals mean the queue was named explicitly:

| Message | Meaning |
| :--- | :--- |
| `Access from host not allowed, or unknown host` | `--queue p2e_1d` (or another `acl_hosts`-restricted queue) from a frontend that is not allowed. Drop `--queue`. |
| `Access to queue is denied` | Queue is `from_route_only` (`q_2h`, `q_4h`, `q_1d`, `q_2d`, …) or not granted to the account. Drop `--queue`. |
| `Unknown queue` | Typo, or the queue was retired. `qstat -q` lists the live ones. |

Cancel and resubmit rather than waiting out a bad request:

```bash
$PY -m _david.code.metacentrum.submit cancel <name>
```

## 3. SSH, Kerberos and 2FA

`submit.py` detects the common Kerberos strings and rewrites them into one instruction:

```
Kerberos ticket missing or expired on 'meta' during job submission.
Fix it interactively: ssh meta kinit kolardavidcz@META
A 2FA code is required; it cannot be refreshed non-interactively.
```

This cannot be automated: MetaCentrum requires an interactive one-time code. Ask the user
to run `kinit` (or reconnect), then retry. `ssh -o BatchMode=yes meta true` is the fast probe;
a non-zero exit means no ticket.

Other transport failures:

- `Permission denied (publickey)` — a proxy/jump host changed, or the SSH config alias is
  wrong. `ssh -v meta` shows which key and host were tried.
- `Connection timed out` on the login node — cluster maintenance; retry after the scheduled
  window.
- `rsync` of the checkpoint fails while `sync` otherwise succeeds — disk full on storage
  (chapter 06), or the checkpoint path moved. Check `$STORAGE/models` by hand.

## 4. Died immediately

Read the first 40 lines of the PBS log, not the last.

| Message | Fix |
| :--- | :--- |
| `Missing validated environment: …/.venv/bin/python` | The venv was deleted. Recreate it (chapter 06); do not point the job at an ad-hoc Python. |
| `Release is incomplete or gone: …` | `releases/<sha>` was pruned while a job still referenced it. Resubmit after `sync`. |
| `FileNotFoundError: Could not locate pretrained …` | `$STORAGE/models/Papyrus05.5_smiles_rnn_PT.{pkg,vocab}` missing. Run `submit sync`. |
| `qsub: Unknown queue` | Retired/typo'd queue. `qstat -q`, then resubmit with `--queue`. |
| `qsub: Job violates queue resource limits` | Request exceeds the queue cap. Lower `--ncpus/--mem/--walltime` (chapter 05). |
| `RuntimeError: BatchMode SSH…` | Kerberos, §3. |

The PBS log's first block is the `job.py` header. If it is absent, the job never reached our
script — the failure is entirely in PBS (request or environment), not in DrugEx.

## 5. Python traceback

The traceback lives in `$OUT_DIR/execution.log` **and** in the PBS log. Two we have actually
hit:

**5a. `TypeError: SequenceExplorer.__init__() got an unexpected keyword argument 'config'`**

Job `25249720` (Nov 3). `SequenceRNNExplorer.build_explorer` passed a keyword its parent no
longer accepted in one branch, so the job died on the first model construction. Fixed in
commit `027c2ca`. The lesson is generic: a cluster traceback is a **code** bug, so the fix is
a new commit — never an edit of the release directory or of a copied script, both of which
make the run unreproducible. Then `submit` again with the same name (the run id gets `_2`).

**5b. Silent zero-work run**

The same evening's `25249783` reached `R` but produced no arms and no epoch rows. When that
happens:

```bash
ssh meta 'qstat -f <id> | grep -E "exec_host|resources_used"'
$PY -m _david.code.metacentrum.submit logs <name> -n 200
# if the job is still alive, look at the scratch run dir
ssh <node> 'ls -l /scratch*/job_<id>*/output/*/; tail -20 /scratch*/job_<id>*/output/*/execution.log'
```

If `execution.log` stops before Stage 1 and no epoch appears within an hour, the process is
wedged in setup or a pathological conformer embedding. Kill it and resubmit with
`--timeout 20` (per-molecule ETKDG cap). Do not "wait it out": a held core still burns
walltime, and there is no cheaper probe than reading the first epoch's telemetry in
`logs <name>`.

For any traceback, reproduce locally first when it is cheap — the same commit runs on WSL:

```bash
.venv/bin/python -m _david.code.pipeline --epochs 1 --samples 10 --batch-size 10 \
    --eval-samples 10 --output-dir /tmp/opencode/repro
```

## 6. Killed mid-run: walltime, memory, scratch

```bash
ssh meta 'qstat -x -f <id> | grep -E "Exit_status|resources_used.walltime|resources_used.mem"'
```

| `Exit_status` | Meaning |
| :--- | :--- |
| `-15` / `271` | `SIGTERM`: walltime expiry or `qdel`. Normal kill path; the trap copies results. |
| `-9` / `137` | `SIGKILL`: OOM killer or hard PBS kill. No cleanup ran; scratch may still hold results. |
| `-11` / `139` | `SIGSEGV`: native crash (RDKit/OpenMP). Resubmit with `--conformer-threads 1` and a smaller `--conformers`. |
| `0` but no results | Transfer failed before the trap's `rsync`; see §7. |

**Walltime.** The job is `SIGTERM`'d, the `EXIT` trap copies `$OUT_DIR` back, then scratch is
cleaned. The run directory exists on storage with `manifest.state = "running"` (the pipeline
never got to write `complete`). Epochs and last checkpoint are usable; figures may be absent.
Options: accept the partial run, or rerun with a walltime that fits the measured epoch rate
(chapter 02 gives the ETA; chapter 05 the arithmetic).

**OOM.** PBS kills the process group; the trap usually still runs, so results are copied but
the model is mid-epoch. `resources_used.mem` in `qstat -x` shows the peak. Raise `--mem`
(64 GB for 32 cores or large ligands) or reduce `--ncpus` so fewer workers hold conformers
simultaneously. `PipelineConfig.cpu_ratio = 0.99` means the pipeline will use every granted
core if asked; the worker pool is derived from `ncpus`.

**Scratch overflow.** The log contains `No space left on device` and RDKit may raise while
writing an SDF. The run's scratch holds the release (~23 MB), the run directory (SDFs for
5000 eval molecules plus figures), and per-molecule temporary conformers. 15 GB is enough for
EGCG-scale runs; for large ligands or several thousand minimised conformers go to 30 GB:

```bash
… submit --name big_lig --scratch 30 -- --conformers 40
```

## 7. Results never arrived

If the job ended green but `$STORAGE/_david/output/` has no new directory, the copy-back
failed (storage full, permission, or the node lost the mount). The trap prints:

```
Result transfer FAILED; scratch kept at /scratch…/job_<id>
```

Rescue before the scratch is purged (MetaCentrum reclaims it after the job ends):

```bash
ssh <node> 'du -sh /scratch*/job_<id>*'
rsync -a "<node>:/scratch…/job_<id>/output/<run-id>/" \
         _david/output/<run-id>/
```

Then verify with the comparison tool and re-run `audit_rl_runs`. If the scratch is already
gone, the run is lost; the PBS log and spec remain as provenance, and the job must be
resubmitted.

## 8. After any failure

- Keep the PBS log: `rsync -a "meta:\$HOME/<name>.o*" _david/output/_metacentrum_jobs/pbs_logs/`.
- Keep the spec: `rsync -a "meta:$STORAGE/jobs/<name>/" _david/output/_metacentrum_jobs/jobs/`.
- If the cause was code, fix it locally, add a regression test, commit, and resubmit; do not
  patch the release on storage.
- If the cause was a setting, write down what changed in the next job name (e.g.
  `egcg_c30_unmin_r2`) so the audit table remains readable.
