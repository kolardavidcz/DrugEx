# 02 — Monitoring a run

Monitoring has three layers, in increasing detail:

| Layer | Command | Answers |
| :--- | :--- | :--- |
| Queue | `submit status` or `qstat -u kolardavidcz` | Is it queued, running, held? Since when? |
| Log | `submit logs <name> -f` | Did it start? Which traceback killed it? |
| Epochs | `monitor_live_epochs.py` | Is it learning? How fast? When will it finish? |

The first two are cheap and always work from the login node. The third reads the worker
node's scratch directory and therefore needs the job to be running.

## 1. Queue states

`submit status` parses `qstat -f` (long format). It prints job id, state, queue, elapsed,
walltime limit and name, plus a running/queued count.

| `job_state` | Meaning | Action |
| :--- | :--- | :--- |
| `Q` | Queued. | Nothing; check queue depth in chapter 05. |
| `R` | Running. | Monitor epochs, or wait. |
| `H` | Held by PBS or by the scheduler. | `qstat -f <id> \| grep comment` — usually a resource request the scheduler cannot satisfy or an admin hold. |
| `W` | Waiting (subjob/dependency). | Only relevant if you used dependencies; we do not. |
| `S` | Suspended (preemption). | Usually resumes; if not, it burns walltime — consider `qdel` and resubmit. |
| `E` | Exiting. | Temporary; re-check in a minute. |
| `C` | Completed (only with `qstat -x`). | Check `Exit_status` and pull results. |

Do **not** use `qstat -F json`: this PBS build rejects the format (`qstat: conflicting
options.`). The long format `qstat -f` is the parseable one and is what `submit.py` uses.

Useful raw queries:

```bash
ssh meta 'qstat -u kolardavidcz'                      # compact table
ssh meta 'qstat -f 25244717'                          # every field of one job
ssh meta 'qstat -x -f 25244717 | grep -E "Exit_status|resources_used"'   # after it ends
ssh meta 'qstat -q | head -20'                         # queue depths
ssh meta 'qstat -u kolardavidcz -n'                    # running jobs with node list
```

Fields worth reading in `qstat -f`:

- `exec_host` — worker node; strip everything after `/` (that is the core index).
- `Resource_List.walltime` vs `resources_used.walltime` — budget vs burn.
- `Resource_List.select` — what was actually granted (`ncpus`, `mem`, `ngpus`).
- `comment` — hold/queue reason, e.g. "Not Running: Insufficient amount of resource".
- `Submit_arguments` — the exact `-v RUN_SPEC=…` path, hence the exact spec that ran.
- `estimated.start_time` — present on some queues; when absent, the queue depth is the only
  estimate.

## 2. PBS log

Each job writes `$HOME/<job-name>.o<sequence>` (stderr is merged, `-j oe`). `$HOME` is
`/storage/praha1/home/kolardavidcz`, *not* the storage root.

```bash
$PY -m _david.code.metacentrum.submit logs <name>          # last 40 lines
$PY -m _david.code.metacentrum.submit logs <name> -f       # follow (Ctrl-C to stop)
$PY -m _david.code.metacentrum.submit logs <name> -n 200
ssh meta 'ls -t $HOME/<name>.o* | head -3'                 # several attempts of one name
```

The first 15 lines are the template header — job id, node, queue, cpus, scratch path,
release, revision, run id and the final results directory. After that, the pipeline's own
log. Two harmless lines that look alarming:

- `RuntimeWarning: '_david.code.pipeline' found in sys.modules after import of package
  '_david.code'` — an artefact of `python -m` plus the package `__init__`; ignore it.
- Matplotlib backend messages — the template forces `Agg`, so figures are written, not shown.

The log does **not** contain per-epoch metrics (those go to the run directory). It contains
startup guards, tracebacks, and the final `Results copied to …` line from the EXIT trap.

## 3. Epoch telemetry

`monitor_live_epochs.py` (next to this chapter) is stdlib-only and runs on the login node:

```bash
ssh meta 'python3 /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/help/monitor_live_epochs.py'
```

Every `sync` repoints `current`, so the helper you run is always the one from the staged
release. Useful options:

```bash
… monitor_live_epochs.py --lines 8            # more epochs per job
… monitor_live_epochs.py --watch 120         # refresh every 2 min, Ctrl-C to stop
… monitor_live_epochs.py --user kolardavidcz --json   # machine-readable summary
```

For each running job it prints: job id, name, node, elapsed/limit walltime, the last
`--lines` rows of `models/SequenceRNN_reinforced_fit.tsv`, the total epoch count, the age of
the newest epoch row, an epoch-rate estimate and an ETA against the walltime limit. If the
scratch directory cannot be found (job just started, or a node filesystem we cannot see), it
falls back to the tail of the PBS log.

### Reading `SequenceRNN_reinforced_fit.tsv`

```
Epoch  valid_ratio  unique_ratio  desired_ratio  avg_amean  avg_gmean  loss_train  best_epoch
1      0.88         0.88          0.78           0.746      0.717      25.58       1
```

- `valid_ratio` — fraction of generated strings parse as SMILES. Below ~0.7 early is a bad
  sign; a slow decline over epochs means the policy is drifting.
- `unique_ratio` — distinct molecules / generated. Collapse (< 0.3) means mode collapse;
  the usual remedy is not "more epochs" but a different epsilon schedule.
- `desired_ratio` — fraction above all desirability thresholds.
- `avg_amean`, `avg_gmean` — per-molecule objective means. `avg_gmean` is the gold-standard
  checkpoint criterion (`--criteria gmean`); a run whose `avg_amean` rises while `avg_gmean`
  falls is buying one objective with another, which is exactly what gmean penalises.
- `best_epoch` — the checkpoint actually kept (`SequenceRNN_reinforced.pkg`). If it has not
  moved for `patience` epochs, the run is about to early-stop — that is success, not failure.

### Health rules of thumb

| Observation | Interpretation |
| :--- | :--- |
| New row every few minutes to ~15 min | Normal for `--no-minimize-candidates`; minimised runs are ~4× slower. |
| No new row for 30+ min, log still alive | One pathological molecule is stuck in conformer embedding. `--timeout 30` bounds it; check whether the node is CPU-saturated. |
| `valid_ratio` → 0 near epoch 1 | Wrong vocabulary/checkpoint pairing, or a reference SMILES the tokenizer cannot represent. Read the PBS log. |
| `loss_train` = `nan` | Numerical blow-up; kill and lower `--epsilon`/`--beta` rather than resuming. |
| `best_epoch` frozen and epoch > `min_epochs + patience` | Early stopping is about to fire; the run is effectively done. |
| ETA > walltime remaining | The job will be killed mid-run. The EXIT trap keeps partial results, but plan a longer `--walltime` or fewer epochs. |

## 4. Inside the worker node

While the job runs you can look at the node directly (MetaCentrum allows SSH to nodes where
you have a running job):

```bash
ssh meta 'qstat -f <id> | grep exec_host'          # e.g. elmo3-6.hw.elixir-czech.cz/0-15
ssh <node> 'ls -d /scratch*/job_<id>* /scratch*/*/job_<id>* 2>/dev/null'
ssh <node> 'tail -f /scratch*/job_<id>*/output/*/execution.log'
ssh <node> 'du -sh /scratch*/job_<id>*; df -h /scratch*'
```

The exact scratch root differs per node generation (`/scratch`, `/scratch.ssd`,
`/scratch.dir`), which is why the monitor script globs instead of hard-coding one. Inside it,
the run directory mirrors the local layout:

```
$SCRATCHDIR/output/<run-id>/
├── submission.json     # the spec that produced this run
├── execution.log       # live pipeline output (tee'd)
├── models/             # checkpoints + SequenceRNN_reinforced_fit.tsv
├── raw_data/           # baseline/reinforced arms
└── figures/            # written at the end
```

`du -sh` is also the fastest way to confirm scratch pressure: the 15 GB `scratch_local`
request must hold the release, the run directory (SDFs + figures) and any temporary
conformer files. A run with 5000 eval molecules and figures can approach it; chapter 05 has
the sizing rule.

## 5. Stopping a run

```bash
$PY -m _david.code.metacentrum.submit cancel <name>        # all jobs with that name
$PY -m _david.code.metacentrum.submit cancel 25244717      # one job id
ssh meta 'qselect -u kolardavidcz -s Q | xargs -r qdel'    # everything queued, nothing running
```

`qdel` on a running job sends `SIGTERM`; the template's trap copies partial results back
before cleaning scratch. Give it a minute, then `qstat -x -f <id>` to confirm the copy line
in the log. Never delete a scratch directory by hand while the trap may still be copying.
