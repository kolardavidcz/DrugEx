# MetaCentrum runbook — the one template

MetaCentrum is an **ad-hoc** host for this thesis: occasional single jobs, used when the
local workstation or `dichtator` cannot deliver the cores. It is not a production queue
that mirrors the local tree.

If an agent is asked to "run X on meta", this file is the whole contract. Do not invent a
new shell script, do not edit `pbs_launcher.sh` or `job.py` for one experiment, do not call
`qsub` directly. `submit.py` stages a release, writes the job spec and submits it; the PBS
side is a six-line launcher plus `job.py`, and neither changes per run.

---

## 0. Three rules

1. **Defaults live in code.** `_david/code/config.py::PipelineConfig` is the single source of
   truth. A submission carries **only the deltas** the user asked for. "Defaults, but
   geometric-mean checkpointing" means `--args "--criteria gmean"` and nothing else.
   If a knob is not a field of `PipelineConfig` / a `pipeline.py` flag, it is a PBS
   resource (`--ncpus`, `--mem`, `--scratch`, `--walltime`, `--queue`) — nothing else exists.
2. **One entry point.** `python -m _david.code.metacentrum.submit` for every action.
   `scripts/runed/` is frozen provenance from the grid era and is never re-run or copied.
3. **Commit before submitting.** `git archive HEAD` is what the job executes, so `sync`
   refuses a tree with uncommitted tracked changes. Untracked files never travel; add
   inputs that the job needs to git, or place them on storage yourself.

---

## 1. The five commands

Run from the repository root, with the project virtualenv:

```bash
PY=.venv/bin/python
$PY -m _david.code.metacentrum.submit check                      # SSH, env, release, queue
$PY -m _david.code.metacentrum.submit submit --name <name> [flags] [-- <pipeline flags>]
$PY -m _david.code.metacentrum.submit status                     # queue table for the user
$PY -m _david.code.metacentrum.submit logs <name> -f             # PBS stdout, live
$PY -m _david.code.metacentrum.submit pull                       # finished runs → _david/output
```

Also available: `sync` (stage HEAD without submitting), `cancel <id|name>...`.

`submit` syncs automatically. Use `--no-sync` only when deliberately re-running an already
staged revision, and `--revision <sha>` to run an older one.

---

## 2. Submitting a run

### 2a. There is no preview mode

`submit` always hands the job to PBS. There is no `--dry-run`: a submission that prints
resources but does not queue anything is a rehearsal, and rehearsals hide the errors that
matter (ACL refusals, Kerberos expiry, a bad resource expression). The submission prints the
release, the routed queue, the resource expression and the pipeline deltas **before** `qsub`
runs, so the block on screen is the record of what was actually requested:

```
  release : /storage/brno2/brno2/kolardavidcz/DrugEx/releases/<sha>
  queue   : q_1d (default router)
  resources: select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0, walltime=24:00:00
  pipeline: --criteria gmean

Submitted egcg_gold_gmean: <job id>
```

`sync` runs as part of that block, so HEAD is staged before the queue sees the job; commit
first (an uncommitted tracked change is refused). If the `pipeline:` line shows anything
beyond the intended deltas, the job is already submitted — cancel it, fix the invocation,
resubmit. Getting the invocation right is a matter of reading `PipelineConfig`, not of
printing a rehearsal command.

### 2b. The default job

A CPU job: 16 cores, 32 GB RAM, 15 GB scratch, 24 h, no GPU, **no `-q`** — PBS routes it by
walltime (24 h → `q_1d`). This is the scientifically relevant default (ROCS + conformer
generation is CPU-bound; the RNN is tiny).

```bash
$PY -m _david.code.metacentrum.submit submit --name egcg_default
```

Reads as: "run the pipeline with `PipelineConfig` defaults, 200 epochs max, early stop
after 30 epochs without improvement (never before epoch 50), raw ROCS TanimotoCombo + SA,
Harrington reward, `avg_gmean` checkpoint, 5000-molecule evaluation arm". Nothing is passed because
`config.py` already says that.

**Queue selection.** MetaCentrum frontends cannot submit to an execution queue directly:
`p2e_1d` has `acl_hosts = pbs-m1.metacentrum.cz` (error: `Access from host not allowed`) and
`q_2h`/`q_4h`/`q_1d`/`q_2d` are `from_route_only` (`Access to queue is denied`). Submitting
without `-q` goes through the `default` router, which picks the queue from the walltime:

| `--walltime` up to | Queue |
| :--- | :--- |
| 2 h | `q_2h` |
| 4 h | `q_4h` |
| 24 h | `q_1d` |
| 48 h | `q_2d` |
| 1 week | `q_1w` |

**The default is the only lane we use: 24 h → `q_1d`.** Do not shorten the walltime to land in
a shallower queue — a job that needs 20 hours but asks for 2 is killed at hour two, and the
queue wait is not a thing to optimise by lying about the budget. Ask for the compute the run
needs. `--walltime 48:00:00` (→ `q_2d`) is the only documented deviation, for minimised
candidate sets that genuinely do not fit in a day. `--queue` is reserved for a route queue or
a special queue an administrator grants; it is never needed for normal work.

### 2c. "Defaults plus one speciality"

```bash
# Geometric-mean checkpointing only
$PY -m _david.code.metacentrum.submit submit \
    --name egcg_gmean -- --criteria gmean

# No QED objective, 50 epoch cap
$PY -m _david.code.metacentrum.submit submit \
    --name egcg_noqed_50 -- --epochs 50 --no-include-qed

# Minimised candidates (slower, ~4x; only when asked)
$PY -m _david.code.metacentrum.submit submit \
    --name egcg_min -- --minimize-candidates --conformers 20
```

Both spellings of the pipeline arguments are equivalent (`--args "…"` or after `--`);
never use both in one command. **After `--` every flag is its own argument** — write
`-- --epochs 50 --no-include-qed`, not `-- "--epochs 50 --no-include-qed"`. The quoted
form is refused with an explicit error, because the whole string would otherwise reach
argparse as a single nonsensical argument (this is exactly how job `25256773` failed on
2026-11-04).
`--output-dir`, `--agent-path` and `--vocab-path` are reserved: the release contract and
the staged checkpoint own them, and `JobSpec` rejects them loudly.

### 2d. GPU

GPU queues are currently 2000+ jobs deep. Only reach for one when the user explicitly asks
for GPU compute, and prefer `gpu_long` (a route destination, so `--queue gpu_long` is
accepted; the generic `gpu` queue is the same but far deeper):

```bash
$PY -m _david.code.metacentrum.submit submit \
    --name egcg_gpu --queue gpu_long --ngpus 1 --gpu-cap "sm_89" --walltime 24:00:00 \
    -- --epochs 100
```

`--spec` defaults to `8.0` for both CPU and GPU: it excludes ancient nodes without
shrinking the pool to `fobos`/`bee` the way `spec=10` does.

### 2e. Job naming

`--name` is both the PBS job name and the run-directory tag. It must match
`[A-Za-z][A-Za-z0-9_-]{0,63}` and should say what the run is:
`egcg_gold_gmean`, `dnmt1_c30_unmin`. The run directory becomes
`$STORAGE/_david/output/YYYYmmdd_HHMM_<name>` (collision → `_2`, `_3`), so two jobs with
the same name are still distinguishable but `cancel <name>` would hit both.

---

## 3. While it runs

```bash
$PY -m _david.code.metacentrum.submit status
$PY -m _david.code.metacentrum.submit logs egcg_gold_gmean -f
ssh meta 'python3 /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/help/monitor_live_epochs.py'
```

`status` parses `qstat -f`, so long job names do not shift columns. `logs` tails
`$HOME/<name>.o<seq>` on the login node — the PBS stdout, which includes startup guards and
Python tracebacks. Epoch telemetry is in the scratch run directory and is what
`monitor_live_epochs.py` shows.

Details, thresholds and failure triage: [`help/02-monitor.md`](help/02-monitor.md) and
[`help/04-failures.md`](help/04-failures.md).

---

## 4. Finishing

```bash
$PY -m _david.code.metacentrum.submit pull                 # all runs
$PY -m _david.code.metacentrum.submit pull --name 20261104_1530_egcg_gold_gmean
$PY -m _david.code.audit_rl_runs                           # gold-standard + figure audit
```

`pull` rsyncs `$STORAGE/_david/output/` into `_david/output/`, preserving `manifest.json`
and timestamps. Verify before deleting anything remote:

```bash
python3 _david/code/metacentrum/archive/file_folder_tree_comparasion/compare_output_trees.py \
    --compare --local-root _david/output \
    --remote "ssh meta" \
    --remote-root "/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output"
```

A clean run reports `local-only 0 | remote-only 0 | moved 0`. Deletion policy, checksums and
storage layout: [`help/03-collect.md`](help/03-collect.md) and
[`help/06-storage.md`](help/06-storage.md).

---

## 5. What the job actually does

1. `submit.py` runs `git archive HEAD` locally and unpacks it to
   `$STORAGE/releases/<sha>/`, repoints `$STORAGE/current`, and rsyncs
   `Papyrus05.5_smiles_rnn_PT.{pkg,vocab}` to `$STORAGE/models/`.
2. It writes `$STORAGE/jobs/<name>/spec.json` with the exact argument vector, release path
   and revision.
3. `qsub` starts `pbs_launcher.sh` with `RUN_SPEC`, `STORAGE_DIR` and `RELEASE_DIR` in the
   environment. PBS spools the script outside the release, so the release path must be passed
   explicitly — never derived from the launcher's own location. The launcher checks the three
   variables and the stored venv, then `exec`s
   `$STORAGE/.venv/bin/python $RELEASE_DIR/_david/code/metacentrum/job.py --release $RELEASE_DIR`.
4. `job.py` validates `spec.json`, claims a new run id, copies the release into
   `$SCRATCHDIR/DrugEx`, and runs
   `python -u -m _david.code.pipeline --output-dir $SCRATCHDIR/output/<run-id> <args>`,
   mirroring stdout+stderr into `<run>/execution.log`.
5. A `finally` block copies the run directory back to `$STORAGE/_david/output/<run-id>` on
   success, failure, `SIGTERM` (walltime) and `SIGINT`, then cleans scratch. A failed
   transfer keeps scratch alive and exits non-zero.

Consequences worth remembering: a job is immune to later local edits (immutable release);
the checkpoint and vocabulary always come from `$STORAGE/models`, never from the release;
and partial results survive a walltime kill.

---

## 6. Resource quick reference

| Situation | Flags |
| :--- | :--- |
| Default CPU ROCS run | *(nothing)* — 16 CPU / 32 GB / 15 GB / 24 h → `q_1d` |
| Many conformers / large ligand | `--mem 64 --ncpus 32` (walltime stays 24 h → `q_1d`) |
| Minimised 200-epoch run that cannot fit in a day | `--walltime 48:00:00` → `q_2d` |
| GPU, only when the user asks | `--queue gpu_long --ngpus 1 --gpu-cap sm_89` |
| Debug node | `--queue runone` if the account is allowed there (usually denied) |

Queue lengths, memory sizing and walltime arithmetic: [`help/05-resources.md`](help/05-resources.md).

---

## 7. Before you say "submitted"

- [ ] `git status --porcelain --untracked-files=no` is empty (or `sync` already passed).
- [ ] The submission block printed `pipeline:` with the intended deltas and nothing else.
- [ ] Job name is unique and descriptive; it becomes the run tag.
- [ ] `status` shows the job (`Q` is fine; `H` means held — see `help/04-failures.md`).
- [ ] You have told the user the `run id`, the queue (`q_1d` by default) and the expected
      wall-clock cost.

---

## 8. Standard re-evaluation of finished runs (task 2.4)

The RL runs were evaluated at different sizes and with different scorer windows, so they
cannot be compared directly. `standard_eval` adds one comparable arm beside the historical
one, never overwriting `raw_data/reinforced_raw.csv.gz`:

```bash
PY=.venv/bin/python
$PY -m _david.code.standard_eval plan                        # what each arm allows
$PY -m _david.code.prior_dataset sync --name <dataset>       # dataset → <storage>/datasets/
$PY -m _david.code.standard_eval push --all                  # manifests + checkpoints → storage
$PY -m _david.code.standard_eval submit --dataset <dataset> --max-runs 20 \
    [--build-dataset --dataset-seed 20261105]   # optional: freeze it on storage first
$PY -m _david.code.standard_eval pull                        # only <run>/standard_eval/ back
$PY -m _david.code.standard_eval verify
```

Classification is by artifact, not by history:

| On disk | Action |
| :--- | :--- |
| `models/SequenceRNN_reinforced*.pkg` | reload, sample 5 000 with a fixed seed (`resampled: true`) |
| only `raw_data/reinforced_raw.csv.gz` | re-score the stored molecules (`resampled: false`, keeps n) |
| neither | `skip` with the reason |

The job is idempotent: a run whose `standard_eval/reinforced_standard.csv.gz` already exists
is `done`, so a walltime-killed job is resumed by submitting it again. The dataset must be on
storage first; the job refuses a missing dataset rather than falling back to a local arm.
With `--build-dataset` the same job first builds the dataset on storage (skipped when its
`manifest.json` already exists), so a fresh dataset and the re-evaluation need only one
submission. Pull the dataset back with
`python -m _david.code.prior_dataset pull --name <dataset>`.
Every standard manifest records the dataset hash, the seed, the checkpoint hash and the
fixed reward definition (`harrington`, weights 0.50/0.25/0.25, raw TanimotoCombo).
