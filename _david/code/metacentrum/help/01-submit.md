# 01 — Submitting a run

The whole submission path is three small pieces: a **release**, a **spec**, a **PBS file**.
Understanding them makes every error message obvious.

```
local git HEAD ──git archive──▶ $STORAGE/releases/<sha>/ ──rsync──▶ $SCRATCHDIR/DrugEx/
                                      ▲                                   │
                                      │                                   ▼
$STORAGE/jobs/<name>/spec.json ───────┘            python -m _david.code.pipeline
```

## 1. The release

```bash
$PY -m _david.code.metacentrum.submit sync
```

What happens, in order:

1. `git rev-parse HEAD` → `<sha>`.
2. `git status --porcelain --untracked-files=no` must be empty. A modified tracked file is a
   hard error: the job would run HEAD while you believe it runs your edit. Untracked files
   are ignored because `git archive` cannot contain them anyway.
3. `git archive <sha>` is streamed over SSH and unpacked into
   `$STORAGE/releases/<sha>/`. Existing content of that directory is only ever additive —
   the same commit staged twice yields the same tree.
4. `$STORAGE/current` is atomically repointed (`ln -sfn`) at the new release.
5. `Papyrus05.5_smiles_rnn_PT.pkg` and `.vocab` are rsynced into `$STORAGE/models/` from the
   paths resolved by `PipelineConfig.resolve_pretrained_paths()`.

Consequences:

- A submitted job keeps its code even if you `git push`, rebase, or edit for the next hour.
- `releases/<sha>/` is the artefact to inspect when you ask "what exactly ran?".
- Nothing is ever rsynced into `current/`; it is always a symlink to an immutable release.
- `sync` is idempotent and cheap after the first time for a given commit (only the tarball
  transfer repeats).

Storage daemon details and cleanup rules: [`06-storage.md`](06-storage.md).

## 2. The spec

`$STORAGE/jobs/<name>/spec.json` is written by `submit.py` and read by the PBS job:

```json
{
  "name": "egcg_gold_gmean",
  "args": [
    "--criteria", "gmean",
    "--agent-path", "/storage/brno2/brno2/kolardavidcz/DrugEx/models/Papyrus05.5_smiles_rnn_PT.pkg",
    "--vocab-path", "/storage/brno2/brno2/kolardavidcz/DrugEx/models/Papyrus05.5_smiles_rnn_PT.vocab"
  ],
  "source": "/storage/brno2/brno2/kolardavidcz/DrugEx/releases/<sha>",
  "revision": "<sha>"
}
```

- `args` is a JSON array, never a shell string. Quoting survives; a SMILES containing `[`,
  `(`, `=` or `%` cannot be re-split by the shell.
- The template appends `--output-dir`, so the run lands in scratch and is copied back to
  `$STORAGE/_david/output/<run-id>`.
- `--agent-path` / `--vocab-path` are appended by `submit.py` because the checkpoint lives on
  storage, not inside the release. Passing them yourself is rejected by `JobSpec`
  (`RESERVED_FLAGS`), as is `--output-dir`.
- The spec is kept after the job finishes. Together with `manifest.json` in the run
  directory it answers "which flags, which commit, which job id?" without guessing.
- A copy of the spec is written into the run directory as `submission.json` before the
  pipeline starts.

## 3. The PBS file

`job_template.sh` carries only fallback `#PBS` defaults; every real submission passes
resources on the `qsub` command line (`-N`, `-q`, `-l select=…`, `-l walltime=…`,
`-v RUN_SPEC=…,STORAGE_DIR=…`). The template:

1. Fails fast if `PBS_JOBID`, `SCRATCHDIR`, `STORAGE_DIR` or `RUN_SPEC` is missing, or if
   `$STORAGE/.venv/bin/python` is not executable.
2. Claims `RUN_ID=$(date +%Y%m%d_%H%M)_<name>`, appending `_2`, `_3` … if the directory
   already exists. The same rule as local runs (`config.free_run_directory`).
3. Prints a header with job id, name, node, queue, cpus, scratch, release, revision and the
   final run id — so the first 15 lines of the PBS log are a complete provenance record.
4. Installs an `EXIT` trap **before** copying the release: success, failure, `SIGTERM`
   (walltime) and `SIGINT` all copy scratch results back and then clean scratch.
5. Copies the release into `$SCRATCHDIR/DrugEx`, exports
   `MPLBACKEND=Agg`, thread pinning (`OPENBLAS/OMP/MKL=1`) and
   `DRUGEX_MP_CONTEXT=forkserver`, and runs the pipeline with `-u` so the execution log is
   live.
6. Pipes stdout+stderr through `tee` into `<run>/execution.log`.

If the copy back fails, the trap prints the scratch path and exits non-zero instead of
deleting the only copy of the results. That message is the one case where you must log into
the worker node before the job is purged.

## 4. Flags that reach the pipeline

Anything after `--` (or inside `--args "…"`) is handed to
`_david.code.pipeline` verbatim. All defaults come from `PipelineConfig`; see
`python -m _david.code.pipeline --help`. The ones used in practice:

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--epochs` | 200 | Maximum RL epochs. |
| `--min-epochs` | 50 | Epochs before early stopping can fire. |
| `--patience` | 50 | Epochs without checkpoint improvement before stopping. |
| `--criteria` | `avg_gmean` | Checkpoint criterion (`gmean` is the CLI alias). |
| `--include-qed` / `--no-include-qed` | off | Adds QED as a third objective. |
| `--reward-scheme` | `pareto_crowding` | `pareto_crowding`, `harrington`, `weighted_sum`. |
| `--minimize-candidates` / `--no-…` | on | MMFF94s minimisation of candidates (~4× slower). |
| `--conformers` | 20 | Candidate conformers per isomer. |
| `--reference-conformers` | 4 | Reference ensemble cap K. |
| `--molecule` | EGCG SMILES | Reference SMILES or SDF path. |
| `--eval-samples` | 5000 | Molecules per evaluation arm. |
| `--run-tag` | none | Only for local runs; on MetaCentrum the job name is the tag. |

Never pass `--output-dir`, `--agent-path`, `--vocab-path`; the template and staging own them.

## 5. Dry run and the qsub line

```bash
$PY -m _david.code.metacentrum.submit submit --name test --dry-run -- "--epochs 1"
```

```
  release : /storage/brno2/brno2/kolardavidcz/DrugEx/releases/<sha>
  queue   : default router → q_1d
  resources: select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0, walltime=24:00:00 (routed by walltime; shorter walltime = shorter queue)
  pipeline: --epochs 1

$ ssh meta qsub -N test -l select=... -l walltime=24:00:00 -v RUN_SPEC=...,STORAGE_DIR=... /storage/.../job_template.sh
```

There is **no `-q`** in the default command. That is deliberate: from the frontends we are
allowed to use, execution queues reject direct submissions (`p2e_1d` via `acl_hosts`,
`q_2h`…`q_2d` via `from_route_only`), and the `default` router selects the queue from
`-l walltime`. The printed `queue :` line is the router's destination, so a 1 h smoke test
shows `q_2h` and a 24 h production run shows `q_1d`. Details and the full table:
[`05-resources.md`](05-resources.md) §2.

`--dry-run` still runs `sync` unless `--no-sync` is given, because the printed release path
must exist. If you only want to see resources without touching storage, add `--no-sync`.

## 6. Command mapping from the grid era

`grid.py` was deleted with the grid/sprint machinery. Nothing in it is needed for ad-hoc
runs; the only genuinely useful part (release staging + qsub) is now `submit.py`.

| Old | Now |
| :--- | :--- |
| `grid sync` | `submit sync` |
| `grid submit --name X --args "…"` | `submit submit --name X -- "…"` |
| `grid status` | `submit status` |
| `grid pull` | `submit pull` |
| `grid sprint` / `grid submit-sprint` | gone — submit each arm explicitly |
| `grid grid` / `grid submit-grid` | gone |
| `scripts/submit_all_hypotheses.sh` | `scripts/runed/` (frozen, do not run) |

`grid.py` also defaulted to `ngpus=1`, `spec=10.0` and `-q p2e_1d`, which is wrong for CPU
ROCS work and no longer accepted from this frontend; the new default is `ngpus=0`,
`spec=8.0`, no `-q` (PBS routes by walltime).

## 7. Common submission errors

| Message | Cause and fix |
| :--- | :--- |
| `Tracked files are modified; commit them…` | Commit or stash. The job runs HEAD. |
| `Missing validated environment: …/.venv/bin/python` | `.venv` was deleted on storage; recreate it (chapter 06) before submitting. |
| `Kerberos ticket missing or expired` | `ssh meta kinit kolardavidcz@META` interactively (2FA), then retry. |
| `qsub: Unknown queue` | `qstat -q` to list queues; typically a typo or a queue retired since the last run. |
| `qsub: Access from host not allowed, or unknown host` | You passed `--queue` for an execution queue (`p2e_1d` and friends). Drop `--queue` and let the router pick, or shorten `--walltime`. |
| `qsub: Access to queue is denied` | The queue is `from_route_only` or not granted to this account. Same fix: omit `--queue`. |
| `--output-dir is owned by the MetaCentrum template…` | Remove the flag from the pipeline arguments. |
| `Pass pipeline flags either with --args or after --, not both` | Pick one spelling. |
| `.pkg`/`.vocab` not found in `$STORAGE/models` | `submit sync` again; the checkpoint is outside the release by design. |

Deeper failure analysis: [`04-failures.md`](04-failures.md).
