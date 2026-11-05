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
- `job.py` appends `--output-dir`, so the run lands in scratch and is copied back to
  `$STORAGE/_david/output/<run-id>`.
- `--agent-path` / `--vocab-path` are appended by `submit.py` because the checkpoint lives on
  storage, not inside the release. Passing them yourself is rejected by `JobSpec`
  (`RESERVED_FLAGS`), as is `--output-dir`.
- The spec is kept after the job finishes. Together with `manifest.json` in the run
  directory it answers "which flags, which commit, which job id?" without guessing.
- A copy of the spec is written into the run directory as `submission.json` before the
  pipeline starts.

## 3. The launcher and the job body

PBS needs a script, so the entry point is [`pbs_launcher.sh`](../pbs_launcher.sh): a few
lines that require `STORAGE_DIR`, `RUN_SPEC` and `RELEASE_DIR`, check
`$STORAGE/.venv/bin/python`, and `exec` [`job.py`](../job.py) from that release with the
stored interpreter. `RELEASE_DIR` is passed by `submit.py` because **PBS spools the script
outside the release**: `dirname(BASH_SOURCE)` resolves to `/var/spool/pbs` on the worker, not
to `releases/<sha>/`. Everything else is Python, because the interesting part is not
shell-shaped:

1. `job.py` fails fast if `PBS_JOBID`, `SCRATCHDIR`, `STORAGE_DIR` or `RUN_SPEC` is missing,
   if the venv is gone, if the spec is malformed, or if the launcher's release and
   `spec["source"]` disagree.
2. It claims `YYYYmmdd_HHMM_<name>`, appending `_2`, `_3` … if the directory already exists —
   the same rule as local runs (`config.free_run_directory`).
3. It prints the header (job id, name, node, queue, cpus from `PBS_NODEFILE`, scratch,
   release, revision, run id, results path), so the first lines of the PBS log are a complete
   provenance record.
4. It copies the release into `$SCRATCHDIR/DrugEx`, exports `MPLBACKEND=Agg`, thread pinning
   (`OPENBLAS/OMP/MKL=1`) and `DRUGEX_MP_CONTEXT=forkserver`, then runs the pipeline with
   `-u`, copying every line to stdout and to `<run>/execution.log` — no `tee` process.
5. A `finally` block copies the run directory back on success, failure, `SIGTERM` (walltime)
   and `SIGINT`, then cleans scratch. A failed transfer prints the scratch path, keeps it and
   exits non-zero instead of deleting the only copy of the results.

The launcher is a shim on purpose: bash `trap` semantics cannot be unit-tested, while
`job.py` (run id, environment pinning, copy-back, exit codes) is covered by
`code/tests/test_metacentrum.py`. `#PBS` lines in the launcher are fallback defaults only;
resources always arrive on the `qsub` command line.

## 4. Flags that reach the pipeline

Anything after `--` (or inside `--args "…"`) is handed to
`_david.code.pipeline` verbatim. All defaults come from `PipelineConfig`; see
`python -m _david.code.pipeline --help`. The ones used in practice:

| Flag | Default | Effect |
| :--- | :--- | :--- |
| `--epochs` | 200 | Maximum RL epochs. |
| `--min-epochs` | 50 | Epochs before early stopping can fire. |
| `--patience` | 30 | Epochs without checkpoint improvement before stopping (floor: `min_epochs`). |
| `--criteria` | `avg_gmean` | Checkpoint criterion (`gmean` is the CLI alias). |
| `--include-qed` / `--no-include-qed` | off | Adds QED as a third objective. |
| `--reward-scheme` | `harrington` | `pareto_crowding`, `harrington`, `weighted_sum`. |
| `--color-ff` | `oe_mills_dean` | ROCS pharmacophore typing: `oe_mills_dean`, `default`, `fdef`. |
| `--rocs-clip-upper` | `2.0` | ClippedScore bound when `--rocs-use-modifier` is on; combo ceiling is 2.0. |
| `--minimize-candidates` / `--no-…` | on | MMFF94s minimisation of candidates (~4× slower). |
| `--conformers` | 20 | Candidate conformers per isomer. |
| `--reference-conformers` | 4 | Reference ensemble cap K. |
| `--molecule` | EGCG SMILES | Reference SMILES or SDF path. |
| `--eval-samples` | 5000 | Molecules per evaluation arm. |
| `--run-tag` | none | Only for local runs; on MetaCentrum the job name is the tag. |

Never pass `--output-dir`, `--agent-path`, `--vocab-path`; `job.py` and staging own them.

**Each flag is one argument after `--`.** `-- --epochs 50 --no-include-qed` works;
`-- "--epochs 50 --no-include-qed"` is rejected on purpose. The quoted form reaches argparse
as a single string, which the pipeline cannot parse — that failure
(`unrecognized arguments: --molecule CC(O)C --epochs 1 …`) killed job `25256773` on
2026-11-04. `--args "…"` is the spelling that *does* take one quoted string, and it is
tokenised with `shlex.split`.

## 5. What a submission prints

There is no preview or dry-run mode. `submit` always queues the job; before `qsub` it prints
the block that describes the request:

```
  release : /storage/brno2/brno2/kolardavidcz/DrugEx/releases/<sha>
  queue   : q_1d (default router)
  resources: select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0, walltime=24:00:00
  pipeline: --criteria gmean

Submitted egcg_gold_gmean: <job id>
Watch it: python -m _david.code.metacentrum.submit logs egcg_gold_gmean -f
```

There is **no `-q`** in the default command. That is deliberate: from the frontends we are
allowed to use, execution queues reject direct submissions (`p2e_1d` via `acl_hosts`,
`q_2h`…`q_2d` via `from_route_only`), and the `default` router selects the queue from
`-l walltime`. The default request is 24 h, which the router places in `q_1d`; that is the
lane every job uses. Do not shorten the walltime to reach a shallower queue — a truncated
budget kills the job before it finishes. Details and the full table:
[`05-resources.md`](05-resources.md) §2.

Because `sync` runs before the queue sees the job, the printed `release:` path exists by the
time the job starts. If the `pipeline:` line is wrong, cancel and resubmit: the job is
already queued, which is exactly why the block is printed before `qsub` rather than in a
rehearsal mode.

## 6. Command mapping from the grid era

`grid.py` was deleted with the grid/sprint machinery. Nothing in it is needed for ad-hoc
runs; the only genuinely useful part (release staging + qsub) is now `submit.py`.

| Old | Now |
| :--- | :--- |
| `grid sync` | `submit sync` |
| `grid submit --name X --args "…"` | `submit submit --name X --args "…"` |
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
| `qsub: Access from host not allowed, or unknown host` | You passed `--queue` for an execution queue (`p2e_1d` and friends). Drop `--queue`; the router picks `q_1d` from the default 24 h walltime. |
| `qsub: Access to queue is denied` | The queue is `from_route_only` or not granted to this account. Same fix: omit `--queue`. |
| `--output-dir is owned by the MetaCentrum job body…` | Remove the flag from the pipeline arguments. |
| `Pass pipeline flags either with --args or after --, not both` | Pick one spelling. |
| `Pipeline flags after '--' must be separate arguments…` | You quoted the whole tail. Drop the outer quotes, or use `--args "…"`. |
| `.pkg`/`.vocab` not found in `$STORAGE/models` | `submit sync` again; the checkpoint is outside the release by design. |

Deeper failure analysis: [`04-failures.md`](04-failures.md).
