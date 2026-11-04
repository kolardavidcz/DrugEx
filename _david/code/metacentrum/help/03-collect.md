# 03 — Collecting results

Direction matters: **MetaCentrum → WSL**. Nothing is pushed from the workstation to the
cluster except code releases (`sync`) and the checkpoint (`models/`). Results always travel
home with `pull`.

## 1. Pull

```bash
$PY -m _david.code.metacentrum.submit pull
$PY -m _david.code.metacentrum.submit pull --name 20261104_1530_egcg_gold_gmean
```

`pull` runs

```
rsync -az --info=progress2 meta:$STORAGE/_david/output/  _david/output/
```

- `-a` keeps `manifest.json`, nested directories, permissions and mtimes.
- `-z` compresses; run directories are mostly SDF/CSV/PNG, so this matters over VPN.
- Without `--name`, every remote run directory is pulled. With `--name`, one directory is
  written under `_david/output/<name>/`, not flattened.
- Pulling is idempotent and additive. It never deletes a local run, never overwrites a newer
  local file with an older remote one (`rsync` default is "update if different", and remote
  is always the producer here).
- A run still in progress can be pulled safely: scratch is copied to storage only at the
  end, so an incomplete pull simply misses the newest files. Re-pull after it finishes.

After pulling:

```bash
$PY -m _david.code.audit_rl_runs
```

The audit checks each run's `manifest.json` against the gold standard (`avg_gmean`,
`n_eval_samples=5000`, raw ROCS TanimotoCombo, SA only) and reports missing figures. It
writes `_david/code/RL_RUN_AUDIT.md`, which is the authoritative run table.

## 2. Verify before you delete anything

```bash
python3 _david/code/metacentrum/archive/file_folder_tree_comparasion/compare_output_trees.py \
    --compare \
    --local-root _david/output \
    --remote "ssh meta" \
    --remote-root "/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output"
```

The script streams itself to the remote host (`sys.executable`), so it must be started with
bare `python3`, and it is read-only unless `--write`/`--json-out` is given. A healthy state
after a full pull is:

```
WSL 48 folders / 29,556 files | remote 0 folders / 0 files | matched 0 | moved 0 | remote-only 0 | local-only 48
```

Read it as: remote is empty (everything was already pulled), so `local-only` is expected and
not a loss. Any non-zero `remote-only` or `moved` means a pull is incomplete — re-run `pull`
before deleting anything. Full semantics:
[`../archive/file_folder_tree_comparasion/COMPARE_OUTPUT_TREES.md`](../archive/file_folder_tree_comparasion/COMPARE_OUTPUT_TREES.md).

For byte-level proof of a specific file (the layout checker deliberately does not audit CSV
content), use rsync's checksum comparison mode or md5:

```bash
rsync -anc "meta:$STORAGE/_david/output/" _david/output/ | head
# empty output (or only directory-time lines) = identical
md5sum <local-file>
ssh meta "md5sum <remote-file>"
```

`rsync -anc` compares by checksum, not size+mtime, which is what makes it trustworthy for
`*.pkg` checkpoints and `*.sdf` ensembles.

## 3. Keep the provenance off-cluster

The job specs and PBS logs are cheap and explain runs months later. They are pulled to
`_david/output/_metacentrum_jobs/` (git-ignored):

```
_david/output/_metacentrum_jobs/
├── jobs/<experiment>/spec.json      # exact args + revision per submission
└── pbs_logs/<name>.o<sequence>      # stdout/stderr of every attempt
```

When a run raised, the PBS log is the only place the traceback survives (scratch is gone).
Copy it before deleting remote logs:

```bash
rsync -a "meta:$STORAGE/jobs/"           _david/output/_metacentrum_jobs/jobs/
rsync -a "meta:\$HOME/*.o*"              _david/output/_metacentrum_jobs/pbs_logs/
```

## 4. Freeing remote storage

Policy: **pull → verify → delete**, in that order, with the comparison tool output as the
gate. Never delete a remote run because "it looks the same" or because it is old.

Safe to delete once verified:

- `$STORAGE/_david/output/<run>/` — the authoritative copy now lives in WSL.
- `$STORAGE/jobs/<experiment>/` — after the specs are in `_david/output/_metacentrum_jobs/`.
- `$STORAGE/_david/data/` — canonical inputs are in git (`_david/data/`); only ad-hoc
  reference SDFs need pulling first.
- `$HOME/<name>.o*` — after the PBS logs are pulled. They sit in home, not storage, but they
  are the largest accidental files there (tens of MB per long run due to warning floods).
- Old releases: `releases/<sha>` that are neither `current` nor referenced by a running job
  or by a spec you still care about. Keep at least the current one.

Never delete:

- `$STORAGE/.venv/` — the validated environment. Rebuilding it is a multi-hour, version-
  sensitive operation; every job depends on it.
- `$STORAGE/models/` — `sync` refreshes it, but a job already queued expects it to exist.
- `$STORAGE/current` — the symlink `monitor_live_epochs.py` and manual inspection use.

Check what a directory holds before removing it:

```bash
ssh meta 'S=/storage/brno2/brno2/kolardavidcz/DrugEx; du -sh $S/* | sort -h; du -sh $HOME'
```

A completed cleanup leaves storage at roughly `.venv` (5.7 GB) + one release (~23 MB) +
`models` (~22 MB). That is the state as of 2026-11-04; see [`06-storage.md`](06-storage.md).

## 5. What a completed run directory contains

```
_david/output/YYYYmmdd_HHMM_<name>/
├── manifest.json                 # run_id, state, full config, host, revision, versions
├── submission.json               # the MetaCentrum spec (args + release)
├── execution.log                 # pipeline stdout/stderr
├── comparison_scorecard.csv      # baseline vs reinforced, per metric
├── summary.md                    # human-readable run summary
├── top_prioritized_leads.tsv     # ranked candidates
├── models/                       # SequenceRNN_reinforced*.pkg, *_fit.tsv, *_fit.log
├── raw_data/                     # baseline + reinforced arms + reference SDF
├── figures/                      # 8 publication figures + aligned lead SDF
└── logs/                         # reserved for structured logs
```

`manifest.state` is `complete` for a finishing run and records the PBS job id, which
cross-references the pulled PBS log. If `state` is not `complete`, treat the directory as a
partial run: figures may be missing, and the audit will say so.
