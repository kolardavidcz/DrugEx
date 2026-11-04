# 06 — Storage, quotas and cleanup

MetaCentrum gives you **two different filesystems with different lifetimes**. Confusing them
is the most expensive mistake in this package.

| | `$HOME` | `$STORAGE_DIR` |
| :--- | :--- | :--- |
| Path | `/storage/praha1/home/kolardavidcz` | `/storage/brno2/brno2/kolardavidcz/DrugEx` |
| Backing site | vestec1-elixir | brno2 |
| Lifetime | long, personal | long, project |
| Written by | PBS (`<job>.o<seq>` logs) | `sync`, jobs, `pull` |
| Holds | PBS stdout/stderr | releases, models, specs, run results, `.venv` |

`~` on the login node is the **praha1** home, not brno2. `job_template.sh` never writes to
`$HOME`; PBS does (stdout/stderr). Long runs produce tens of megabytes there because Python
warning floods are repeated per process, so sweep `$HOME/*.o*` after pulling them (chapter
03 §3).

`/storage/brno2/home/kolardavidcz` is a home-style symlink into the brno2 tree and resolves
to the same files as `/storage/brno2/brno2/kolardavidcz`. The canonical spelling — the one
in `storage_root()` and in every spec — is the doubled site form. Use it; the `home` alias is
owned by `nobody` and confuses some tools.

## 1. Layout and the size of each part

```
$STORAGE_DIR/
├── .venv/                       5.7 G  validated Python 3.11 environment (shared, versioned)
├── releases/<sha>/              ~23 M  immutable `git archive` of one commit
├── current -> releases/<sha>           what `sync` last staged
├── models/                      ~22 M  Papyrus .pkg/.vocab staged by `sync`
├── jobs/<name>/spec.json         ~2 K  exact args + revision of each submission
└── _david/
    ├── output/<run-id>/                finished (or partial) runs, copied back by the job
    └── data/                           ad-hoc reference inputs, e.g. EGCG_ensemble_4conf.sdf
```

Everything except `.venv` is reconstructible from git or from a re-run. `.venv` is not: it is
the environment in which every result was produced.

## 2. `.venv` is a contract, not a cache

Observed state (2026-11-04):

- Python 3.11.11, created by `uv` from a CVMFS-provided interpreter
  (`pyvenv.cfg: home = /cvmfs/software.metacentrum.cz/.../python-3.11.11...`).
- `include-system-site-packages = false`, so nothing leaks in from CVMFS at runtime except
  the interpreter itself.
- `site-packages/drugex_local.pth` adds `/storage/brno2/brno2/kolardavidcz/DrugEx` to
  `sys.path`. The job still puts the scratch release first via `PYTHONPATH`, so the release —
  not storage — is imported.
- No `pip` module. Packages are managed with `uv`, not `pip install` inside the venv.
- Includes `rdkit 2026.9.1`, `numpy 2.4.6`, `pandas 3.1.0`, `torch` with the CUDA wheels,
  `matplotlib`, `gitpython`, `joblib`, `networkx`, and the rest of `pyproject.toml`.
- 5.7 GB, dominated by CUDA libraries.

Rules:

- Never delete or "refresh" `.venv` to fix a dependency problem without the user's explicit
  go-ahead. Rebuilding is a manual, version-sensitive operation (CVMFS python + uv + the CUDA
  torch wheel), and the result must be validated by a real submission: the template guards
  plus the first epoch row in `models/SequenceRNN_reinforced_fit.tsv` are the acceptance test.
- `sync` does not touch `.venv`. Code updates never invalidate it, because the code is
  supplied by the release, not installed into the venv.
- If `.venv` ever disappears, every submission fails at the template's first guard with
  `Missing validated environment`; that is the intended failure mode (chapter 04 §4).

## 3. Releases

`sync` writes `releases/<sha>/` and repoints `current`. Properties:

- One directory per commit, never modified afterwards. Re-running `sync` for the same commit
  overwrites nothing important (`tar -xf` into an existing directory).
- `current` is a symlink; `readlink -f` gives the release path to quote in a paper or a bug
  report.
- Old releases accumulate at ~23 MB each. Pruning them is safe **once no queued or running
  job references them**. A job's spec names its release explicitly, so check before deleting:

```bash
ssh meta 'S=/storage/brno2/brno2/kolardavidcz/DrugEx
  ls -1 $S/releases
  grep -h "\"source\"" $S/jobs/*/spec.json | sort -u
  readlink -f $S/current'
```

Keep the current release and anything referenced by a spec you still care about. In practice
`.venv` dwarfs releases, so pruning is an aesthetic exercise, not a space emergency.

## 4. `models/`

`sync` rsyncs `Papyrus05.5_smiles_rnn_PT.pkg` and `.vocab` from the local checkout (resolved
by `PipelineConfig.resolve_pretrained_paths()`). This is deliberate: the checkpoint is 20 MB
and does not belong in git or in every release.

A job that is already queued expects these files to be present when it starts. Do not delete
or replace them while jobs are queued; a mismatched `.pkg`/`.vocab` pair produces the
`valid_ratio → 0` symptom described in chapter 02.

## 5. `jobs/` and `_david/output/`

- `jobs/<name>/spec.json` — one per submission, overwritten if the same name is submitted
  again. This is why unique names matter. Pull it before deleting (chapter 03 §3).
- `_david/output/<run-id>/` — written by the job's EXIT trap from scratch, then never touched
  by the cluster again. `pull` copies it to WSL.

Scratch (`$SCRATCHDIR`, e.g. `/scratch/job_<id>`) is **not** storage: it is local disk on the
worker, invisible from the login node unless you SSH to the node, and purged after the job.
The template copies the run directory back before cleaning it, and refuses to clean if the
copy failed.

## 6. Quotas and accounting

The frontends do not expose a per-user quota tool (`lquota`, `lfs_quotas`, `quotacheck` are
absent). `df -h` reports the shared filesystem, not your allowance:

| Filesystem | Size | Used | Available |
| :--- | ---: | ---: | ---: |
| brno2 (storage) | 2.0 P | 1.7 P | 353 T |
| vestec1-elixir (home) | 1.4 P | 478 T | 938 T |

Practical consequences:

- You will not exhaust brno2, but a full home filesystem breaks PBS log writing and therefore
  hides job output. Keep `$HOME` clean.
- A job that dies with `No space left on device` is almost always out of **scratch**, not
  storage (chapter 05 §5).
- Measure your own footprint rather than guessing:

```bash
ssh meta 'S=/storage/brno2/brno2/kolardavidcz/DrugEx
  du -sh $S $S/.venv $S/releases $S/models $S/_david/output 2>/dev/null
  du -sh $HOME'
```

## 7. The 2026-11-04 cleanup (reference state)

Everything below was checksum-verified against WSL before deletion:

- 16 run directories, 462 MB, byte-identical (`rsync -anc`) → deleted from storage.
- `_david/data/` copied back except one unique SDF → deleted.
- 6 `spec.json` files and 24 PBS logs → pulled to `_david/output/_metacentrum_jobs/` → deleted.
- `releases/*` and `models/` rebuilt clean by `sync` from `ece2f23`.
- `.venv` kept.

Result: 6.2 GB → 5.7 GB, with storage containing only `.venv`, one release, `current`,
`models`. Post-cleanup comparison: `WSL 48 folders / 29,556 files | remote 0 folders / 0
files | local-only 48`. The 48 local-only directories are expected: the remote copy was
intentionally removed after verification.

After the first end-to-end validation of the rewritten package (job `25256775`, release
`78ec60a`):

- Two run directories (`20261104_0339_…`, `20261104_0344_…`) pulled and verified → deleted.
- Three `spec.json` files (`jobs/drugex_smoke_{e7c6b8a,a8e26b4,78ec60a}`) and four PBS logs
  (`drugex_smoke_*.o*`, `acl_*.o*`) → pulled to `_david/output/_metacentrum_jobs/` → deleted.
- Stale releases `e7c6b8a`, `a8e26b4`, `ece2f23`, `5012debb` removed; `78ec60a` is the only
  release left and no job referenced the others (`qstat -u` was empty).

Since then, **local `_david/output/` is the only copy of every historical result**. If a
backup is wanted, make it off both hosts (external drive / cloud), not by re-uploading to
MetaCentrum.

## 8. Housekeeping checklist

After each ad-hoc campaign:

- [ ] `submit pull` and re-run it until it reports nothing new.
- [ ] `compare_output_trees.py --compare …` shows `remote-only 0`.
- [ ] Pull `jobs/` and `$HOME/*.o*` into `_david/output/_metacentrum_jobs/`.
- [ ] Delete `$STORAGE/_david/output/<run>/` and `$HOME/*.o*`.
- [ ] Leave `.venv`, `models/`, `current`, and the current release untouched.
- [ ] Record the job ids and the storage state in the next session log if anything was
      deleted.
