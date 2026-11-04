# How `compare_output_trees.py` works

Agent note for [`compare_output_trees.py`](compare_output_trees.py). Read this before
changing the script, trusting its numbers, or "fixing" a reported difference.

**Scope:** `_david/output/` on two hosts (WSL ↔ `dichtator`). It answers *layout* questions —
what moved, what is missing, which folders are thesis-grade results — not "are the numbers
in these CSVs the same". Content is used only to disambiguate, never audited field by field.

**Hard guarantees:** stdlib only, read-only (every write goes to `--write` / `--json-out`),
no locks, safe to run while a job is writing. Remote side runs the same file over SSH stdin,
so host and tooling can never drift apart.

## Three CLI modes

| Mode | Command | Purpose |
| --- | --- | --- |
| scan | `--scan-only --root PATH` | print one host's inventory as JSON; this is what the SSH side executes |
| compare | `--compare --local-root R --remote "ssh dichtator" --remote-root '~/DrugEx/_david/output'` | scan both, match, print a summary table |
| report | compare + `--write X.md [--json-out X.json.gz]` | full markdown report + reproducible snapshot |

The remote command is built as
`ssh dichtator python3 - --scan-only --root … --hash-max … --exclude … < script.py`.
Consequence: the remote needs `python3` ≥ 3.8 and nothing else — no repo checkout, no venv,
no imports from `_david`. Do not add package imports here; they will not exist on the other host.

## 1. Scan (`scan_tree`)

1. `_walk_files` walks the tree (symlinks are **not** followed; a symlinked directory is
   recorded as a single non-recursive entry). `.agents`, `.git`, `__pycache__`,
   `.ipynb_checkpoints`, caches are skipped by `DEFAULT_EXCLUDES`; `--exclude` adds more.
2. `_find_units` marks a directory as an **experiment folder** when it *directly* contains a
   marker file (`manifest.json`, `submission.json`, `execution.log`, `summary.md`,
   `provenance.json`) or a marker dir (`models/`, `raw_data/`, `figures/`, `logs/`).
3. Nested candidates are folded into the outermost ancestor (see edge case 1), so the unit is
   the experiment, not `raw_data/pilot` inside it.
4. Every file is attributed to its *deepest* unit; files outside any unit are counted in
   `orphan_files` and otherwise ignored (loose reports, `_COOL_STUFF`, notebooks).
5. Per unit it records file names, sizes, md5 of every file ≤ `--hash-max` (default 1 MiB,
   `.pkg/.pt/.pth/.ckpt/.sdf/.npy/.npz/.zip/.tar` skipped), plus:

| Metric | Source | Note |
| --- | --- | --- |
| `epochs_done` | max first column of the last line of `models/*_smiles.tsv`, else `epoch N` in logs | tail-read, never loads the TSV |
| `epochs_planned` | `manifest.json` → `config.epochs` (also `n_epochs`, `training.epochs`, …) | `None` for old manifests — reported as `?` |
| `n_pkg` | count of `models/*.pkg` | presence/absence of weights |
| `canonical_figs` | `figures/*.png` ∩ `CURRENT_FIGURES` | `04_…` is legacy, counted separately |
| `raw_reinforced_rows` / `raw_baseline_rows` | data rows of `raw_data/*_raw.csv.gz` | header excluded, streamed |
| `smiles_rows` | rows of all `models/*_smiles.tsv` | ⚠ one row **per molecule per epoch**, not an arm size |

## 2. Classification (`classify`)

First matching rule wins; returns `(kind, verdict, reason)`:

1. name matches `smoke|test|incomplete|negative_control|debug|demo|dry|pilot|scratch|tmp` → `test`
2. no files → `aborted`
3. `epochs_done == 0` **and** no evaluation arm → `aborted` (surviving checkpoints do not save it)
4. `0 < epochs_done ≤ 20` → `test` (the "we only ran a few epochs" rule)
5. 0 epochs but a full evaluation arm → falls through to 6–8, so a run whose telemetry was
   trimmed is still judged by its deliverables
6. canonical panels < 10 **and** no legacy `04_…` → `real / partial`
7. molecules < 500 → `real / partial`
8. else `real / include`

`molecules = raw_reinforced_rows or smiles_rows` — the evaluation arm wins; the per-epoch TSV
is only a fallback because it inflates by the epoch count.

## 3. Matching (`match_inventories`)

Cascade, and a unit leaves the pool as soon as it matches:

| Step | Rule | Reported as |
| --- | --- | --- |
| 1 | identical relative path | `matched`, via `path` |
| 2 | identical folder basename, unique on **both** sides | `moved`, via `name` |
| 3 | `max(Jaccard, containment) ≥ 0.6` on relative file names, with ≥ `min(3, |local|)` shared | `moved`, via `file-overlap` |
| 4 | shared md5 ≥ 20 % of the smaller hash set | `moved`, via `content` |
| — | leftovers | `remote_only` / `local_only` |

`containment = |shared| / min(|L|, |R|)` is what makes a folder that *lost* files still pair
with its sibling instead of appearing twice. Jaccard alone is too strict for that case and
useless for runs that differ by a handful of figures.

After matching, `_partial_parent` runs over `remote_only` against **all** local units, so a
dichtator folder whose files already live under a different WSL path is not called missing:

* sha-first scoring (names are identical across all runs — `execution.log`, `raw_data/*.csv.gz` —
  so name-only scoring happily pairs two unrelated runs; this bit us once),
* ≥ 0.6 → `partial copy` (reports the files that really are absent) or `duplicate` (no gaps,
  "mirrored byte-for-byte") when every remote file exists locally.

## 4. Report sections (`build_report`)

| Section | Answers |
| --- | --- |
| snapshot table | hosts, roots, timestamps, file/byte/folder counts, `orphan_files` |
| bottom line | 0–3 bullets: files only on dichtator, folders only on dichtator, count of relocated folders |
| 1a | dichtator folder that survives in WSL **without** these files (with sizes) |
| 1b / 1c | absent outright, split into worth-rescuing vs test/aborted |
| 2 | moved / duplicated folders, per-pair file counts, files a move failed to carry |
| 3 | WSL-only folders with verdict |
| 4 | matched folders whose file list differs; "extra files by frequency" summary |
| appendix | every dichtator experiment folder with its verdict — the "is this a real run" table |
| re-run | copy-pasteable command |

## Edge cases that already produced wrong output

1. **Nested data dirs.** `raw_data/pilot`, `raw_data/rl_smoke`, `checkpoints/…` carry their own
   `manifest.json` and were counted as separate experiments. Fixed by folding into the
   outermost unit; the price is that a genuinely missing nested sub-study shows up as drift
   in section 4, not as a missing folder.
2. **Name-only parent matching.** Before hash-first scoring, a checkpoint folder was paired
   with `sprint_h1` because both contain `models/*.pkg` with the same base names. Scores are
   now computed on md5, names only as fallback when a unit has no hashes.
3. **`max(raw_rows, smiles_rows)`.** The TSV holds epoch × molecule rows, so a 50-epoch run
   looked like 2500 molecules. Prefer the evaluation arm.
4. **"Absent outright" for a folder that was fully copied elsewhere.** `_partial_parent` used
   to require a non-empty gap list; a fully mirrored folder then fell through to "missing".
   It now returns the empty gap and the pair is reported as a `duplicate`.
5. **Name matches were invisible.** Unique-basename matches were pushed to `matched`, so
   section 2 ("folders that only moved") stayed empty and a rename looked like a no-op. All
   non-path matches now land in `moved`.
6. **Live churn.** Both trees get reorganised while the script runs (observed: a folder renamed
   twice within ten minutes). The report is a snapshot by design — it stamps both scan times
   in the header and the intro says so. Never quote the counts without the timestamp.
7. **Folders with identical panel sets.** Panel counts come from file names only, so a run with
   8 canonical + legacy `04` is `include`, not `partial`. That is deliberate: the pre-2026-11
   suite shipped 9 panels.
8. **`.agents/`** is excluded on purpose: it holds agent reports, not results. Excluding it
   means "everything moved into `.agents`" cannot silently satisfy a missing-folder check.
9. **Duplicate runs on both hosts.** Flat `sprint_h1_*` and
   `20261027_reinforcement_learning/20261101_1500_sprint_h1_*` are separate units on both hosts
   (the reorg copied, not moved). The script compares both rather than assuming one is canonical.
10. **Big JSON.** A full snapshot with hashes is ~19 MB; `--json-out …json.gz` compresses it to
    ~2.6 MB. Use `.gz` unless you need to grep the raw file.

## Reading the verdicts correctly

* `include` = real run, full panel suite (or the 9-panel legacy suite), ≥ 500 molecules. It does
  **not** mean the run matches the current gold-standard config — that is `audit_rl_runs.py`.
* `partial` on a checkpoint-only folder (e.g. dichtator's weights mirror) is expected; its
  figures live in the analysis half.
* `test` is a *name or horizon* judgement. `rl01_opt1_mild_color_heavy` (250 epochs) is `real`
  even though it is a scientific failure; failures are still thesis evidence.

## When you change something

Thresholds and lists live at module top (`TEST_EPOCH_CEILING`, `MIN_MOLECULES`,
`CURRENT_FIGURES`, `DEFAULT_EXCLUDES`, `HASH_EXCLUDE_SUFFIXES`,
`UNIT_FILE_MARKERS` / `UNIT_DIR_MARKERS`). After touching matching or classification, re-run the
synthetic check: two temp trees with a same-path run, a renamed parent dir, a renamed folder
missing its `.pkg`, a remote-only real run and a remote-only smoke test — assert
`matched / moved / remote_only / local_only` and the verdicts. Do not test against
`_david/output/`: it changes under you and is not a fixture.

Then regenerate the real report:

```bash
python3 _david/code/compare_output_trees.py --compare \
    --local-root _david/output \
    --remote "ssh dichtator" --remote-root "~/DrugEx/_david/output" \
    --write _david/output/OUTPUT_TREE_DIFF.md \
    --json-out _david/output/.agents/output_tree_diff_snapshot.json.gz
```

`ruff check _david/code/compare_output_trees.py` must stay clean; the file is not imported by the
package (`_david/code/__init__.py` pulls in torch), so it must remain import-free.
