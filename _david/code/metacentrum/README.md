# MetaCentrum operations

Single guide for DrugEx on MetaCentrum. MetaCentrum runs ~95 % of production
RL; Dichtator is a secondary host and local WSL is for development, figures and
the test suite.

---

## 1. Storage layout

`STORAGE=/storage/brno2/brno2/kolardavidcz/DrugEx`

```
$STORAGE/
├── releases/<commit-sha>/    # immutable `git archive` of a commit, staged by `grid sync`
├── current -> releases/...   # the release a submission should use
├── models/                   # Papyrus .pkg/.vocab (copied by `grid sync`)
├── jobs/<experiment>/spec.json   # exact arguments submitted, per experiment
├── _david/output/<run-id>/   # one directory per completed run
├── _david/data/              # canonical conformer inputs
└── .venv/                    # validated Python environment, kept across code syncs
```

Code is **never** rsynced into a live tree. `grid sync` stages `git archive
HEAD` under `releases/<sha>` and repoints `current`, so a submitted job always
names one immutable commit. `job_template.sh` copies that release into
`$SCRATCHDIR` and runs from there.

---

## 2. Run directories

Every run lands in one directory named `YYYYmmdd_HHMM[_tag]`:

| Where | Default | Override |
| :--- | :--- | :--- |
| Local WSL | `_david/output/20261104_1530_egcg_gold/` | `--run-tag egcg_gold`, or `--output-dir <path>` verbatim |
| MetaCentrum | `$STORAGE/_david/output/20261104_1530_<experiment>/` | set by `job_template.sh` from `EXP_NAME` |

- The tag is optional: SMILES runs get the bare minute (`20261104_1530`);
  file-based runs (`--molecule foo.sdf`) use the file stem.
- A second run in the same minute gets `_2`, `_3` …, so two runs can never
  share or overwrite a `manifest.json`.
- `manifest.json` records `run_id`, `pbs_job_id` and `source_revision`, which
  is what makes a pulled directory self-describing.

Historical directories (`sprint_h1_*`, `egcg_k4_*`, …) predate this rule and
keep their names: `RL_RUN_AUDIT.md` refers to them.

---

## 3. Submit

```bash
# Stage HEAD (refuses a dirty tracked tree; untracked files are never archived)
python -m _david.code.metacentrum.grid sync

# One job; `--args` is a shell-quoted string passed to _david.code.pipeline
python -m _david.code.metacentrum.grid submit \
    --name egcg_gold_gmean \
    --args "--epochs 200 --min-epochs 50 --patience 50 --criteria gmean" \
    --queue p2e_1d --ncpus 16 --mem 32 --ngpus 0

# Predefined arms
python -m _david.code.metacentrum.grid sprint          # inspect 4-hypothesis matrix
python -m _david.code.metacentrum.grid grid            # inspect 8-arm matrix
python -m _david.code.metacentrum.grid submit-sprint --queue p2e_1d --ngpus 0
```

`submit`/`submit-grid`/`submit-sprint` sync first unless `--no-sync` is given.
The job name is also the run-directory tag, so keep it unique and descriptive.

---

## 4. Monitoring

```bash
# Active jobs + Queued/Running counts
ssh meta 'qstat -u kolardavidcz | awk "NR>5 {print \$0; if (\$10==\"R\") r++; else if (\$10==\"Q\") q++} END {print \"---------------------------------------------------\"; printf \"TOTAL: %d Running, %d Queued\n\", r, q}"'

# Live epoch progress across running jobs (reads worker scratch)
ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/monitor_live_epochs.py"

# Queue lengths
ssh meta "qstat -q | grep -E '^(Queue|q_4h|q_1d|p2e_1d|gpu)'"
```

| Queue | Walltime | Profile | Queue Waiting Time | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`p2e_1d`** | 24h | 16+ CPU cores | **0 – 5 jobs (< 1 min)** | **Fastest for CPU ROCS RL** |
| **`q_4h`** | 4h | 16+ CPU cores | **10 – 30 jobs (< 5 min)** | Quick checks / short runs |
| **`q_1d`** | 24h | 16+ CPU cores | Fast turnaround | Standard 24h CPU runs |
| **`gpu`** | 24–48h | GPU nodes | **2,000+ jobs (Days)** | Massive deep learning only |
- usfull: runone, elixir_1d, p2e_1d, cvut
- what is: charon, gpu_dgx, iti, luna, biocev, elixircz (default), (maybe even p2e_gpu)

```bash
ssh meta "qdel <JOB_ID>"                                    # one job
ssh meta "qselect -u kolardavidcz -s Q | xargs -r qdel"      # all queued, not running
```

---

## 5. Results

```bash
# Last 10 epochs of every run on storage
ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/models/SequenceRNN_reinforced_fit.tsv" ]; then echo "==================== $(basename "$r") ===================="; tail -n 10 "$r/models/SequenceRNN_reinforced_fit.tsv"; fi; done'

# Comparison scorecards
ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/comparison_scorecard.csv" ]; then echo "=== $(basename "$r") ==="; cat "$r/comparison_scorecard.csv"; fi; done'

# Pull every run into _david/output/ (keeps timestamps and manifest.json)
python -m _david.code.metacentrum.grid pull
```

After a pull, `python -m _david.code.audit_rl_runs` re-checks each run against
the gold-standard config and reports missing figures.

---

## 6. Historical runners

`scripts/` holds the frozen per-hypothesis PBS scripts (`h1`…`h9`, `egcg_*`,
supervisor replication, batch submitter). They embed the defaults of their day
and duplicate `job_template.sh`; keep them as provenance, submit through
`grid.py` instead. Their trailing comment records the job id that executed them.

Dichtator scripts remain in `_david/code/tmp/previous_runs/` (git-ignored
scratch). Dichtator is not a supported submission target anymore.
