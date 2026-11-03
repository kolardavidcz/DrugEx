# MetaCentrum & Remote Cluster Operations Guide

A practical, single-source cheat sheet for executing, monitoring, and synchronizing DrugEx reinforcement learning runs across **MetaCentrum PBS Pro**, **Remote Dichtator (RTX PRO 4000)**, and **Local WSL**.

---

## 1. Quick Host Reference

| Target Host | SSH Host Alias | Root Path | Python Environment | Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **MetaCentrum** | `meta` / `perun` | `/storage/brno2/brno2/kolardavidcz/DrugEx` | `.../.venv/bin/python` | PBS Pro cluster (CPU & GPU nodes) |
| **Dichtator** | `dichtator` | `/home/kolarv/DrugEx` | `.../.venv/bin/python` | 48 CPUs, 128 GB RAM, 2x RTX PRO 4000 (Blackwell, 24GB) |
| **Local WSL** | `localhost` | `/home/kolar/DrugEx` | `.../.venv/bin/python` | Local development & visualization |

---

## 2. Unified Run Directory Protocol (`_david/code/tmp/`)

All execution scripts across MetaCentrum and Dichtator live in `_david/code/tmp/`:

```
_david/code/tmp/
├── to_run/          <-- Staged scripts awaiting execution
└── previous_runs/   <-- Archived scripts already executed (can be safely cleaned)
```

### Execution Protocol Rule:
1. When triggering a script, append execution metadata to the end:
```bash
# ==============================================================================
# CONFIGURASION RUNED at DICHTATOR / METACENTRUM ID: <TARGET_AND_JOB_ID>
# ==============================================================================
```
2. Move the script to `previous_runs/`:
```bash
mv _david/code/tmp/to_run/<script_name> _david/code/tmp/previous_runs/
```
3. **If a run fails**, move the script back to `to_run/` for debugging.
4. `previous_runs/` can be safely deleted at any time because every completed run records its exact parameters in `_david/output/<category>/<run_id>/manifest.json`.

---

## 3. High-Signal Job & Queue Monitoring

### Informative Job Status with Summary Counts
```bash
# Print active jobs + summary counts of Queued vs Running
ssh meta 'qstat -u kolardavidcz | awk "NR>5 {print \$0; if (\$10==\"R\") r++; else if (\$10==\"Q\") q++} END {print \"---------------------------------------------------\"; printf \"TOTAL: %d Running, %d Queued\n\", r, q}"'
```

### Fast Live Epoch & Training Progress (One-Liner)
Automatically checks all running jobs on their assigned worker scratch nodes:
```bash
ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/monitor_live_epochs.py"
```

### Compare Queue Lengths (Filter Internal Reservations)
```bash
ssh meta "qstat -q | grep -E '^(Queue|q_4h|q_1d|p2e_1d|gpu)'"
```

| Queue | Walltime | Profile | Queue Waiting Time | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`p2e_1d`** | 24h | 16+ CPU cores | **0 – 5 jobs (< 1 min)** | **Fastest for CPU ROCS RL** |
| **`q_4h`** | 4h | 16+ CPU cores | **10 – 30 jobs (< 5 min)** | Quick checks / short runs |
| **`q_1d`** | 24h | 16+ CPU cores | Fast turnaround | Standard 24h CPU runs |
| **`gpu`** | 24–48h | GPU nodes | **2,000+ jobs (Days)** | Massive deep learning only |

### Cancel Specific or All Queued Jobs
```bash
# Cancel single job:
ssh meta "qdel <JOB_ID>"

# Cancel all your currently queued (Q) jobs without killing running ones:
ssh meta "qselect -u kolardavidcz -s Q | xargs -r qdel"
```

---

## 4. Inspecting Stored Run Results & Scorecards

### Print Last 10 Training Epochs for All Runs
```bash
ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/models/SequenceRNN_reinforced_fit.tsv" ]; then echo "==================== $(basename "$r") ===================="; tail -n 10 "$r/models/SequenceRNN_reinforced_fit.tsv"; echo ""; fi; done'
```

### Print Comparison Scorecards (ROCS, SA, QED, HAC) for All Runs
```bash
ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/comparison_scorecard.csv" ]; then echo "=== $(basename "$r") ==="; cat "$r/comparison_scorecard.csv"; echo ""; fi; done'
```

---

## 5. Bidirectional Synchronization Workflows

### A. WSL $\leftrightarrow$ MetaCentrum

```bash
# PUSH CODE to MetaCentrum:
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    /home/kolar/DrugEx/_david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/

# PULL RUN RESULTS to WSL (skips heavy 22MB .pkg model weights):
rsync -avz --exclude='*.pkg' \
    meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/
```

### B. WSL $\leftrightarrow$ Dichtator

```bash
# PUSH CODE to Dichtator:
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    /home/kolar/DrugEx/_david/code/ dichtator:~/DrugEx/_david/code/

# PULL RESULTS from Dichtator to WSL:
rsync -avz --exclude='*.pkg' \
    dichtator:~/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/
```

---

## 6. Productive `.bashrc` Aliases

Add to your `~/.bashrc`:

```bash
# Informative job status with count summary
alias meta-jobs='ssh meta '\''qstat -u kolardavidcz | awk "NR>5 {print \$0; if (\$10==\"R\") r++; else if (\$10==\"Q\") q++} END {print \"---------------------------------------------------\"; printf \"TOTAL: %d Running, %d Queued\n\", r, q}"'\'''

# Live epoch progression
alias meta-live='ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/monitor_live_epochs.py"'

# Quick queue comparison
alias meta-queues='ssh meta "qstat -q | grep -E \"^(Queue|q_4h|q_1d|p2e_1d|gpu)\""'

# Sync commands
alias meta-push='rsync -avz --exclude=".venv" --exclude="_david/output" --exclude="*.log" /home/kolar/DrugEx/_david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/'
alias meta-pull='rsync -avz --exclude="*.pkg" meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/'
alias dichtator-push='rsync -avz --exclude=".venv" --exclude="_david/output" --exclude="*.log" /home/kolar/DrugEx/_david/code/ dichtator:~/DrugEx/_david/code/'
alias dichtator-pull='rsync -avz --exclude="*.pkg" dichtator:~/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/'
```
