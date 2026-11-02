# MetaCentrum & Remote Cluster Operations Tutorial & Cheat Sheet

A comprehensive, copy-paste ready reference guide for monitoring jobs, inspecting training epochs and logs, comparing queue lengths, and synchronizing code/results across **MetaCentrum PBS Pro**, **Remote Dichtator (RTX PRO 4000)**, and **Local WSL**.

---

## 1. Quick Host Reference

| Target Host | SSH Host Alias | Root Path | Python Environment | Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **MetaCentrum** | `meta` / `perun` | `/storage/brno2/brno2/kolardavidcz/DrugEx` | `.../.venv/bin/python` | PBS Pro cluster (CPU & GPU nodes) |
| **Dichtator** | `dichtator` | `/home/kolarv/DrugEx` | `.../.venv/bin/python` | 48 CPUs, 128 GB RAM, 2x RTX PRO 4000 (Blackwell, 24GB) |
| **Local WSL** | `localhost` | `/home/kolar/DrugEx` | `.../.venv/bin/python` | Local development & visualization |

---

## 2. Queue Status: Tracking Your Jobs

### Check Your Active Jobs (One-liner)
```bash
ssh meta "qstat -u kolardavidcz"
```
* **Job States**:
  * `Q` = **Queued** (Waiting in queue for an available node)
  * `R` = **Running** (Currently executing on a compute node)
  * `E` = **Exiting** (Finished computation, flushing logs/rsyncing back)
  * `H` = **Held** (Job is placed on hold)

### Full Job Details (Find Worker Node, Memory & Walltime)
```bash
ssh meta "qstat -f <JOB_ID>"

# Quick filter: See which worker node (e.g. elmu1-24) is executing your job
ssh meta "qstat -f <JOB_ID> | grep -E '(exec_host|resources_used|Job_Name)'"
```

### Cancel a Job
```bash
ssh meta "qdel <JOB_ID>"

# Cancel all your jobs at once
ssh meta "qselect -u kolardavidcz | xargs qdel"
```

---

## 3. Queue Comparisons: Queue Length & Waiting Times

MetaCentrum has many queues. By default, jobs with `:ngpus=1` route to `gpu`, which often has **thousands of waiting jobs**. CPU queues like `p2e_1d` or `q_4h` have **almost zero wait time**.

### Compare Queue Lengths (Running vs Waiting)
Filter out internal reservation queues to see real cluster queues:
```bash
ssh meta "qstat -q | grep -v '^M[0-9]'"
```

### Compare Relevant Queues for DrugEx
```bash
ssh meta "qstat -q | grep -E '^(Queue|gpu|q_4h|q_1d|p2e_1d)'"
```

| Queue | Typical Walltime | Hardware Profile | Queued (Wait Time) | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`p2e_1d`** | 24 hours | Fast multi-core CPU (`ncpus=16`) | **0 – 10 jobs (< 1 min wait)** | **Fastest for CPU ROCS RL** |
| **`q_4h`** | 4 hours | Fast turnaround CPU (`ncpus=16`) | **10 – 30 jobs (< 5 min wait)** | Short exploration / sanity checks |
| **`q_1d`** | 24 hours | Standard CPU cluster | Moderate wait | Long standard CPU runs |
| **`gpu`** | 24–48 hours | GPU nodes (`ngpus=1`) | **1,500 – 2,200 jobs (Hours/Days wait)** | Large GPU training (only if required) |

### Inspect What Is in a Specific Queue
```bash
# See all jobs waiting or running in the GPU queue
ssh meta "qstat gpu"

# See all jobs in the fast 4-hour CPU queue
ssh meta "qstat q_4h"

# See queue limits, priority, and enabled nodes
ssh meta "qstat -Qf q_4h"
```

---

## 4. Live Epoch Tracking & Last 10 Lines of Runs

### A. Automated 1-Step Live Epoch Monitor (Easiest)
We created a dedicated monitor script that finds the worker host of every running job and displays its latest training epochs automatically:
```bash
ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/monitor_live_epochs.py"
```

### B. Manually View Live Epochs from a Running Job (Worker Scratch)
While a job is running, its output lives in the worker node's local `/scratch*/` space:
```bash
# 1. Find the worker hostname
HOST=$(ssh meta "qstat -f <JOB_ID> | grep 'exec_host =' | cut -d'/' -f1 | awk '{print \$NF}'")
echo "Running on $HOST"

# 2. View the last 10 training epochs (epoch, loss, reward):
ssh meta "ssh $HOST 'tail -n 10 /scratch*/kolardavidcz/job_<JOB_ID>/output/models/SequenceRNN_reinforced_fit.tsv'"

# 3. View the last 10 lines of the live execution log (policy progress & sampling):
ssh meta "ssh $HOST 'grep -v RuntimeWarning /scratch*/kolardavidcz/job_<JOB_ID>/output/execution.log | tail -n 10'"
```

### C. View Last 10 Lines for ALL Finished / Stored Runs
Completed runs are stored in `/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/`:

#### 1. List All Available Runs
```bash
ssh meta "ls -ld /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*"
```

#### 2. Loop & Print Last 10 Epochs for Every Finished Run
```bash
ssh meta '
for run in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do
  if [ -f "$run/models/SequenceRNN_reinforced_fit.tsv" ]; then
    echo "=================================================================="
    echo "RUN: $(basename "$run")"
    echo "=================================================================="
    tail -n 10 "$run/models/SequenceRNN_reinforced_fit.tsv"
    echo ""
  fi
done
'
```

#### 3. Loop & Print Final Scorecards (ROCS Tail, SA, QED, HAC) for Every Run
```bash
ssh meta '
for run in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do
  if [ -f "$run/comparison_scorecard.csv" ]; then
    echo "=== $(basename "$run") ==="
    cat "$run/comparison_scorecard.csv"
    echo ""
  fi
done
'
```

#### 4. View Last 10 Lines of Execution Log for a Specific Run
```bash
ssh meta "tail -n 10 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/execution.log"
```

---

## 5. Bidirectional Synchronization

### A. WSL $\leftrightarrow$ MetaCentrum

```bash
# ---------------------------------------------------------
# 1. PUSH CODE (WSL -> MetaCentrum)
# Sends updated code while ignoring virtualenvs and output
# ---------------------------------------------------------
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    /home/kolar/DrugEx/_david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/

# ---------------------------------------------------------
# 2. PULL RUN RESULTS (MetaCentrum -> WSL)
# Fast sync: pulls scorecards, figures, and CSVs (skips 22MB .pkg weights)
# ---------------------------------------------------------
rsync -avz --exclude='*.pkg' \
    meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/

# ---------------------------------------------------------
# 3. PULL SPECIFIC RUN WITH ALL CHECKPOINTS (MetaCentrum -> WSL)
# ---------------------------------------------------------
rsync -avz \
    meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/ /home/kolar/DrugEx/_david/output/<RUN_NAME>/
```

---

### B. WSL $\leftrightarrow$ Dichtator

> [!IMPORTANT]
> **Zero-Disruption VPN Rule**: Never reconnect or restart Cisco AnyConnect VPN while active. Test connection with `ssh -o BatchMode=yes -o ConnectTimeout=5 dichtator "echo OK"` before syncing.

#### Option 1: Fast Rsync (Recommended)
```bash
# ---------------------------------------------------------
# 1. PUSH CODE (WSL -> Dichtator)
# ---------------------------------------------------------
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    /home/kolar/DrugEx/_david/code/ dichtator:~/DrugEx/_david/code/

# ---------------------------------------------------------
# 2. PULL RESULTS (Dichtator -> WSL)
# ---------------------------------------------------------
rsync -avz --exclude='*.pkg' \
    dichtator:~/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/

# ---------------------------------------------------------
# 3. PUSH RESULTS (WSL -> Dichtator)
# If you want to mirror results generated on MetaCentrum to Dichtator
# ---------------------------------------------------------
rsync -avz --exclude='*.pkg' \
    /home/kolar/DrugEx/_david/output/ dichtator:~/DrugEx/_david/output/
```

#### Option 2: Git Remote Sync
```bash
# Push committed changes to Dichtator
git push dichtator main

# Pull changes from Dichtator
git pull dichtator main
```

---

## 6. Remote Dichtator Inspection

Dichtator has 48 CPU cores, 128 GB RAM, and dual Blackwell GPUs (2x 24GB).

```bash
# Check GPU utilization & free RAM on Dichtator
ssh -o BatchMode=yes dichtator "nvidia-smi && free -h"

# Check running Python processes on Dichtator
ssh -o BatchMode=yes dichtator "ps aux | grep python | grep -v grep"

# Interactive SSH login to Dichtator workspace
ssh -t dichtator "cd ~/DrugEx && bash"
```

---

## 7. Productive `.bashrc` Aliases (1-Word Commands)

Add these aliases to your local WSL `~/.bashrc` (`nano ~/.bashrc`):

```bash
# ==============================================================================
# DrugEx Remote Operations Aliases
# ==============================================================================

# --- MetaCentrum ---
alias meta-jobs='ssh meta "qstat -u kolardavidcz"'
alias meta-queues='ssh meta "qstat -q | grep -v \"^M[0-9]\""'
alias meta-runs='ssh meta "ls -ld /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*"'
alias meta-live='ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/monitor_live_epochs.py"'

meta-push() {
    rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
        /home/kolar/DrugEx/_david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/
}

meta-pull() {
    rsync -avz --exclude='*.pkg' \
        meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/
}

meta-scorecards() {
    ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/comparison_scorecard.csv" ]; then echo "=== $(basename "$r") ==="; cat "$r/comparison_scorecard.csv"; echo ""; fi; done'
}

meta-epochs() {
    ssh meta 'for r in /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/*; do if [ -f "$r/models/SequenceRNN_reinforced_fit.tsv" ]; then echo "=== $(basename "$r") ==="; tail -n 10 "$r/models/SequenceRNN_reinforced_fit.tsv"; echo ""; fi; done'
}

# --- Dichtator ---
alias dichtator-status='ssh -o BatchMode=yes dichtator "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv && free -h"'
alias dichtator-ssh='ssh -t dichtator "cd ~/DrugEx && bash"'

dichtator-push() {
    rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
        /home/kolar/DrugEx/_david/code/ dichtator:~/DrugEx/_david/code/
}

dichtator-pull() {
    rsync -avz --exclude='*.pkg' \
        dichtator:~/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/
}
```

Then activate them:
```bash
source ~/.bashrc
```
