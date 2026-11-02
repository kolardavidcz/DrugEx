# MetaCentrum & Remote Cluster Operations Guide

A practical, single-source cheat sheet for executing, monitoring, and synchronizing DrugEx reinforcement learning runs across **MetaCentrum PBS Pro**, **Remote Dichtator (RTX PRO 4000)**, and **Local WSL**.

---

## 1. Quick Reference: Hosts & Environments

| Target | SSH Host Alias | Root Directory | Python Virtualenv | Hardware Profile |
| :--- | :--- | :--- | :--- | :--- |
| **MetaCentrum** | `meta` / `perun` | `/storage/brno2/brno2/kolardavidcz/DrugEx` | `.../.venv/bin/python` | PBS Pro GPU cluster (`gpu`, `p2e_1d`) |
| **Dichtator** | `dichtator` | `/home/kolarv/DrugEx` | `/home/kolarv/DrugEx/.venv/bin/python` | 48 CPUs, 128 GB RAM, 2x RTX PRO 4000 (24 GB) |
| **Local WSL** | `localhost` | `/home/kolar/DrugEx` | `/home/kolar/DrugEx/.venv/bin/python` | Development & Visualization orchestration |

---

## 2. MetaCentrum: Checking Queue & Active Runs

### Check All Your Jobs
```bash
ssh meta "qstat -u kolardavidcz"
```
*Job states: `Q` = Queued (waiting for GPU node), `R` = Running, `E` = Exiting / transferring results.*

### Detailed Status & Finding Execution Node
```bash
# Find which worker node (e.g., elmu1-24) is running your job
ssh meta "qstat -f <JOB_ID> | grep exec_host"
```

### Check GPU Utilization on the Running Worker Node
```bash
# Example if job is running on worker node elmu1-24:
ssh meta "ssh elmu1-24 'nvidia-smi'"
```

### Cancel a Job
```bash
ssh meta "qdel <JOB_ID>"
```

---

## 3. Real-Time Run Monitoring: Epochs, Metrics & Scores

### A. Live Epoch Tracking (Active Running Job)
While training runs on a worker node's local scratch, progress is recorded in `execution.log` and `models/SequenceRNN_reinforced_fit.tsv`:

```bash
# 1. View latest training epochs, reward, and losses directly from worker scratch:
ssh meta "ssh <EXEC_HOST> 'tail -n 20 /scratch*/kolardavidcz/job_<JOB_ID>/output/models/SequenceRNN_reinforced_fit.tsv'"

# 2. Live policy gradient progress bar:
ssh meta "ssh <EXEC_HOST> 'grep -v RuntimeWarning /scratch*/kolardavidcz/job_<JOB_ID>/output/execution.log | tail -n 25'"
```

### B. View Completed Run Results (Scorecards & Diagnostics)
Once a run finishes, output is automatically rsync'd to `/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/`:

```bash
# List all run directories on cluster storage:
ssh meta "ls -l /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/"

# View final comparison scorecard (ROCS tail, QED, SA, HAC, Validity):
ssh meta "cat /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/comparison_scorecard.csv"

# Read markdown scientific summary:
ssh meta "cat /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/summary.md"

# Check epoch fit progression of a finished run:
ssh meta "tail -n 15 /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/models/SequenceRNN_reinforced_fit.tsv"
```

---

## 4. Submitting New Training Runs

All submission scripts reside in `_david/code/metacentrum/scripts/`:

```bash
# Submit specific hypothesis script:
ssh meta "qsub /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/scripts/h7_tversky_scaffold_hop.sh"
ssh meta "qsub /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/scripts/h8_beta_baseline_penalization.sh"
ssh meta "qsub /storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/metacentrum/scripts/h9_dynamic_epsilon_decay.sh"
```

---

## 5. Synchronization Workflows (WSL $\leftrightarrow$ MetaCentrum $\leftrightarrow$ Dichtator)

### A. WSL $\leftrightarrow$ MetaCentrum Sync

```bash
# --- PUSH CODE: Send modified code from local WSL to MetaCentrum ---
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    _david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/

# --- PULL RUN RESULTS: Download scorecards, logs & generated figures to WSL ---
# (Excludes heavy 22MB .pkg model checkpoints for fast download)
rsync -avz --exclude='*.pkg' \
    meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ _david/output/

# --- PULL EVERYTHING (including model weights): ---
rsync -avz \
    meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/<RUN_NAME>/ _david/output/<RUN_NAME>/
```

---

### B. WSL $\leftrightarrow$ Dichtator Sync

> [!IMPORTANT]
> **Zero-Disruption VPN Rule**: Never reconnect or restart Cisco AnyConnect VPN while active. Test connection with `ssh -o BatchMode=yes -o ConnectTimeout=5 dichtator "echo OK"` before syncing.

#### Option 1: Fast Rsync Sync (Recommended for quick code edits & testing)
```bash
# --- PUSH CODE from WSL to Dichtator ---
rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
    _david/code/ dichtator:~/DrugEx/_david/code/

# --- PULL RUN RESULTS from Dichtator to WSL ---
rsync -avz --exclude='*.pkg' \
    dichtator:~/DrugEx/_david/output/ _david/output/
```

#### Option 2: Git Remote Sync
```bash
# Push committed local changes to remote dichtator branch:
git push dichtator main

# Pull remote commits from dichtator into local WSL:
git pull dichtator main
```

---

## 6. Productive Shell Aliases (Add to `~/.bashrc`)

Copy these functions into your local WSL `~/.bashrc` to monitor and sync with simple one-word commands:

```bash
# --- MetaCentrum Helpers ---
alias meta-jobs='ssh meta "qstat -u kolardavidcz"'
alias meta-runs='ssh meta "ls -la /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/"'

# Sync code to MetaCentrum:
meta-push() {
    rsync -avz --exclude='.venv' --exclude='_david/output' --exclude='*.log' \
        /home/kolar/DrugEx/_david/code/ meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/code/
}

# Pull results and figures from MetaCentrum:
meta-pull() {
    rsync -avz --exclude='*.pkg' \
        meta:/storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/ /home/kolar/DrugEx/_david/output/
}

# View scorecard of a run: meta-scorecard sprint_h2_without_qed_harrington
meta-scorecard() {
    ssh meta "cat /storage/brno2/brno2/kolardavidcz/DrugEx/_david/output/$1/comparison_scorecard.csv"
}

# --- Dichtator Helpers ---
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

After updating `~/.bashrc`, run:
```bash
source ~/.bashrc
```
