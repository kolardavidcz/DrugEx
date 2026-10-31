#!/usr/bin/env bash
# ==============================================================================
# DrugEx SequenceRNN 6-Arm RL Experiment Suite Launcher (Resumable + tmux)
# ==============================================================================
# Prioritized 6-arm study order (most meaningful -> least meaningful):
#   1. rl01_opt1_mild_color_heavy   (Primary Thesis Anchor: Option 1 (0.8, 1.2) + 0.2S/0.8C, Harrington, beta=0.0, 250 ep)
#   2. rl02_opt2_scaffold_hop       (Option 2 Sub-Scaffold Hopping: (0.6, 1.4) + 0.3S/0.7C, Harrington, beta=0.0, 250 ep)
#   3. rl03_baseline_tanimoto_combo (Standard ROCS Control: (1.0, 1.0) + 0.5S/0.5C equal-weight TanimotoCombo, beta=0.0, 250 ep)
#   4. rl04_legacy_dice_tversky03_07(Negative Control: Legacy DrugEx (0.3, 0.7) alpha+beta=1 Dice subshape, beta=0.0, 250 ep)
#   5. rl05_opt1_pareto_vs_harrington(Multi-Objective Reward Ablation: Pareto crowding vs Harrington on Option 1, beta=0.0, 250 ep)
#   6. rl06_opt1_beta_penalty_ablation(Penalty Ablation: Beta=0.15 Penalty vs Beta=0.0 on Option 1, Mode Collapse Control)
#
# All 6 runs share the calibrated 3D conformer physics defaults from config.py:
#   - minimize_candidates=True, max_conformers=20, Optimization("MMFF94s", 78.4)
#   - max_reference_conformers=4, reference_rmsd_cutoff=0.35, reference_selection_mode="diverse"
#   - n_samples=500, batch_size=64 (8 Adam steps/epoch), training_seed=42
#
# Quick Usage (works both directly on dichtator OR from local WSL2):
#   ./_david/code/launch_rl_suite.sh                 # Launch/resume in detached tmux (with 12:00 cutoff ON)
#   ./_david/code/launch_rl_suite.sh --restart       # Kill existing session and restart suite
#   ./_david/code/launch_rl_suite.sh --run <EXP_ID>  # Run ONLY a specific experiment (e.g. rl01_opt1_mild_color_heavy)
#   ./_david/code/launch_rl_suite.sh --no-time-limit # Launch/resume ignoring the 12:00 cutoff
#   ./_david/code/launch_rl_suite.sh --status        # Show table of completed/running/pending runs + GPU status
#   ./_david/code/launch_rl_suite.sh --attach        # Attach to live tmux session (Ctrl+B, D to detach)
#   ./_david/code/launch_rl_suite.sh --pull          # Pull completed results from dichtator to local WSL2
#   ./_david/code/launch_rl_suite.sh --stop          # Kill active tmux session
# ==============================================================================

set -euo pipefail

# ==============================================================================
# USER CONFIGURATION (Toggle here or pass --no-time-limit on the CLI)
# ==============================================================================
dont_start_run_after_12_00="${DONT_START_RUN_AFTER_12_00:-true}"

# Daytime cutoff window in 24-hour local time [12:00 .. 19:59].
# When dont_start_run_after_12_00=true, the runner will finish the currently
# active experiment and stop cleanly before starting the next experiment if the
# local clock is between 12:00 (noon) and 19:59 (8:00 PM).
CUTOFF_START_HOUR=12
CUTOFF_END_HOUR=20

TMUX_SESSION="drugex_rl_suite"
SUITE_NAME="rl_suite_egcg_6arm"
REMOTE_HOST="dichtator"
REMOTE_REPO="/home/kolarv/DrugEx"

# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================
ACTION="launch"
INSIDE_WORKER=0
FORCE_LOCAL=0
ONLY_EXP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-time-limit|--ignore-cutoff|--after-12-ok)
            dont_start_run_after_12_00="false"
            shift
            ;;
        --time-limit)
            dont_start_run_after_12_00="true"
            shift
            ;;
        --status|status)
            ACTION="status"
            shift
            ;;
        --attach|attach)
            ACTION="attach"
            shift
            ;;
        --stop|stop|kill)
            ACTION="stop"
            shift
            ;;
        --restart|--rerun|restart|rerun)
            ACTION="restart"
            shift
            ;;
        --run|--only)
            ONLY_EXP="${2:-}"
            shift 2
            ;;
        --pull|pull|sync)
            ACTION="pull"
            shift
            ;;
        --local)
            FORCE_LOCAL=1
            shift
            ;;
        --worker-loop)
            INSIDE_WORKER=1
            shift
            ;;
        --help|-h)
            sed -n '2,27p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)" >&2
            exit 1
            ;;
    esac
done

# ==============================================================================
# HOST ROUTING: If executed on local WSL2, transparently delegate to dichtator
# ==============================================================================
CURRENT_HOST="$(hostname)"
SCRIPT_REL="_david/code/launch_rl_suite.sh"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$ACTION" == "pull" ]]; then
    mkdir -p "${LOCAL_REPO}/_david/output/${SUITE_NAME}"
    echo "[PULL] Syncing ${REMOTE_HOST}:${REMOTE_REPO}/_david/output/${SUITE_NAME}/ -> ${LOCAL_REPO}/_david/output/${SUITE_NAME}/ ..."
    rsync -avz --progress \
        "${REMOTE_HOST}:${REMOTE_REPO}/_david/output/${SUITE_NAME}/" \
        "${LOCAL_REPO}/_david/output/${SUITE_NAME}/"
    echo "[PULL] Complete. Local deliverables at: ${LOCAL_REPO}/_david/output/${SUITE_NAME}"
    exit 0
fi

if [[ "$CURRENT_HOST" != "dichtator"* && "$FORCE_LOCAL" -eq 0 ]]; then
    # Ensure latest script is synced to dichtator before delegating
    scp -q "${LOCAL_REPO}/${SCRIPT_REL}" "${REMOTE_HOST}:${REMOTE_REPO}/${SCRIPT_REL}"
    ssh "${REMOTE_HOST}" "chmod +x ${REMOTE_REPO}/${SCRIPT_REL}"
    if [[ "$ACTION" == "attach" ]]; then
        exec ssh -t "${REMOTE_HOST}" "tmux attach -t ${TMUX_SESSION}"
    else
        RemoteArgs=()
        [[ "$dont_start_run_after_12_00" == "false" ]] && RemoteArgs+=("--no-time-limit")
        [[ -n "$ONLY_EXP" ]] && RemoteArgs+=("--run" "$ONLY_EXP")
        [[ "$ACTION" != "launch" ]] && RemoteArgs+=("--${ACTION}")
        exec ssh -t "${REMOTE_HOST}" "cd ${REMOTE_REPO} && bash ${SCRIPT_REL} --local ${RemoteArgs[*]:-}"
    fi
fi

# ==============================================================================
# EXECUTION ON TARGET COMPUTE HOST (dichtator)
# ==============================================================================
REPO_ROOT="$LOCAL_REPO"
SUITE_DIR="${REPO_ROOT}/_david/output/${SUITE_NAME}"
MASTER_LOG="${SUITE_DIR}/suite_master.log"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

mkdir -p "$SUITE_DIR"

# Prioritized experiments: "EXP_ID|DESCRIPTION|EXTRA_CLI_ARGS"
EXPERIMENTS=(
    "rl01_tanimoto_combo_default|#1 Production Baseline: TanimotoCombo (Color+Shape), Harrington, beta=0.0, eps=0.0, default config|--score-variant tanimoto_combo"
)

# Common calibrated 3D physics & RL arguments shared across runs
COMMON_ARGS=(
    --minimize-candidates
    --optimization MMFF94s
    --dielectric 78.4
    --conformers 20
    --isomers 4
    --reference-conformers 4
    --reference-rmsd 0.35
    --reference-selection diverse
    --reference-attempts 100
    --samples 500
    --batch-size 64
    --eval-samples 1000
    --epsilon 0.0
    --epsilon-end 0.0
    --no-use-decay
    --training-seed 42
    --seed 42
    --timeout 30
    --rocs-clip-upper 1.8
)

# Check whether an experiment directory has completed all stages cleanly
is_run_complete() {
    local exp_dir="$1"
    local manifest="${exp_dir}/manifest.json"
    local reinf_csv="${exp_dir}/raw_data/reinforced_raw.csv.gz"
    local final_pkg="${exp_dir}/models/SequenceRNN_reinforced_final_epoch.pkg"
    if [[ -f "$manifest" && -f "$reinf_csv" && -f "$final_pkg" ]]; then
        if grep -q '"state": *"complete"' "$manifest" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Print human-readable status summary table
print_status() {
    echo "========================================================================================"
    echo " DrugEx 6-Arm SequenceRNN RL Suite Status (Host: $(hostname))"
    echo " Suite Directory : ${SUITE_DIR}"
    echo " Time Gate       : dont_start_run_after_12_00=${dont_start_run_after_12_00} (cutoff ${CUTOFF_START_HOUR}:00..${CUTOFF_END_HOUR}:00)"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo " tmux Session    : ACTIVE (${TMUX_SESSION}) -> attach with: ./_david/code/launch_rl_suite.sh --attach"
    else
        echo " tmux Session    : INACTIVE (no running session)"
    fi
    echo "----------------------------------------------------------------------------------------"
    printf " %-30s | %-22s | %s\n" "EXPERIMENT ID" "STATUS" "PROGRESS / ELAPSED"
    echo "----------------------------------------------------------------------------------------"
    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_id exp_desc _ <<< "$entry"
        local exp_dir="${SUITE_DIR}/${exp_id}"
        local status_str="PENDING"
        local detail_str="$exp_desc"
        if is_run_complete "$exp_dir"; then
            status_str="COMPLETE"
            local elapsed
            elapsed=$("$PYTHON_BIN" -c "import json; d=json.load(open('${exp_dir}/manifest.json')); print(f\"{d.get('elapsed_seconds',0)/60:.1f} min\")" 2>/dev/null || echo "done")
            detail_str="Finished in ${elapsed}"
        elif [[ -f "${exp_dir}/manifest.json" ]]; then
            local raw_state
            raw_state=$("$PYTHON_BIN" -c "import json; print(json.load(open('${exp_dir}/manifest.json')).get('state','unknown'))" 2>/dev/null || echo "unknown")
            local fit_tsv="${exp_dir}/models/SequenceRNN_reinforced_fit.tsv"
            local ep_done=0
            if [[ -f "$fit_tsv" ]]; then
                ep_done=$(( $(wc -l < "$fit_tsv") - 1 ))
                (( ep_done < 0 )) && ep_done=0
            fi
            status_str="${raw_state^^}"
            detail_str="Epoch ${ep_done} logged (${exp_desc})"
        fi
        printf " %-30s | %-22s | %s\n" "$exp_id" "$status_str" "$detail_str"
    done
    echo "========================================================================================"
    if [[ -f "$MASTER_LOG" ]]; then
        echo " Recent Suite Log (${MASTER_LOG}):"
        tail -n 12 "$MASTER_LOG" | sed 's/^/   /'
        echo "========================================================================================"
    fi
}

if [[ "$ACTION" == "status" ]]; then
    print_status
    exit 0
fi

if [[ "$ACTION" == "restart" ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
        echo "[RESTART] Killed existing tmux session '${TMUX_SESSION}'."
    fi
    ACTION="launch"
fi

if [[ "$ACTION" == "stop" ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
        echo "[STOP] Killed tmux session '${TMUX_SESSION}'."
    else
        echo "[STOP] No active tmux session '${TMUX_SESSION}' found."
    fi
    exit 0
fi

if [[ "$ACTION" == "attach" ]]; then
    exec tmux attach -t "$TMUX_SESSION"
fi

# ==============================================================================
# TMUX DETACHMENT: Spawn detached tmux session if not already inside worker loop
# ==============================================================================
if [[ "$INSIDE_WORKER" -eq 0 ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "[INFO] tmux session '${TMUX_SESSION}' is ALREADY running on $(hostname)!"
        print_status
        exit 0
    fi

    WORKER_CMD="cd ${REPO_ROOT} && DONT_START_RUN_AFTER_12_00=${dont_start_run_after_12_00} bash ${REPO_ROOT}/${SCRIPT_REL} --local --worker-loop"
    [[ "$dont_start_run_after_12_00" == "false" ]] && WORKER_CMD="${WORKER_CMD} --no-time-limit"
    [[ -n "$ONLY_EXP" ]] && WORKER_CMD="${WORKER_CMD} --run ${ONLY_EXP}"

    tmux new-session -d -s "$TMUX_SESSION" "$WORKER_CMD"
    sleep 1
    echo "[LAUNCHED] Detached tmux session '${TMUX_SESSION}' started on $(hostname)!"
    print_status
    echo ""
    echo "Useful commands (zero LLM tokens required):"
    echo "  Check status : ./_david/code/launch_rl_suite.sh --status"
    echo "  Watch live   : ./_david/code/launch_rl_suite.sh --attach   (detach with Ctrl+B, then D)"
    echo "  Restart/Rerun: ./_david/code/launch_rl_suite.sh --restart"
    echo "  Run one only : ./_david/code/launch_rl_suite.sh --run <EXP_ID>"
    echo "  Resume >12:00: ./_david/code/launch_rl_suite.sh --no-time-limit"
    echo "  Pull results : ./_david/code/launch_rl_suite.sh --pull"
    exit 0
fi

# ==============================================================================
# WORKER LOOP (Runs inside detached tmux session)
# ==============================================================================
log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$MASTER_LOG"
}

log_msg "================================================================================"
log_msg "Starting DrugEx 6-Arm SequenceRNN RL Suite on $(hostname)"
log_msg "dont_start_run_after_12_00 = ${dont_start_run_after_12_00} (cutoff window: ${CUTOFF_START_HOUR}:00..${CUTOFF_END_HOUR}:00)"
log_msg "Suite output directory     = ${SUITE_DIR}"
[[ -n "$ONLY_EXP" ]] && log_msg "Filter: running ONLY experiment '${ONLY_EXP}'"
log_msg "================================================================================"

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_id exp_desc exp_extra_args <<< "$entry"
    if [[ -n "$ONLY_EXP" && "$exp_id" != "$ONLY_EXP" ]]; then
        continue
    fi
    exp_dir="${SUITE_DIR}/${exp_id}"

    # 1. Skip already completed runs
    if is_run_complete "$exp_dir"; then
        log_msg "[SKIP] ${exp_id} is already COMPLETE. Moving to next run."
        continue
    fi

    # 2. Check 12:00 time gate BEFORE starting the next run
    if [[ "$dont_start_run_after_12_00" == "true" ]]; then
        current_hour=$(( 10#$(date +%H) ))
        current_hm="$(date +%H:%M)"
        if (( current_hour >= CUTOFF_START_HOUR && current_hour < CUTOFF_END_HOUR )); then
            log_msg "[TIME GATE] Current local time is ${current_hm} (within ${CUTOFF_START_HOUR}:00..${CUTOFF_END_HOUR}:00 cutoff) and dont_start_run_after_12_00=true."
            log_msg "[TIME GATE] Stopping cleanly BEFORE starting '${exp_id}' (${exp_desc})."
            log_msg "[TIME GATE] To resume from '${exp_id}', set dont_start_run_after_12_00=false or run:"
            log_msg "            ./_david/code/launch_rl_suite.sh --no-time-limit"
            exit 0
        fi
    fi

    # 3. Archive any incomplete/interrupted previous attempt for this experiment
    if [[ -d "$exp_dir" ]]; then
        backup_dir="${exp_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
        log_msg "[CLEANUP] Archiving incomplete directory ${exp_dir} -> ${backup_dir}"
        mv "$exp_dir" "$backup_dir"
    fi

    mkdir -p "$exp_dir"
    run_log="${exp_dir}/console.log"

    log_msg "--------------------------------------------------------------------------------"
    log_msg "[START] ${exp_id} — ${exp_desc}"
    log_msg "        Output : ${exp_dir}"
    log_msg "        Log    : ${run_log}"
    log_msg "--------------------------------------------------------------------------------"

    # Convert exp_extra_args into array
    read -r -a extra_arr <<< "$exp_extra_args"

    t0=$(date +%s)
    if "$PYTHON_BIN" -u "${REPO_ROOT}/_david/code/pipeline.py" \
        "${COMMON_ARGS[@]}" \
        "${extra_arr[@]}" \
        --output-dir "$exp_dir" \
        2>&1 | tee -a "$run_log" "$MASTER_LOG"; then
        t1=$(date +%s)
        dt_min=$(awk "BEGIN {printf \"%.1f\", ($t1 - $t0) / 60.0}")
        log_msg "[DONE] ${exp_id} completed successfully in ${dt_min} min."
    else
        rc=$?
        log_msg "[ERROR] ${exp_id} exited with code ${rc}. Stopping suite so error can be inspected."
        exit "$rc"
    fi
done

log_msg "================================================================================"
log_msg "[SUITE COMPLETE] All 6 prioritized SequenceRNN RL experiments finished!"
log_msg "Deliverables available in: ${SUITE_DIR}"
log_msg "================================================================================"
