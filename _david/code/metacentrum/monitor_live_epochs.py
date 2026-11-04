#!/usr/bin/env python3
"""
Real-time MetaCentrum DrugEx RL Epoch & Run Monitor.
Scans active PBS Pro jobs for 'kolardavidcz', identifies the worker node and scratch directory,
and prints live training progress (last epochs, rewards, and execution logs).
"""

import re
import subprocess
import sys

USER = "kolardavidcz"

def run_cmd(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return res.stdout
    except Exception:
        return ""

def main():
    qstat_out = run_cmd(["qstat", "-u", USER])
    if not qstat_out:
        print(f"No active or queued jobs found for {USER}.")
        return

    running = []
    queued = []

    for line in qstat_out.splitlines():
        parts = line.split()
        if len(parts) >= 10:
            jid = parts[0]
            jname = parts[3]
            state = parts[9]
            elap = parts[10] if len(parts) > 10 else "--"
            if state == "R":
                running.append((jid, jname, elap))
            elif state in ("Q", "H", "W"):
                queued.append((jid, jname, state))

    print("=" * 70)
    print(" METACENTRUM REAL-TIME RUN MONITOR")
    print("=" * 70)

    if queued:
        print(f"\n[Queued Jobs] ({len(queued)} waiting for GPU allocation):")
        for jid, jname, st in queued:
            print(f"  • {jid.split('.')[0]:<12} | {jname:<28} | State: {st}")

    if not running:
        print("\n[Running Jobs]: None currently active.")
        print("=" * 70)
        return

    print(f"\n[Active Running Jobs] ({len(running)} executing on cluster nodes):")
    for jid, jname, elap in running:
        short_id = jid.split('.')[0]
        q_f = run_cmd(["qstat", "-f", short_id])
        m_host = re.search(r"exec_host\s*=\s*([^/\n]+)", q_f)
        host = m_host.group(1).strip() if m_host else "unknown"

        print(f"\n>>> Job {short_id} ({jname}) on node: {host} (Elapsed: {elap})")

        # Query worker node via SSH
        scratch_base = f"/scratch*/kolardavidcz/job_{short_id}*"
        tsv_path = f"{scratch_base}/output/models/SequenceRNN_reinforced_fit.tsv"
        log_path = f"{scratch_base}/output/execution.log"

        tsv_check = run_cmd(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host, f"tail -n 6 {tsv_path} 2>/dev/null"])
        if tsv_check and tsv_check.strip():
            print("  --- Latest Training Epochs (fit.tsv) ---")
            for tsv_line in tsv_check.strip().splitlines():
                print(f"  {tsv_line}")
        else:
            log_tail = run_cmd(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host, f"grep -v RuntimeWarning {log_path} 2>/dev/null | tail -n 8"])
            if log_tail and log_tail.strip():
                print("  --- Live Execution Log ---")
                for l_line in log_tail.strip().splitlines():
                    print(f"  {l_line}")
            else:
                print("  (Job is initializing / preparing environment on scratch node)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
