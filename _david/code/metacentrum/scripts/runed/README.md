# `runed/` — frozen PBS scripts (provenance only)

These files are the exact PBS scripts that produced historical MetaCentrum results. They are
kept for reproducibility and for the record, **not** for reuse.

Do not run them. They predate the current contract and each one hard-codes things that are
now wrong:

- storage detection that scans `/storage/brno2/brno2` and `/storage/brno2/home` at runtime;
- per-script `#PBS` resources, several with `spec=10.0` and even `ngpus=1` on a CPU queue;
- their own copy of the pipeline bootstrap, so they can drift from `job_template.sh`;
- argument lists that name old code paths and old reference ensembles.

Use [`../../RUNBOOK.md`](../../RUNBOOK.md): one commit, one `submit.py` invocation, one
`job_template.sh`.

## What each script was

| Script | Experiment |
| :--- | :--- |
| `h1_without_qed_pareto.sh` | Sprint arm H1 — supervisor replication champion: K=1, C=30 unminimised, ε=0.05, TanimotoCombo, no QED, Pareto reward. |
| `h2_without_qed_harrington.sh` | H2 — same regime, Harrington reward, no QED. |
| `h3_with_qed_pareto.sh` | H3 — QED shrinkage ablation, Pareto reward. |
| `h4_with_qed_harrington.sh` | H4 — modernised Run 01 baseline (QED + Harrington). |
| `h7_tversky_scaffold_hop.sh` | H7 — asymmetric Tversky scoring for scaffold hopping. |
| `h8_beta_baseline_penalization.sh` | H8 — baseline-policy penalisation, β=0.1. |
| `h9_dynamic_epsilon_decay.sh` | H9 — cosine ε decay instead of constant ε. |
| `egcg_k4_c20_candidates_unminimized.sh` | Candidate conformers C=20, ETKDGv3, unminimised. |
| `egcg_k4_c20_candidates_minimized_gold.sh` | C=20 minimised with MMFF94s ε=78.4 (gold elbow). |
| `egcg_k4_c20_min_gold_gmean.sh` | The minimised gold elbow, checkpointed by `avg_gmean`. |
| `egcg_k4_c20_raw_tanimoto_gmean_5k.sh` | Gold-standard run: raw TanimotoCombo, gmean, 5000-molecule evaluation arm. |
| `submit_all_hypotheses.sh` | Batch submitter for the H1–H4 sprint (the only file that submitted others). |
| `submit_run_h6_ensemble.sh` | H6 — multi-conformer reference ensemble. |
| `submit_supervisor_eps01.sh` | Supervisor tutorial replication at ε=0.10. |

Each runner ends with a comment naming the PBS job id that executed it
(`METACENTRUM_<jobid>`), which is how these scripts are cross-referenced with the pulled PBS
logs in `_david/output/_metacentrum_jobs/pbs_logs/`.

## How to reproduce one of these runs today

Read the script's header, translate its intent into pipeline flags, and submit through the
current template. Example, the gold-standard 5k run:

```bash
.venv/bin/python -m _david.code.metacentrum.submit submit \
    --name egcg_raw_5k_replay \
    -- "--criteria gmean --eval-samples 5000 --conformers 20 --minimize-candidates"
```

The point of the exercise is the same science, not the same file.
