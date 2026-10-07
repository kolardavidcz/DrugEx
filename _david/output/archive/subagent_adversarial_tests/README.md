# Archived Subagent Adversarial Stress-Tests

This directory preserves the temporary adversarial test harnesses authored by autonomous subagents during earlier milestones (R1–R5, M4). They are archived here to maintain 100% auditability and historical record while keeping the root `testing/` directory clean.

---

## 1. Archived Test Inventory

| Script | Purpose & Scope | Why It Was Superseded |
|---|---|---|
| [`adversarial_r1_stress_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r1_stress_test.py) | Aggressively probed `QueryConformerPreparer` and `DynamicPropertyEnvelope` across extreme small molecules, zero-TPSA aromatics, acyclic molecules, hydrophilic salts, and macrocycles. | Subsumed into [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py) (`TestRequirementR1QueryConformation`) and [`_david/tests.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/tests.py). |
| [`adversarial_r2_rocs_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r2_rocs_test.py) | Tested `RDKitROCSScorer` (`oe_mills_dean`, `TanimotoCombo`), Mills-Dean 6th hydrophobe SMARTS, and RDKit issue #8513 pose invariance. | Subsumed into [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py) (`TestRequirementR2Native3DROCSScorer`). |
| [`adversarial_r3_reward_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r3_reward_test.py) | Probed `AntiCollapseSoftGateReward` against Pareto cheaters, early RL non-vanishing gradients, and the single SA synthesizability floor ($\text{SA} \le 5.5$). | Subsumed into [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py) (`TestRequirementR3AntiCollapseRewardScheme`) and [`_david/tests.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/tests.py). |
| [`adversarial_r4_recovery_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r4_recovery_test.py) | Stress-tested `PeakRecoveryMonitor`, early stopping triggers (`patience=10`), and model weight checkpoint restoration. | Subsumed into [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py) (`TestRequirementR4GraphTransformerTraining`). |
| [`adversarial_r5_rocs_scaffolds.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r5_rocs_scaffolds.py) | Evaluated Bemis-Murcko framework diversity across synthetic candidate batches and Butina clustering. | Subsumed into [`_david/code/result_analyzer.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/code/result_analyzer.py) and [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py). |
| [`adversarial_r5_vun_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_r5_vun_test.py) | Evaluated VUN metrics (Validity $\ge 80\%$, Uniqueness $\ge 80\%$, Novelty against Papyrus) on sampled candidate pools. | Subsumed into [`_david/code/result_analyzer.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/code/result_analyzer.py) and [`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py). |
| [`adversarial_m4_remote_execution_test.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/subagent_adversarial_tests/adversarial_m4_remote_execution_test.py) | Probe harness verifying remote SSH execution over `dichtator` Blackwell GPU 0. | Replaced by active direct execution through [`_david/run.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/run.py) and live monitor [`_david/code/monitor.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/code/monitor.py). |

---

## 2. Authoritative Test Suite

The current authoritative test suite consists of:
1. **[`_david/tests.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/tests.py)**: 16 active unit tests for pipeline components, rewards, dynamic envelope, and fallback monitoring.
2. **[`testing/test_lbdd_requirements.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/test_lbdd_requirements.py)**: 60 authoritative tests covering requirements R1 through R5.
3. **[`testing/adversarial_m4_empirical_verification.py`](file:///home/kolar/build_projects/drugex_edu/repo/testing/adversarial_m4_empirical_verification.py)**: 14 independent verification tests evaluating the empirical leads from `_david/output/m4_empirical`.
