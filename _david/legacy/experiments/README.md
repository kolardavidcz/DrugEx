# Legacy Multi-Study Experiment Runners

This directory archives early monolithic experiment and ablation suite runners:
- `hypotheses.py` / `run_hypotheses.py`: 3-hypothesis comparative evaluation runner.
- `ablations.py` / `run_threshold_ablation.py`: 6-study threshold and MORL ablation grid.
- `reserch_test/`: Early research scripts and runners.

---

## Why Are They Archived?

1. **Tight Coupling to Deprecated Heuristics**: These scripts were hardcoded to test `AntiCollapseSoftGateReward`, dynamic $\pm \sigma$ property envelopes, and preset strings that have been retired.
2. **Monolithic Architecture**: Rather than small, composable scripts, they attempted to manage complex study permutations, logging, and plotting in single long files.
3. **Replacement**: Replaced by our clean sequential 3-level Ground Model Ladder (Level 1 Baseline, Level 2 Exploration, Level 3 Positive-Feedback Property).
