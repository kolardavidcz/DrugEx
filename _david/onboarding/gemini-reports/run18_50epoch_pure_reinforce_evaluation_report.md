# Run 18: verified evaluation record

Historical experiment: run18_epsilon_zero_50ep. The original previous-agent report is preserved in Git at fb1d1e4. Reviewed against saved remote CSVs during onboarding on 2026-10-26; no training was rerun.

## Sources

- _david/output/run18_epsilon_zero_50ep/models/SequenceRNN_reinforced_fit.tsv: 50 recorded epochs.
- raw_data/baseline_raw.csv.gz and raw_data/reinforced_raw.csv.gz: 1,000 rows each.
- summary.md and comparison_scorecard.csv: generated reports affected by the scaling issue below.
- Intended current launcher configuration: epsilon=0.00, beta=0.0. Verify historical provenance before asserting every other setting matched Run 17.

## Directly checked observations

- Baseline valid rows: 951/1,000; reinforced valid rows: 845/1,000.
- Stored ROCS-column mean: baseline 0.326311; reinforced 0.276163.
- Stored ROCS-column median: baseline 0.347394; reinforced 0.323566.
- Stored ROCS-column maxima: baseline 0.499356; reinforced 0.459876.
- Saved Desired entries are zero in both evaluation arms, and training desired_ratio remained zero.

These are all-row stored-column statistics, not an independently established raw physical score scale. They do not support the handoff's 100% validity or increased mean ROCS claims.

## Reporting and continuation limits

The same nested ROCS conversion as Run 17 affects scorecards and figures. Saved scorecard medians 1.390 and 1.294 match approximately four times the stored medians. Do not use those absolute values as calibrated physical results.

At SSH inspection, the subsequent cosine-decay Run 19 had 18 recorded epochs and no completed reinforced evaluation, with no recognised training process active. Old statements that it was currently training at epoch 2-3 were stale.

Selected top leads do not establish whole-distribution improvement or causal exploration effects. Reconcile score scale, filtering, invalid/duplicate handling, code revision, and evaluation protocol before comparisons or production decisions.
