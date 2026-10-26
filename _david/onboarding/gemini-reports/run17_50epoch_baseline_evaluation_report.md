# Run 17: verified evaluation record

Historical experiment: run17_baseline_50ep. The original previous-agent report is preserved in Git at fb1d1e4. Reviewed against saved remote CSVs during onboarding on 2026-10-26; no training was rerun.

## Sources

- _david/output/run17_baseline_50ep/models/SequenceRNN_reinforced_fit.tsv: 50 recorded epochs.
- raw_data/baseline_raw.csv.gz and raw_data/reinforced_raw.csv.gz: 1,000 rows each.
- summary.md and comparison_scorecard.csv: generated reports affected by the scaling issue below.
- Intended launcher configuration in code/run_overnight_matrix.py: epsilon=0.20, beta=0.0, 50 epochs. That current file alone does not prove every historical parameter.

## Directly checked observations

- Baseline valid rows: 955/1,000; reinforced valid rows: 883/1,000.
- Stored ROCS-column mean: baseline 0.328771; reinforced 0.303615.
- Stored ROCS-column median: baseline 0.348048; reinforced 0.341882.
- Stored ROCS-column maxima: baseline 0.496405; reinforced 0.498017.
- Saved Desired entries are zero in both evaluation arms, and recorded training desired_ratio values remained zero.

These values describe the stored column with all rows included, not a claimed raw physical TanimotoCombo scale. Filtered means, medians, and training-validation rates are different quantities and must be labelled separately.

## Reporting limitations

ResultAnalyzer.analyze() converts ROCS and build_diagnostic_scorecard() can convert it again when the observed maximum stays at most one. The saved scorecard medians, 1.392 and 1.368, match approximately four times the stored medians. Absolute scorecard values and affected figures need correction before thesis use.

The previous handoff's 99.80% validity and increased mean ROCS claims are not supported by these evaluation files. Do not quote them as results.

The study does not by itself establish a causal benefit of an exploration setting. Do not infer improvement from selected leads, automatic PASS labels, or desired-ratio thresholds alone. Runtime claims in earlier notes conflict and were not reconstructed here.
