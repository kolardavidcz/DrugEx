# Conformer study: verified record and limitations

Reviewed during Codex onboarding on 2026-10-26. The original previous-agent report is preserved in Git at fb1d1e4. Its interpretations are not treated as established physical conclusions.

## Evidence

Source: _david/output/conformer_study/conformer_convergence_scorecard.csv. Implementation baseline: 9261af8, especially code/conformer/orchestrator.py, generator.py, physics.py, and analyzer.py.

The saved scorecard's rounded values are:

- MMFF94s vacuum: 2 reference basins; K1/Q1 score 0.7808; requested K3/Q30 score 1.9712.
- UFF vacuum: 2 basins; 1.9963 and 1.9988.
- MMFF94s pocket-labelled variant: 3 basins; 1.9865 and 1.9865.
- MMFF94s water-labelled variant: 3 basins; 1.9988 and 1.9988.
- MMFF94s distance-dependent variant: 2 basins; 0.8056 and 1.8174; recorded maximum 1.9707.

These are saved observations, not independently rerun results. Scores are TanimotoCombo on the study's [0,2] scale.

## Interpretation limits

Reference and probe pools use EGCG with seeds 42 and 9999. The analyzer clips requested counts to retained conformers. Requested K values are therefore not necessarily actual evaluated counts; repeated grid entries need not represent additional sampling.

The suite changes both dielectric settings and energy-window treatment. The pocket/water labels describe configured variants, not validated representations of every binding pocket or solvent. The original report's universal dielectric and physiological explanations require primary-source justification.

High self-similarity does not prove performance on chemically diverse candidates. Duration_sec is variant study time, not per-candidate scoring latency for each K. No production K or expected K=10 optimum has been established.

The standalone pipeline differs from the RL conformer path. Do not claim these variants were deployed during Run 17/18 or that timeouts are disabled end to end.

## Artifacts

Study artifacts live in _david/output/conformer_study/, including the scorecard, convergence figures, interactive HTML, and models_3d/. These are generated/ignored outputs and may not exist in a fresh clone. Inspect actual filenames before linking. The unit suite has 4 conformer tests and 6 pipeline tests; the old claim of 10 tests in test_conformer.py was inaccurate.
