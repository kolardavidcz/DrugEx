# Code-review triage — claims 6, 9, 10 (2026-11-06)

Scope: the three findings in the external review that touch ROCS colour handling and
post-hoc lead ranking. Each claim was checked against the code, against git history,
and — where the claim is numerical — against a direct RDKit measurement.

Verdict summary: **one hallucination, one real but diagnostic-only, one real and
materially misleading in the analysis layer.** None of them touches the training
reward of the five MetaCentrum arms currently running.

---

## Claim 9 — `Tanimoto_to_Ref` compares explicit-H reference with implicit-H candidates

**Verdict: hallucination. No action.**

- `Tanimoto_to_Ref` does not exist anywhere in the repository or in the git history
  (`git log -S "Tanimoto_to_Ref"` is empty; the only `Tanimoto*` schema constant is
  `Tanimoto_to_Seed`).
- The cited `DrugExEnvironment.__init__` / `getScores` with `self.reference_mol` does
  not exist. `DrugExEnvironment` is in `drugex/training/environment.py` and has no
  `reference_mol` attribute. The cited `_david/code/training/environment.py#L140` is
  `resolve_reward_scheme`, and there is no `reference_mol` in that file at all.
- The quoted RDKit "verification" is also wrong: self-Tanimoto of a fingerprint
  against itself is 1.0 **by definition**, for both the implicit-H and the explicit-H
  graph. The real number is the *cross* similarity between the two representations:
  **0.30**, not 0.274.

There is no corrupt similarity column to repair. If we ever add a 2D
`Tanimoto_to_Seed` column, the rule is to compute both fingerprints from
`RemoveHs(...)`/parsed SMILES so the graphs match.

---

## Claim 6 — `score_pair` / `align_to_reference` skip `_annotate_color_features`

**Verdict: confirmed. Severity P2 — diagnostics only, not training.**

Mechanism is exactly as described: `rdShapeAlign.AlignMol(useColors=True)` reads the
`PUBCHEM_PHARMACOPHORE_FEATURES` property; without it RDKit falls back to its
5-pattern fdef typing (no hydrophobe type), while production training/standard
scoring annotates both molecules with the 6-type OpenEye ImplicitMillsDean scheme.

**Why training is unaffected.** `create_rocs_scorer` passes `color_ff="oe_mills_dean"`,
and core `_score_single_reference` annotates both reference and query before aligning.
Every RL reward and every `standard_eval` number therefore already uses the correct
scheme. What was wrong is the auxiliary path: `diverse_order` (K=4 reference
selection), `_shape_color_for_smiles` (figures 03/08 shape-vs-colour decomposition)
and `align_to_reference` (figure 10 superposition and `top_leads_aligned_to_seed.sdf`).

**Measured impact** (RDKit, this machine):

| comparison | mean \|Δcombo\| | max \|Δcombo\| |
|---|---|---|
| EGCG 20-conformer pool, pairwise, annotated vs raw | 0.0086 | 0.0188 |
| 20 prior molecules vs EGCG reference | 0.0091 | 0.0333 |

- `diverse_order(prefix=4)` selected the **identical four conformers** with and
  without annotation, so the K=4 reference bank does not change.
- Rank correlation of the prior set is 0.973 — the ordering is essentially preserved,
  the level shifts by ~0.01.

So the figures were drawn on a colour channel ~0.01 combo away from the channel the
run optimised. That is not enough to explain any run-level conclusion, but it is a
real inconsistency and cheap to remove.

**Fix (landed):** `score_pair` and `align_to_reference` now take
`color_ff="oe_mills_dean"` and annotate both molecules before aligning; passing
`"default"` reproduces the legacy fallback deliberately. Contract test
`_david/code/tests/test_rocs_color_contract.py` asserts that `score_pair` reproduces
`_score_single_reference(..., "oe_mills_dean")` to 1e-6 on a flexible probe, so the
two paths cannot drift apart again.

---

## Claim 10 — post-hoc lead ranking hardcodes `+ 1.0 * QED`

**Verdict: confirmed. Severity P1 for anything that reads a "top lead" list;
no effect on training.**

Two separate defects:

1. **Ranking contradicts the objective.** `include_qed=False` is the EGCG default
   (QED 0.21 on the target, and optimising it fights the polyphenol scaffold), yet the
   lead index was `ROCS + QED − 0.10·SA` in `ResultAnalyzer.analyze`,
   `epoch_molecule_table` and `dashboard._candidate_table`. On `ctrl_seed42`
   (4 165 valid molecules): std(ROCS) 0.133, std(QED) 0.200, std(−0.1·SA) 0.076, so
   QED carries ~1.5× the leverage of the actual objective; Spearman(index, QED) 0.90
   versus Spearman(index, ROCS) 0.36; the top-100 shares **9 molecules** with the
   top-100 by ROCS, and the top-10 mean ROCS is 0.789 instead of 0.935.
2. **File integrity.** `df_leads = _valid_subset(df_sorted)` was computed and then
   `df_sorted` (all 5 000 rows, 1 330 unparsable) was written to
   `top_prioritized_leads.tsv`. The top rows are clean because NaN sorts last, so no
   published top-N table is wrong — but the file is not what its name says.

**Fix (landed):**

- New shared module `_david/code/analysis/ranking.py` with
  `lead_ranking_weights(run_config)` and `lead_rank_score(frame, run_config)`.
  With `include_qed=False` the index is `ROCS − 0.10·Raw_SA` (training objective plus
  an SA tie-breaker); with `include_qed=True` it is the old `ROCS + QED − 0.10·SA`.
  A missing/unreadable manifest defaults to `include_qed=False`, i.e. the EGCG default.
- `analyze()` now loads the manifest before ranking, writes **`df_leads`** (valid only)
  to `top_prioritized_leads.tsv`.
- The dashboard pre-ranks by the run's own objective before paying for SA, then ranks
  with the same shared index.
- Tests assert the weight contract and that a high-QED/low-ROCS molecule cannot
  outrank a high-ROCS one when QED is disabled.

**Consequence for existing artifacts:** every run's `top_prioritized_leads.tsv`,
figures 05/10/12 and the dashboard "best molecules" table change. The five MetaCentrum
arms in flight are scored by code that predates this fix only on the *analysis* side;
their arms are untouched, so their figures just need a `--force-render` regeneration
after `pull`. No retraining is needed.

---

## What this does and does not explain

These three findings are analysis-layer defects. They do not touch the reward, the
conformer protocol or the reference geometry, and they cannot explain the
+0.11–0.14 prior→after gap between the supervisor's arm and ours. The gap remains a
policy/calibration question: our fixed `(0.35, 0.50)` thresholds saturate
(`desired_ratio` 0.86 at epoch 1) where the supervisor's p85 calibrates to 0.14.

The one thing worth watching: the supervisor's reported "top hits" are presumably
ranked only by ROCS, while our report's lead tables were ranked with QED. After this
fix both are ROCS-anchored, so the two arms become comparable at the lead level as
well as at the median.
