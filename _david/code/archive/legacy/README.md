# Archived `_david/legacy/` Package (retired 2026-11-03)

Frozen copy of the pre-refactor `_david/legacy/` tree, kept verbatim (with its
own `__init__.py` and `tests.py`) purely so that old notes and drafts that cite
these modules still resolve.

**Nothing imports this.** The RL pipeline, the conformer suite and the
MetaCentrum runner never referenced these modules, which is exactly why they
moved here instead of being silently deleted:

| Module | What it tried to do | Why it was retired |
|---|---|---|
| `consensus_alignment/fetch_target_actives.py` | scrape target actives for cross-target consensus alignment | superseded by the ChEMBL/PDB mining path used in `_david/output/conformer/04_molecule_preparation/` |
| `dynamic_envelopes/envelopes.py` | adapt desirability thresholds over epochs | the thesis settled on fixed desirability bounds plus Pareto crowding (`config.reward_scheme`) |
| `morl_scalarizations/scalarizations.py` | extra multi-objective scalarisation functions | DrugEx's `WeightedSum` / `ParetoCrowdingDistance` / `HarringtonDesirabilityReward` cover the evaluated schemes |
| `consult_metacentrum_runs.py` | one-off DeepSeek/GLM consultation prompt for the H1–H4 run matrix | its prompt hard-codes historical defaults (1,000 evaluation samples, Tversky variants as CLI choices); the current gold standard lives in `_david/code/config.py` |
| `tests.py` | smoke tests for the above | never discovered by the maintained suite |

If one of these ideas is revived, promote it into `_david/code/` with tests and a
dated entry in `_david/code/conformer/results.md` rather than importing it from
`archive/`.
