# What changed, against the previous pipeline

Written 2026-11-10, at `feature/david-mmff94s-pipeline`. The comparison is against
`~/WITH_BUGS_drugex/_david/code/` (the tree this project was restarted from), and it is about
**method, not files**: a line-by-line diff is meaningless because `_david/code/` was written from
scratch rather than edited. Every claim below names the file and line on both sides so it can be
checked rather than believed.

| | previous | now |
|---|---:|---:|
| Python files | 118 | 20 |
| lines of Python | 42,582 | 5,413 |
| of which the pipeline itself | 4,882 | 2,307 |
| configuration fields | 55 | 35 |

The rest of the new tree is 1,285 lines of figures and reporting, 910 of cluster tooling and 898
of tests. The old tree additionally held `archive/`, `conformer/`, `run_audit/`, `to_do/` and a
4,532-line `analysis/visualization.py`, none of which has an equivalent here.

---

## 1. The reward is DrugEx's own, and nothing weights it

**Previously:** a bespoke reward scheme, `HarringtonDesirabilityReward`
(`training/environment.py:22`). Each objective was mapped through a two-parameter Gompertz
desirability fitted from hard-coded anchor points — ROCS `(0.40, 0.05, 1.20, 0.80)`, SA
`(0.20, 0.05, 0.80, 0.90)`, QED `(0.30, 0.05, 0.70, 0.90)` — and combined as a **weighted
geometric mean** over a configured `weights` vector (`config.py:175`). The class also zeroed the
reward of any SMILES matching `([A-Za-z])\1{7,}`, or longer than 20 characters with more than 50 %
nitrogen (`training/environment.py:141-146`).

**Now:** `scorers/objectives.py:build_environment` hands the scorers to one of DrugEx's own
reward schemes and nothing else. `PRCD` (`ParetoCrowdingDistance`) is the default and is DrugEx's
own default; `PRTD` and `WS` are selectable through `config.scheme`. There are no weights, no
desirability curves, no anchor constants and no string-pattern penalty anywhere in the reward
path.

**Why it matters.** Four numbers per objective, a weight vector, and a regex are four places where
a result can come from the scoring rule rather than from the chemistry, and none of them appeared
in any output. Pareto ranking is also scale-free, which is what lets the ROCS objective stay on its
native `[0, 2]` TanimotoCombo instead of being rescaled to match the others.

The one exception is deliberate and is stated wherever it applies: DrugEx's `WeightedSum` builds
its weights from the thresholds (`drugex/training/rewards.py:304`), so a `WS` run — four of the
148 planned — does let the thresholds reach the reward. `report.md` says so in those runs, a test
fails if that prose ever reverts, and `DECISIONS.md` carries the dated note.

## 2. Two failure values that scored as "best"

Both are the same mistake in two places: a quantity where *lower is better* was given `0.0` when it
could not be computed, and `0.0` is off-scale **below** the best real value.

**Synthetic accessibility.** Native `Property.getScores` initialised its array with
`np.zeros` and left that value in place when RDKit raised
(`drugex/training/scorers/properties.py:46` in the old tree). Raw SA is defined on `[1, 10]`, lower
better, and the modifier is `ClippedScore(lower_x=5, upper_x=3)` — descending. A molecule that
could not be scored therefore arrived as raw `0`, which the modifier mapped to **1.0, the maximum
reward**: an unscorable molecule outranked every synthesizable one. Now `FAILURE_VALUES = {'SA':
10.0}` puts the failure at SA's own worst value, and a test
(`tests.py:FailureValues`) pins both halves — that ethanol beats an unscorable
molecule, and that the modifier maps the failure to `0.0`.

**Conformer energy.** `_record_energy` returned **`0.0` when a conformer had no recorded energy**
(`reference.py:41-47`), and the reference bank sorted ascending on that value
(`reference.py:98`). MMFF energies for EGCG are around +90 kcal/mol, so any conformer whose
minimization had failed sorted *first* — it became the lowest-energy pose, the seed of the bank
and the anchor of the diversity selection. Now `single_point_energy` returns `math.inf` when MMFF
cannot parameterize a molecule (`scorers/conformers.py`), so an unscorable conformer sorts last,
and a conformer whose relaxation produced a non-finite energy is dropped and **counted** rather
than kept.

## 3. The reference bank is chosen by geometry, not by energy rank

**Previously:** Butina RMSD pruning at 0.35 Å, grouped by stereoisomer, each group sorted by
energy, the lowest-energy member seeding a farthest-first pass, inside a 100 kcal/mol energy
window (`reference.py:89-98`, `reference.py:185`).

**Now:** every distinct conformer ETKDG produced is a candidate; the global minimum anchors the
bank and each further pose is the one whose nearest already-kept neighbour is farthest away
(farthest-point selection on heavy-atom RMSD). Energy anchors and breaks ties; it never filters.
The published reason is in `DECISIONS.md`: energy-ranked selection recovered a crystal pose for
57.6 % of ligands against 72.7 % for RMSD-diversity selection, and an absolute kcal/mol window
admitted 2 unminimised poses against 4 minimised ones — a bank-size confound in the one comparison
this project exists to make.

## 4. Minimisation is one switch for both sides

**Previously:** the force-field profile was shared, but `minimize_candidates` toggled the candidate
side alone (`config.py:202`), so a configuration could minimise the reference and not the
candidates. The reference path also carried a **UFF fallback** when MMFF had no parameters
(`reference.py:305-325`), at `maxIters=2000` and `mmffVariant="MMFF94"`.

**Now:** `minimize` is a single switch applied to the reference bank and to every candidate
conformer through the same generator, so the relaxed and unrelaxed arms differ in exactly one
respect. There is no fallback force field at all: relaxing part of a library under different
physics makes two molecules' scores incomparable for a reason nothing records. MMFF94**s** is
fixed in code, not configurable — the arms are "relaxed" and "not relaxed", never "relaxed under a
different force field" — and a molecule MMFF cannot parameterize keeps its embedded geometry and
is tallied as `unparameterized`.

## 5. The training loop has one criterion and one checkpoint

**Previously:** native `Explorer.fit` with a single `--criteria`
(`training/pipeline.py:604-616`), writing `SequenceRNN_reinforced.pkg` (best by that one
criterion) and `SequenceRNN_reinforced_final_epoch.pkg`. `reload_interval` was configurable and
disabled by passing `epochs + 1`.

**Now:** `training.py`'s `StopCheckpointExplorer` trains on `desired_ratio` and writes exactly one
checkpoint, `desired_ratio_at_stop.pkg`, from the live policy at the epoch training ended — either
when the patience window expired or when the epoch cap arrived. There is no best-epoch checkpoint
and no second criterion: a run delivers the policy it actually ended on, and saving a different
state invites reading a score that none of the delivered molecules produced. The summary records
the best value and its epoch for diagnosis, the peak seen, whether patience fired, the value at
stop, and the valid/unique ratios of the population that produced the checkpoint. Periodic
best-state reload is not implemented at all: copying a past checkpoint back over the live agent
would mean the policy a run ends on is not the policy it trained into. The old dual-checkpoint
explorer is archived unchanged in `archive/explorer.py`, so runs it produced remain readable.

## 6. Thresholds, and what they are allowed to touch

The *values* are largely unchanged — p85 of the prior, SA clipped 5 → 3, raw TanimotoCombo — but
their reach is now explicit: thresholds produce `Desired` → `desired_ratio` → checkpointing and
early stopping, and nothing else. A test pins that the Pareto schemes ignore the thresholds they
are handed; another pins that `WS` does not. SA and QED additionally take **the reference
molecule's own score** as their bar (`threshold_from_reference`), because `ClippedScore(5 → 3)`
saturates about 68 % of a drug-like prior at exactly 1.0, where "p85" is not a percentile but a
hard gate at the ceiling — the report now says which of the two every threshold is.

## 7. Diagnostics the previous pipeline computed and discarded

- **Shape and colour separately.** The old RL path hard-coded `opt_param=0.5` (`rocs.py:93`,
  `rocs.py:146`) and reported one number. The pose search computes both components anyway, so
  `scorers/rocs.py` records the winning pose's `ShapeTanimoto` and `ColorTanimoto` for free, as
  diagnostics that never become objectives (`getKey` returns exactly one key, and a test pins the
  count). The alignment objective is now a setting, separate from what is reported.
- **Raw SA.** Arms carried only the modified column, where two thirds of a drug-like prior sits at
  exactly 1.0, so a run could not show whether training moved synthesizability at all. `SA_raw` is
  now a column in every arm.
- **What relaxation lost.** Every relaxation outcome is counted by reason and totalled in the
  manifest, because "both arms scored the same molecules" is a claim that needs a number.
- **Every scored training molecule.** The policy-gradient batch is 10x larger than the evaluation
  arm and was thrown away each epoch; it is now appended to `training_samples.tsv`.

## 8. Deliberately absent

`schema.py` (52 lines of string constants: plain names are used instead), `prior_dataset.py`
(794 lines of cross-run prior storage), `run_audit/`, `to_do/`, `conformer/` studies, the
`WholeCallScorer` and its persistent worker pool, and every "might be useful later" parameter. A
setting that is not exercised by a planned run is a behaviour nobody has tested, and the old tree's
own failures were all in paths that looked harmless.

## 9. Deliberately identical

MMFF94s at dielectric 78.4 with the constant model, `protonate=False`, p85, SA 5 → 3, raw
TanimotoCombo on `[0, 2]`, `reference_attempts=200`, ETKDG's own 0.5 Å pruning, and DrugEx's own
conformer generator, stereoisomer enumeration, tautomer canonicalization and heavy-atom /
rotatable-bond filters. Where the old settings were right, they were kept; `DECISIONS.md` records
the evidence for each.

## 10. The native tree

The old tree now runs against clean upstream `drugex/` (verified: `diff -rq` reports only
`properties.py`), so the "an agent edited native code and the weights stopped updating" failure is
no longer visible in it. This tree makes **exactly one** native change, `FAILURE_VALUES` in
`drugex/training/scorers/properties.py` (§2), which is documented at the definition, in the commit
message, and pinned by a test.

---

### How this is meant to stay true

`tests.py` pins behaviours that were argued about rather than lines of code, so reverting one
says which argument is being reopened. `DECISIONS.md` carries a dated note, with evidence and what
would overturn it, for every choice that moves a number. Every run writes `config.json`,
`manifest.json` and a `report.md` that states its own thresholds, their source, and the regime its
reward is actually in — so a reader does not have to infer any of it from the code.
