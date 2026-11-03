# 02 · Color-shape scoring at the production default

**Question.** Is the production scorer — symmetric TanimotoCombo
(`α = β = 1`, `0.5 · shape + 0.5 · color`), `K=4` references, `C=20` candidates,
`I=4`, MMFF94s ε=78.4 — the right default, and what specifically breaks if a
dial is moved?

**Figure.** `tversky_color_162grid_selectivity.png`
**Table.** `default_scorer_neighbourhood.csv`
**Kernel.** `scoring/rocs.py` (production wrapper) and `scoring/tversky.py`
(asymmetric overlap mathematics).

---

## What the 162 cells are

Every combination of

- 18 asymmetric overlap settings: `α + β = 1` (Dice scale, 9 cells) and
  `α + β = 2` (Tanimoto scale, 9 cells), over `α ∈ [0.1, 0.9]`
- 9 shape/colour weightings: `w_S ∈ {0.1, …, 0.9}`, `w_C = 1 − w_S`

scored on every ordered molecule pair of the 121-drug grid, with a curated
hard-decoy panel on top (small fragments, bloated "monsters", and shape-matched
"trojans" that share the reference pharmacophore).

The overlap is Tversky on the RDKit Gaussian-volume scale:

```
Tversky(α, β) = overlap / (α · V_ref + β · V_cand − overlap)
```

`α` drains *reference* volume, `β` drains *candidate* volume. So:

| Setting | Effect | Failure mode it invites |
|---|---|---|
| `α` small (`α+β=2`) | reference volume barely penalized | small fragments score high |
| `β` small (`α+β=2`) | candidate excess volume barely penalized | bloated blobs score high |
| `w_C` high | colour dominates | shape-blind pharmacophore mimics |
| `w_S` high | shape dominates | rigid volume copies with wrong chemistry |

That last row is why the audit is against *decoys*, not against self-recovery
alone: every one of these dials raises self-recovery too.

---

## Panels

| Panel | Question |
|---|---|
| A | Asymmetric Tversky traps across cross-drug decoys |
| B | Worst-decoy P95 and the fraction of decoy pairs above the RL threshold (0.35) |
| C | Selectivity margin (worst-decoy P95 vs self-recovery) |
| D | EGCG stress test: active sub-scaffolds vs fragment / steric / bloated decoys |

---

## The default cell, with its neighbours

From `default_scorer_neighbourhood.csv` (self-recovery on the 121-drug grid,
worst-decoy P95 over the hard-decoy panel):

| Role | `(α, β)` | `w_S / w_C` | Self mean | Self P10 | Worst decoy P95 | Margin | Decoys ≥ 0.35 |
|---|---|---|---:|---:|---:|---:|---:|
| **default** | (1.0, 1.0) | 0.5 / 0.5 | 0.817 | 0.613 | 0.474 | 0.343 | 0.996 |
| lighter shape | (1.0, 1.0) | 0.4 / 0.6 | 0.801 | 0.595 | 0.414 | 0.387 | 0.556 |
| heavier shape | (1.0, 1.0) | 0.6 / 0.4 | 0.835 | 0.651 | 0.542 | 0.292 | 1.000 |
| sub-shape | (0.8, 1.2) | 0.5 / 0.5 | 0.816 | 0.612 | 0.477 | 0.338 | 0.991 |
| super-shape | (1.2, 0.8) | 0.5 / 0.5 | 0.819 | 0.613 | 0.473 | 0.346 | 0.993 |
| colour-heavy | (1.0, 1.0) | 0.2 / 0.8 | 0.770 | 0.515 | 0.303 | **0.467** | **0.006** |
| shape-heavy | (1.0, 1.0) | 0.8 / 0.2 | 0.872 | 0.727 | 0.683 | 0.189 | 1.000 |

Read the last two rows as the two ends of one trade, not as candidates:

- **Shape-heavy** buys +0.055 self-recovery and loses the decoy discrimination
  entirely: every hard decoy clears the 0.35 threshold.
- **Colour-heavy** drops 4.7 points of self-recovery and drops false positives
  from 99.6 % to 0.6 % — but 0.2 shape weight would let a shape-blind fragment
  through, and the EGCG stress panel (D) shows exactly which decoy class returns
  at each weight.

The default is the interior point where *both* decoy classes stay below the
threshold for most settings, which is why the recommendation is "keep
TanimotoCombo" rather than "improve it".

---

## What the full grid says

- Across all 162 cells, the **median** fraction of decoy pairs ≥ 0.35 is
  **0.999**. The hard-decoy panel is adversarial by construction; a setting that
  looks fine on average is not.
- Only **33 / 162** cells reach a margin ≥ 0.40, and only **16 / 162** keep
  worst-decoy P95 ≤ 0.35.
- The best-margin cell is `(α, β) = (1.2, 0.8)` with `0.1 S + 0.9 C`: margin
  **0.503**, self 0.756, decoys ≥ 0.35 at **0.35 %**. It is *not* recommended:
  at 10 % shape weight it is nearly blind to volume, so it would reward a
  pharmacophore-decorated fragment during RL exactly as the 2026-10-31
  shrinkage autopsy describes.
- `α + β = 1` (Dice scale) settings inflate *both* self and decoy scores, which
  is why panel C plots margin — a metric that only looks at self-recovery would
  rank them first.

---

## Why this matters for the RL reward

The reward threshold used during training is 0.35 on the normalized scale. With
the default scorer, 99.6 % of worst-decoy pairs on the curated panel score above
that threshold — meaning **the 0.35 gate is not a selectivity test**. It is a
"sane 3D pose" gate. Selectivity has to come from the reference bank and from
the objectives alongside ROCS, not from the threshold.

That conclusion is what component 3 tests directly: whether a different reference
policy (clustering rule, `K`) restores a margin that the scorer cannot.

---

## Regenerate

```bash
python -m _david.code.conformer.cli figures --canonical-only
```

Inputs: `tversky_grid` (`drug121_tversky_color_162grid.csv`) and
`egcg_hard_decoys` (optional; panel D degrades to a note if absent). Both
digests are recorded in `tversky_color_162grid_selectivity.manifest.json`.
