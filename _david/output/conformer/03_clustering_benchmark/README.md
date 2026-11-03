# 03 · Clustering and reference-bank benchmark

**Question.** A reference pool of 50–200 energy-surviving poses must be reduced
to `K` deployable references. Which merging rule should do it, and what does the
choice cost in score, time and selectivity?

**Figure.** `drug128_definitive_synthesis.png` (121-drug grid)
**Kernel.** `studies/clustering.py` and `ensemble/select.py`
(`SelectionMode.ENERGY | BUTINA | BALANCED | DIVERSE`).

---

## The four selection rules under test

| Rule | How it reduces a pool |
|---|---|
| `rmsd_0.35` | Butina clusters at 0.35 Å RMSD, lowest-energy representative per cluster |
| `rmsd_0.50` | same at 0.50 Å |
| `rmsd_1.25` | same at 1.25 Å — the historical "one conformer is enough" setting |
| `diverse` | farthest-first shape+colour diversity, seeded by the lowest-energy pose, round-robin across stereoisomers |

The physics are held fixed (`MMFF94s ε=78.4` references, `ETKDG` candidates) so
only the merging rule varies. `C=20` candidates, score = TanimotoCombo
normalized to `[0, 1]`.

---

## Panels

| Panel | Question |
|---|---|
| A | Score vs reference cap for Butina 0.35 / 0.50 / 1.25 Å and farthest-first diversity |
| B | Speed/accuracy frontier at `K=4`: is a minimized-but-small candidate pool ever beaten by a larger unminimized one? |
| C | Selectivity audit: self-recovery vs cross-drug decoy score, so a metric that inflates both cannot masquerade as better |
| D | Distinct reference clusters per drug — *why* 1.25 Å fails |

---

## What `K=4` actually buys

| Rule | Score | Realized `K` | Clusters in full pool (median) | Seconds |
|---|---:|---:|---:|---:|
| diverse | **0.817** | 3.99 | 50 | 1.357 |
| `rmsd_0.35` | 0.816 | 3.70 | 16 | 1.357 |
| `rmsd_0.50` | 0.816 | 3.58 | 14 | 1.357 |
| `rmsd_1.25` | 0.805 | 2.84 | 4 | 1.357 |

Three things to read out of this table:

1. **Farthest-first diversity and Butina 0.35 Å are the same answer** (0.817 vs
   0.816, well inside run-to-run variation). Diversity spends no RMSD budget and
   is scale-free, so it is the cheaper rule to reason about; Butina 0.35 Å has a
   physical interpretation. Either is defensible — *do not* choose based on
   0.001.
2. **1.25 Å loses 0.012** and, more importantly, realizes only `K = 2.84` of 4
   requested poses: it merges genuinely different basins. Panel D shows the
   mechanism — median 4 clusters in the pool vs 50 for diversity, and some drugs
   collapse to a single cluster.
3. **Seconds are identical** because candidate embedding dominates the cost of a
   cell. Clustering is never the reason to choose a rule; the realized `K` is.

`k_actual` matters for a second reason: a "K=4" bank on a multi-stereoisomer drug
spends part of its budget on stereoisomer coverage, so it holds fewer distinct
poses than `K=4` on a single-isomer drug. Any table that reports `K` without
`k_actual` is hiding that.

The cap sweep itself is the evidence for `K=4` as the elbow (diverse rule):

| `K` | 1 | 2 | 3 | **4** | 8 | all |
|---|---:|---:|---:|---:|---:|---:|
| Score | 0.763 | 0.799 | 0.811 | **0.817** | 0.830 | 0.839 |

`K=1 → 4` buys +0.054; `K=4 → all` buys +0.022 for several times the embedding
cost. At `K=1` every rule scores identically (0.763) because there is nothing to
choose — which is why `K=1` cannot be used to compare merging rules.

---

## Regime comparison at `K=4`, `rmsd_0.35`

| Reference | Candidate | Score | Seconds |
|---|---|---:|---:|
| MMFF94s ε=78.4 | MMFF94s ε=78.4 | **0.945** | 1.953 |
| ETKDG (unminimized) | ETKDG (unminimized) | 0.913 | 1.357 |
| MMFF94s ε=78.4 | ETKDG (unminimized) | 0.816 | 1.357 |

The asymmetric mix is the worst of the three: it pays for a minimized reference
bank and then scores it against poses that were never minimized. This is the
clustering-side confirmation of component 1's asymmetry result, and the reason
the production pipeline pairs a minimized reference with a **minimized
candidate** whenever accuracy, not latency, is the binding constraint.

---

## Why this decides the reference policy

The 2026-10-31 shrinkage autopsy showed the RL failure mode was not "the scorer
is too weak" but "the reward gate can be satisfied by fragments". A reference
bank is the other half of that gate:

- **too small a bank** (`K=1`) makes self-recovery depend on one arbitrary pose,
  and the decoy floor rises;
- **too aggressive a merge** (1.25 Å) removes the basins that distinguish a
  genuine match from a volume-compatible decoy;
- **too large a bank** is free in score but not in embedding cost, which is why
  `K=4` is the production elbow.

The figure is titled "definitive synthesis" for that reason: it is the panel that
closes the question rather than adding another exploratory sweep.

---

## Regenerate

```bash
python -m _david.code.conformer.cli figures --canonical-only
```

Inputs: `synthesis_grid` (`drug128_synthesis_grid.csv`) and the optional
`cross_decoy` table used by panel C. Digests are recorded in
`drug128_definitive_synthesis.manifest.json`.
