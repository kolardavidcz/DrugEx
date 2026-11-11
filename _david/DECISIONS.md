# Decision notes

Dated records of scientific choices that change a number. Each note says what was decided, on what
evidence, and what would overturn it.

---

## 2026-11-10 — Conformer generation and minimization settings

**Status:** selected · **Confidence:** moderate · **Cheap to reverse:** yes

Settings in `_david/code/config.py`: `minimize = True`, **MMFF94s** (`conformers.FORCE_FIELD`),
`dielectric = 78.4`, `dielectric_model = 1`, `max_conformers = 20` (lowered from 30), `reference_conformers = 4`,
`reference_rmsd = 0.5`, `max_iterations = 500`, `PROTONATE = False`.

### What the evidence settled

**1. Dielectric: keep ε = 78.4, model 1 (constant).** My first reading of this was wrong. I had
concluded the benefit was zero, based only on this project's own conformer study — but that study
measures *self-recovery* (does a molecule's regenerated ensemble match its own bank), which is
ensemble reproducibility, not agreement with a bound conformation. On the metric that matters the
published evidence favours solvated minimization:

[Gur17], 809 drug-like ligands from crystal structures, symmetry-corrected heavy-atom RMSD, four
force fields (OPLS3, OPLS_2005, MMFF94s, AMBER*):

- Gas-phase minimization gave **"largely worsened results"** than solvated minimization.
- Chloroform → water raised sub-1 Å recovery by **8.5–12.3 points** (charged inputs) and
  **3.5–7.8 points** (combined pools).
- The spread *between force fields* was only **4.7 points** (charged, water) and **5.5** (combined).
  **Solvent treatment matters as much as or more than which force field is used.**
- MMFF94s in water: 77.4 / 81.7 / 84.3 % below 1 Å (charged / neutralized / combined) versus
  67.6 / 74.4 / 77.9 % in chloroform.

Confidence is moderate, not high, because [Gur17] used GB/SA implicit solvent, not a uniform
dielectric constant. ε = 78.4 is a crude proxy for what they tested — the *direction* has 809-ligand
support, the *mechanism* does not. It must be described as "uniform dielectric screening (ε = 78.4)",
never as a solvent model: MMFF94 was parameterized against gas-phase data, and a uniform ε only
divides the Coulomb term, with no cavitation or desolvation term.

This project's own 512-molecule study found `mmff_high_dielectric` and `mmff_vacuum` statistically
indistinguishable on self-recovery (0.9161 vs 0.9160, high-dielectric ahead in **18 of 36** matched
cells, mean delta **+0.0001**). That is a real null — on the wrong metric. Its usable finding is that
**minimizing with MMFF at all** beat raw ETKDG by **+0.039** and UFF by **+0.011**.

On EGCG directly (7 ETKDG conformers, both settings): mean heavy-atom RMSD between the two relaxed
geometries **0.323 Å**, global minima **0.196 Å** apart — the same pose. But the energy *ranking*
differs and the lowest-4 **set** changes (vacuum `{4,5,3,0}`, screened `{5,4,6,0}`). ε moves bank
*membership*, not geometry.

**2. MMFF94s (switched from MMFF94 2026-11-10), never UFF.** [Kan17], 700 molecules up to 250 conformers: MMFF94 > GAFF > UFF for
conformer energy ranking, UFF worst by a wide margin (median R² 0.033 vs MMFF94's 0.125 against PM7).
[Lim20], 22,675 structures from 3,271 molecules against B3LYP-D3BJ/DZVP: MMFF94s marginally ahead of
MMFF94 (ddE range −49.5…+25.1 vs −52.1…+29.8 kcal/mol; 47.0 % of MMFF94s conformers within 1 kcal/mol
of QM). Energy ranking worst → best: SMIRNOFF99Frosst < MMFF94 ≈ MMFF94s < GAFF ≈ GAFF2 ≈ OpenFF-1.0/1.1
< OpenFF-1.2 < OPLS3e. UFF was not tested there.

**Switched to MMFF94s.** Halgren introduced the variant in *"MMFF94s option for energy minimization
studies"* (MMFF VI, 1999): "s" planarises delocalised trigonal nitrogen so a *single static minimised
structure* matches the time-averaged geometry crystallography observes, whereas plain MMFF94
reproduces the instantaneous, slightly pyramidal minimum whose average only emerges if you run
dynamics. This pipeline embeds, minimises to a local minimum and scores — static minimisation, no MD
anywhere — so it is the case the variant was written for. The benchmark used as evidence above
([Gur17], 809 crystal ligands) also ran MMFF94s.

Measured impact, this project: **EGCG is bit-identical** under the two variants (dE 0.0000, RMSD
0.000) because it contains no nitrogen, so the reference bank does not move at all. Amides shift
~0.005 Å (acetanilide N 179.5 deg -> 180.0 deg).

**Correction, 2026-11-10** (an adversarial review failed to reproduce the original claim, and both
measurements turned out to be right about different atoms). On sulfanilamide the **aniline** nitrogen
planarises **131.7 deg -> 173.8 deg** (RMSD 0.031 Å, dE 1.10 kcal/mol); the **sulfonamide** nitrogen
does not move at all (135.4 deg under both variants). The first note here called the moving atom the
sulfonamide nitrogen; it is the aniline one. The variant therefore reaches nitrogen conjugated into
an aromatic ring, not sulfonamides. The switch remains about defensibility of the label rather than
about moving a result.

**3. Minimization stays on, but claim little for it.** [Bai10], 742 ligands: minimization moved
≤1 Å recovery by **+0.0 to +3.3 points** and was not uniformly positive (FFBM/MMFF94 at ≤0.5 Å went
15.8 % → 15.5 %); the authors conclude a minimization step "is not necessary and may reduce
diversity". [Mcn23] found the effect modest and generator-dependent. Important caveat: [Mcn23]
minimized with **UFF**, the weakest field in [Kan17], so it understates MMFF94.

**4. `max_conformers` lowered 30 -> 20.** [Mcn23]: fewer than 25 conformers sufficed for most
structure-based screening; **68 of 102** DUDE targets reached their best F1 at ≤ 25, and larger
libraries can *lower* average performance by raising false positives. [Kim13]: cutting ensembles from
100,000 to 500 cost only **0.18 Å** in mean best-match RMSD. The default was 30, on the wrong side of
that line; 20 is inside it and cuts the dominant per-molecule cost by a third.

**5. `reference_rmsd = 0.5` confirmed, and deliberately identical to native.** Native sets
`params.pruneRmsThresh = 0.5` (`drugex/training/scorers/conformer_generators.py:477`). [Mcn23] tested
0.5 / 1.0 / 1.5 / 2.0 Å and found tighter filtering helps most with *small* ensembles — this project's
case at K = 4; a 2 Å filter collapsed a 250-conformer pool to an average of 6 conformers.

**6. `PROTONATE = False` independently corroborated.** [Gur17] found that neutralizing charged groups
before conformational search improved recovery, especially for large flexible ligands (OPLS3/chloroform
69.3 % → 74.2 % below 1 Å), because strong long-range electrostatics between charged groups distort
conformations during search. This project separately measured that Dimorphite-DL returns an
**octa-anionic** EGCG at pH 7.4, which is not the dominant species and destroys every phenol donor the
colour channel depends on.

### Search record

Undermind deep search "conformer minimization force field choices for shape screening", workspace
`60d664b5-989c-4bff-9157-cd90be034a26`, completed 2026-11-10, 124 ranked papers. Full texts read:
[Mcn23], [Lim20], [Kan17], [Gur17], [Bai10]. Abstract only: [Avg15], [Kim13], [Fri17]. Earlier
Tier-2 pass: Europe PMC REST, `resultType=core`.

**Code revision.** `76c1c39`, tree dirty: one permitted native edit
(`drugex/training/scorers/properties.py`, `FAILURE_VALUES`), plus untracked `_david/`, `_david/`.

---

## 2026-11-10 — Reference-bank selection is the weakest link, not the dielectric

**Status:** IMPLEMENTED 2026-11-10 · **Confidence:** high that the previous rule was wrong

`build_reference()` ranks conformers by MMFF single-point energy and keeps the lowest K. Three
independent sources say energy-ranked selection is the worst available strategy.

- [Mcn23], Platinum 2017, selecting 25 RDKit conformers from a 250-conformer pool — fraction of
  ligands whose retained subset contains a pose within 1.0 Å of the crystal conformation:
  **1.0 Å RMSD-diversity filter 72.7 % · unbiased sampling 67.8 % · 25 lowest-energy 57.6 %.**
  Explicit negative result: taking the N lowest-energy conformers "did not improve bioactive
  conformation recovery for N > 1". Cause given: poor geometric diversity, and isolated-ligand
  energies that ignore protein and solvent.
- [Avg15]: a bound-like conformer was present in the ensemble for **91–97 %** of drugs but was the
  global minimum for only **52–56 %**, typically sitting **4–6 kcal/mol above** it.
- [Kan17]: MMFF94 conformer energies correlate poorly with QM — median R² 0.125 against PM7, median
  Spearman **−0.103** against DFT//PM7 (negative). The authors "explicitly advise against trusting
  generic classical force fields to rank or filter conformers when accurate selection matters".

**Change made** (user decision, 2026-11-10; revised the same day, see below). `build_reference()`
discards all but the lowest-energy poses, then selects
`reference_conformers` poses from the survivors by **farthest-point sampling**: the global minimum
anchors the bank, and each further pose is the one whose nearest already-kept neighbour is furthest
away. Energy is a filter; geometry is the selector.

**Measured on EGCG** (24 embedded, K = 4), old rule vs new:

| bank | energies (kcal/mol) | min pairwise RMSD | mean | max |
|---|---|---:|---:|---:|
| lowest-energy (old) | 92.39, 92.92, 97.48, 97.61 | 0.587 Å | 1.638 Å | 2.201 Å |
| diversity in window (new) | 92.39, 92.92, 97.48, 98.49 | **1.058 Å** | **1.778 Å** | 2.221 Å |

The old bank held two near-duplicate poses 0.587 Å apart — barely past the 0.5 Å prune, so they
contributed almost nothing a max-over-bank score could use. Minimum separation rises 80 %, mean
spread 9 %, and the whole cost is one conformer 0.88 kcal/mol higher, well inside the window.

**What would change the conclusion:** a measurement on this pipeline showing that bank membership
barely moves final ROCS rankings — which is plausible, since the score is a max over the bank, and is
cheap to test by scoring one fixed candidate set against an energy-ranked bank and a diversity-ranked
bank.


### Revision, 2026-11-10 — rank-based pool instead of an absolute energy window

The first implementation cut the pool at 10 kcal/mol above the minimum. Review found that an
absolute kcal/mol cut **means something different in each arm**: MMFF single-point energies of raw
ETKDG geometries are strained and spread far wider than relaxed ones, so the same window admitted
**2 unminimised poses against 4 minimised** (measured on EGCG). The score is a max over the bank, so
a smaller bank scores systematically lower and the measured "effect of minimisation" would have been
partly an effect of bank size.

A first patch widened the window when it left fewer than K poses. That removed the size confound but
silently departed from the stated protocol — `config.json` still recorded a 10 kcal/mol window while
the actual rule was different.

`reference_energy_window` is therefore replaced by **`reference_pool` (default 50)**: keep the N
lowest-energy poses, then farthest-point select K from them. Rank is scale-free, so "the best N this
arm produced" means the same thing on both sides, the two banks are equal in size by construction
rather than by a fallback, and nothing about the rule is conditional or unreported.

Measured after the change (24 embedded, pool 12, K=4): minimised `[92.4, 92.9, 97.5, 98.5]`,
unminimised `[160.7, 161.4, 174.2, 192.1]` — **4 poses in both arms**.


### Revision, 2026-11-10 — the reference pool knob was removed

`reference_pool` replaced the absolute energy window, and was then measured to be **inert at the
shipped settings**: EGCG yields only ~15 distinct conformers from 200 attempts after ETKDG's own
0.5 Å pruning, so any pool above that filtered nothing while `config.json` recorded it as active.

The knob is gone. The bank is now farthest-point diversity selection over **every** distinct
conformer embedded, anchored at the lowest-energy pose, with energy used only to anchor and to break
ties. This is also the better-supported rule: energy-ranked selection recovered a crystal pose for
57.6 % of ligands against 72.7 % for RMSD-diversity selection [Mcn23], so adding an energy pre-filter
moved toward the rule that measured worst. Removing it changed no number for EGCG, because it never
fired.

Bank the run will actually use, at the shipped `reference_attempts=200` (reproduced independently):
minimised `[90.75, 93.32, 97.48, 98.49]`, unminimised `[160.67, 161.44, 166.28, 174.15]` — **4 poses
in both arms**.

---

## 2026-11-10 — The reward scheme is a setting, and `WS` is a different experiment

**Status:** IMPLEMENTED · **Confidence:** high (read from DrugEx's own source) · **Cheap to reverse:** yes

`scheme` (`config.py`) selects one of DrugEx's own reward schemes by DrugEx's own `--scheme`
names: `PRCD` (`ParetoCrowdingDistance`, the default and what every run so far used), `PRTD`
(`ParetoTanimotoDistance`) and `WS` (`WeightedSum`). Nothing of ours is in any of them.

### Why this is not just another knob

Both Pareto schemes accept the thresholds and never read them, which is what lets this project
say that p85/EGCG thresholds reach `Desired`, `desired_ratio` and the stop condition and nothing
else. `WeightedSum` does read them (`drugex/training/rewards.py:304`):

```python
weight = ((scores < thresholds).mean(axis=0, keepdims=True) + 0.01) / (
    (scores >= thresholds).mean(axis=0, keepdims=True) + 0.01
)
weight = weight / weight.sum()
return scores.dot(weight.T)
```

So in a `WS` run the thresholds **are** part of the reward, and the weights move every epoch with
the batch: an objective most of the batch fails gets a larger weight. Two consequences the thesis
has to state rather than discover:

1. a `WS` run and a `PRCD` run are not the same experiment with a different ranking rule — they
   are two reward definitions, one of which depends on the calibration;
2. `WeightedSum` adds *raw* columns, so an objective's range is leverage. TanimotoCombo spans
   [0, 2] while SA and QED span [0, 1], so at equal weight shape moves the reward twice as far.
   Pareto ranking is scale-free and has no such asymmetry — which is exactly why the ROCS scorer
   was left unmodified on [0, 2] in the first place.

Decision (user, 2026-11-10): run `WS` **unchanged**, with no rescaling. The comparison is then
between two of DrugEx's own schemes with no code of ours in the reward, and the asymmetry is
reported rather than removed. `report.md` prints the second paragraph above in any `WS` run, and
a test (`RewardSchemes.test_the_report_never_claims_ws_keeps_thresholds_out_of_the_reward`) fails
if that prose ever reverts to the Pareto wording.

**What would overturn it:** evidence that the shape objective's wider range dominates the weighted
sum enough to make the `WS` arm a shape-only run. Cheap to test on the four planned runs: compare
`avg_ColorTanimoto` against the `PRCD` runs at the same epsilon.
