# Methodological Issue 1: 1D Moment Approximations (USRCAT & ElectroShape)

**Historical Run ID**: `20261002_162530`  
**Execution Mode**: Full 3-Stage Funnel (Sequence-RNN -> Scaffold Funnel -> Graph-Transformer)  
**Duration**: 1,384 seconds (~23 minutes)  
**Host Hardware**: `dichtator` (NVIDIA RTX PRO 4000 Blackwell GPU 0)

---

## 1. What This Experiment Attempted
To accelerate reinforcement learning rollouts, 3D shape scoring was decoupled into:
1. **Inner RL loop**: Evaluated ultrafast 1D moment invariants (USRCAT / ElectroShape; Ballester et al., 2010; Armstrong et al., 2010). Distance distributions (mean, variance, skewness) to 4 reference centroids across 5 pharmacophore atom types formed a 60/75-dimensional descriptor vector.
2. **Post-processing**: Filtered candidates downstream with CDPKit ROCS.

Columns generated in `stage2_graph_transformer/rl_generated_candidates_scored.tsv`:
```text
SMILES    FRAGS    Valid    ElectroShape    Heavy    RAscore    QED    SA    Desired
```

---

## 2. Why It Failed Academically
1. **The "Degenerate Sphere" Problem**:
   - Moment invariants lack angular and orientation information. Any compact or spherical molecule with a similar radial distribution of atom types achieves high USRCAT/ElectroShape similarity (~0.60–0.75) despite having **zero true 3D spatial alignment** with EGCG's elongated, planar tri-aromatic gallate ester geometry.
2. **High False Positive Rate**:
   - The generator learned to produce spherical polycyclic scaffolds that satisfied moment distributions without presenting outward-facing pharmacophores.
3. **Alignment-Free Ambiguity**:
   - While moments avoid numerical rotational optimization, they fail to enforce true steric boundary exclusion, allowing atoms to penetrate forbidden spatial envelopes.

---

## 3. Academic Resolution
In our current pipeline, **all 1D moment approximations (USRCAT and ElectroShape) are strictly excluded**. Reinforcement learning operates directly on authentic physical Gaussian volume overlap (`RDKitROCSScorer`) with OpenEye Mills-Dean color SMARTS (`oe_mills_dean`), evaluating true 3D shape and pharmacophore overlap (`TanimotoCombo` on a 0–2 scale).
