# Bachelor Thesis Project Notes: Ligand-Based De Novo Drug Design in DrugEx

## 1. Primary Query Seed Molecule (EGCG Derivative)
```python
my_molecule = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
# Formula: C22H18O11 | Heavy Atom Count (HAC): 33 | MW: 458.37 g/mol | Phenolic Hydroxyls: 8
```

---

## 2. Skeptical Academic Evaluation: SBDD vs Pure LBDD Methods

### A. ProLIF / PLIF (Protein-Ligand Interaction Fingerprints) — REJECTED
- **Mechanism**: Calculates interaction bit vectors (H-bonds, salt bridges, $\pi$-stacking, hydrophobic contacts) with specific binding pocket amino acid residues (e.g., CCR2 5T1A allosteric pocket residues).
- **Academic Verdict**: **Strictly Out of Scope for Pure LBDD**.
  - ProLIF requires experimentally resolved or modeled 3D protein atomic coordinates and docked ligand poses.
  - In a 100% pure Ligand-Based Drug Design (LBDD) regime, **zero receptor coordinates exist**.
  - Attempting to dock candidates into 5T1A to obtain ProLIF vectors introduces massive docking scoring bias, force-field inaccuracies, and structural artifacts, and fundamentally violates the bachelor thesis definition.
  - Therefore, ProLIF cannot be used in pure LBDD.

### B. 2D Pharmacophore Fingerprints / Graph Filters — REJECTED
- **Mechanism**: Encodes pharmacophore feature pairs separated by topological graph distance (bond path lengths).
- **Academic Verdict**: **Inadequate for 3D De Novo Drug Design**.
  - Flattens 3D conformational geometry into 2D graph steps.
  - Misses out-of-plane dihedral orientations, steric volume, and chiral spatial envelopes (e.g., EGCG's tri-aromatic galloyl ester architecture).
  - High false positive rate where 2D bit patterns match but 3D steric overlap is zero.

### C. USRCAT & ElectroShape (Pure 3D LBDD) — ADOPTED
- **Mechanism**:
  - Ultrafast Shape Recognition with CREDO Atom Types (USRCAT; Ballester et al., 2010) extends USR by evaluating distance distributions (mean, variance, skewness) to 4 reference geometric points (molecular centroid, atom closest to centroid, atom furthest from centroid, atom furthest from that) across 5 atom subsets:
    1. All atoms (12 moments)
    2. Hydrophobic atoms (12 moments)
    3. Aromatic atoms (12 moments)
    4. Hydrogen-bond donors (12 moments)
    5. Hydrogen-bond acceptors (12 moments)
    Total: **60-dimensional rotationally and translationally invariant descriptor vector**.
  - ElectroShape (Armstrong et al., 2010) augments this with partial atomic charge moments.
  - Implemented natively in RDKit: `rdkit.Chem.rdMolDescriptors.GetUSRCAT` and `GetUSR`.
- **Academic Rationale**:
  - 100% pure ligand-based: requires zero protein coordinates.
  - **No alignment / overlay ambiguity**: While ROCS/rdShapeAlign requires numerical optimization of Euler angles and translation vectors (which easily gets trapped in local rotational minima), USRCAT similarity is an exact $O(1)$ inverse Manhattan distance:
    $$S(u, v) = \frac{1}{1 + \frac{1}{60} \sum_{i=1}^{60} |u_i - v_i|}$$
  - Executes in microseconds, enabling fast 3D shape and pharmacophore scoring during RL rollouts and post-generation filtering.

---

## 3. Empirical Diagnosis of Previous Pipeline Failures

Previous pipeline executions yielded mean ROCS ~0.26–0.28 and Desired Ratio only 0.5%–1.0%. The root causes were identified and resolved:

| Root Cause | Mechanism of Failure | Architectural Solution |
|---|---|---|
| **1. Conformer Undersampling** | `ROCS_MAX_CONFS = 2` during RL left flexible molecules (4–10 rotatable bonds) in a conformational desert. | Upgraded to `ROCS_MAX_CONFS = 10` (with 50 reference conformers), allowing realistic conformational search. |
| **2. Pareto Reward Hacking** | In 5-objective Pareto ranking, candidates scoring ~0.98 on RAscore, ~0.98 on 2D Novelty, and ~0.85 on QED dominated the Pareto front while completely ignoring the difficult 3D shape objective (~0.25). | 1) Removed counterproductive 2D Novelty objective from RL (scaffold hopping is handled in Stage 2 funnel). 2) Tightened 3D shape modifiers (`SmoothClippedScore` threshold 0.50). 3) Added USRCAT dual 3D shape scoring. |
| **3. Fragment Volume Mismatch** | High QED and RAscore naturally favored small 15–20 heavy atom fragments. Physically, an 18-atom molecule cannot fill EGCG's 33-heavy-atom steric volume ($\text{ROCS} \le 0.35$). | Added heavy-atom Gaussian size constraint (`Property(prop="Heavy", modifier=Gaussian(mu=33, sigma=6))`) and scaffold heavy-atom filter (14–32 atoms) in Stage 2. |
| **4. Artificial Mock Benchmark** | `_david/thesis_benchmark.py` previously generated synthetic random-normal numbers (`_simulate_regime_data`), masking actual performance. | Completely purged all synthetic simulation code. All benchmark tables and figures are generated strictly from authentic candidate calculations. |

---

## 4. Multi-Stage Pipeline Architecture

1. **Stage 0: 3D Query Conformer Preparation** (`prepare_query_conformer.py`):
   - Generates 50 ETKDGv3 conformers, MMFF94 force-field minimization, RMSD pruning (0.50 Å), and thermodynamic Boltzmann energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
   - Pre-computes and caches 60-dimensional reference USRCAT moment array (`my_molecule_usrcat.npy`).
2. **Stage 1: Sequence-RNN Global Hit Discovery** (`SequenceRNN.py`):
   - Broad unconstrained chemical space exploration using autoregressive LSTM on Papyrus vocab.
   - Rewards: ROCS 3D TanimotoCombo (10 confs), USRCAT 3D moments, Heavy Atom count (targeting 33 atoms), RAscore synthesizability, QED, and SA.
3. **Stage 2: Bemis-Murcko Framework Extraction & Funnel** (`scaffold_funnel.py`):
   - Decomposes Stage 1 hits into Murcko scaffolds.
   - Heavy-atom volume filtering (14–32 atoms).
   - Purges catechol and polyphenol PAINS liabilities (preventing redox/reactive false positives).
   - Enforces scaffold hopping distance from seed (ECFP4 Tanimoto $\le 0.60$).
   - Clusters scaffolds with Butina sphere exclusion (Tanimoto cutoff 0.40) to select diverse lead centroids.
4. **Stage 3: Graph-Transformer Lead Optimization** (`GraphTransformer.py`):
   - Fragment-conditioned generative RL starting from discovered funnel centroids.
   - Same un-cheatable multi-objective rewards (ROCS + USRCAT + Heavy Atoms + RAscore + QED + SA).
5. **Stage 4: Thesis Publication Benchmark** (`thesis_benchmark.py`):
   - Evaluates authentic candidate files with real calculated properties.
   - Exports `thesis_benchmark_table.tsv`, radar chart, Morgan FP PCA projection, score distribution boxplots, and 2D depiction grids.
