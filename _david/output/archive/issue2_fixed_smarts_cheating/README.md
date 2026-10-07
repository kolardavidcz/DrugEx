# Methodological Issue 2: Fixed SMARTS / Hardcoded Toxicophore Cheating

**Historical Run ID**: `20261002_193919`  
**Execution Mode**: Test Mode (Graph-Transformer with Hardcoded Catechol Suppression)  
**Host Hardware**: `dichtator` (NVIDIA RTX PRO 4000 Blackwell GPU 0)

---

## 1. What This Experiment Attempted
EGCG contains 8 phenolic hydroxyl groups arranged in ortho-catechol and pyrogallol patterns. In biochemical assays, catechols undergo spontaneous redox cycling, forming reactive ortho-quinones that act as Pan-Assay Interference Compounds (PAINS).

To suppress this liability, this experiment hardcoded a strict chemical SMARTS filter into the RL objective:
```python
Scaffold(smart="c1c([OH])c([OH])ccc1", is_match=False)  # Threshold: 0.99
```
Columns generated in `stage1_graph_transformer/generated_candidates.tsv`:
```text
SMILES    FRAGS    Valid    ElectroShape    Heavy    RAscore    QED    SA    Scaffold(smart=c1c([OH])c([OH])ccc1,is_match=False)    Desired
```
Summary metric reported in `result_analysis_summary.json`:
```json
"catechol_free_count": 32,
"catechol_free_percentage": 100.0
```

---

## 2. Why It Failed Academically
1. **Destruction of Pipeline Universality**:
   - The primary objective of an academic LBDD framework in DrugEx is **universality**: the system must accept *any* arbitrary query ligand (SMILES or 3D SDF) and optimize leads automatically.
   - Hardcoding `c1c([OH])c([OH])ccc1` specifically tailored the RL objective to EGCG. If given imatinib, methotrexate, or a peptide-mimetic, catechol filtering is irrelevant and introduces arbitrary scoring bias.
2. **Artificial Chemical Bias**:
   - Penalizing specific sub-patterns by hand bypasses general generative learning. Instead of the neural policy learning the steric envelope, it was micro-managed by manual rules.

---

## 3. Academic Resolution
In our current pipeline:
- **Zero hardcoded SMARTS or chemical pattern penalties** exist in the RL loop.
- The pipeline calibrates target physicochemical envelopes **dynamically relative to the query molecule** ($[0.70 \times X, 1.30 \times X]$ for HAC, CLogP, and TPSA) using native DrugEx `SmoothHump` modifiers.
- Core scaffold frameworks (Murcko scaffolds) are extracted dynamically using RDKit's Murcko algorithm, allowing the pipeline to operate purely from any query SMILES with zero human micro-management.
