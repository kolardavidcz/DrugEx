# Ground Truth Milestone 4 Full Run (Empirical Lead Optimization)

**Run ID**: `m4_empirical`  
**Execution Mode**: Full Graph-Transformer Reinforcement Learning  
**Parameters**: Epochs = 20, Samples/Epoch = 200, Query = Epigallocatechin gallate (EGCG)  
**Host Hardware**: `dichtator` (NVIDIA RTX PRO 4000 Blackwell GPU 0)

---

## Architecture Highlights
1. **Universal 3D Reference**: Automated ETKDGv3 conformer ensemble (50 conformers) + MMFF94 force-field relaxation with Boltzmann energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
2. **Native DrugEx 3D ROCS Scorer**: `RDKitROCSScorer` with OpenEye Mills-Dean color SMARTS (`oe_mills_dean`), 22 sampled conformers, and `TanimotoCombo` (0–2 scale).
3. **Anti-Collapse Soft-Gate Reward**:
   $$R(\mathbf{x}) = 0.70 \cdot S_{\text{3D}}(\mathbf{x}) + 0.30 \cdot \min\left(1.0, \frac{S_{\text{3D}}}{0.60}\right) \cdot \sum \tilde{w}_i S_{\text{prop}, i}(\mathbf{x}) + S_{\text{SA\_floor}}(\mathbf{x})$$
4. **Single Synthesizability Floor**: SAscore rewarding 1.0 for $\text{SA} \le 5.5$ and penalizing $\text{SA} > 6.0$.
5. **Peak Recovery Monitor**: Automatically tracked rolling multi-objective desirability and restored the peak historical checkpoint upon RL completion.
6. **Publication Artifacts**:
   - `stage2_result_analysis/lead_structures_2d.png`: 2D depictions of prioritized leads with metric cards.
   - `stage2_result_analysis/property_distributions.png`: 9-panel property histograms benchmarked against query envelopes.
   - `stage2_result_analysis/result_analysis_summary.json`: Standardized VUN metrics ($\text{Validity} \ge 80\%$, $\text{Uniqueness} \ge 80\%$), scaffold diversity, and ROCS distributions.
