# Historical experiment record

These generated metrics describe a legacy fragment-conditioned run. They were not independently revalidated during Codex onboarding and are not current settings or instructions. Check code revision, raw data, score scale, and evaluation protocol before thesis use or comparison with SequenceRNN results.

# DrugEx Run Summary: run15_harrington_deep_sampling

- **Target Molecule**: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(...`
- **Epochs**: `60`
- **Criteria**: `avg_amean`
- **Candidates Evaluated**: `200`

## Presentation Deliverables
- [Lead Structures 2D](lead_structures_2d.png)
- [Property Distributions (2x5 Profile)](property_distributions.png)
- [RL Training Curves](rl_training_curves.png)
- [Top Prioritized Leads (TSV)](top_prioritized_leads.tsv)

## Top Prioritized Chemical Leads

| ROCS | SA | QED | MW | Canonical_SMILES |
| --- | --- | --- | --- | --- |
| 1.678 | 3.106 | 0.602 | 378.380 | O=C(OC1Cc2c(O)cc(O)cc2OC1c1ccc(O)cc1)c1ccccc1 |
| 1.662 | 3.002 | 0.516 | 392.407 | COc1ccc2c(c1)OC(c1ccccc1)C(OC(=O)c1ccc(O)c(O)c1)C2 |
| 1.644 | 3.189 | 0.583 | 392.407 | Cc1cccc(C(=O)OC2Cc3c(O)cc(O)cc3OC2c2ccc(O)cc2)c1 |
| 1.628 | 3.069 | 0.475 | 378.380 | O=C(OC1Cc2ccc(O)cc2OC1c1ccccc1)c1ccc(O)c(O)c1 |
| 1.624 | 2.961 | 0.516 | 392.407 | COc1ccc(C2Oc3ccccc3CC2OC(=O)c2ccc(O)c(O)c2)cc1 |
