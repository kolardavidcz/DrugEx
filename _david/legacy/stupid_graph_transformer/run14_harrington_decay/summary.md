# Historical experiment record

These generated metrics describe a legacy fragment-conditioned run. They were not independently revalidated during Codex onboarding and are not current settings or instructions. Check code revision, raw data, score scale, and evaluation protocol before thesis use or comparison with SequenceRNN results.

# DrugEx Run Summary: run14_harrington_epsilon_decay

- **Target Molecule**: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(...`
- **Epochs**: `100`
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
| 1.614 | 3.043 | 0.637 | 406.434 | COc1cc(O)cc(C(=O)OC2Cc3ccc(OC)cc3OC2c2ccccc2)c1 |
| 1.612 | 2.912 | 0.544 | 362.381 | O=C(OC1Cc2ccccc2OC1c1ccccc1)c1ccc(O)c(O)c1 |
| 1.584 | 3.279 | 0.416 | 377.396 | O=C(NC1(O)Cc2ccccc2OC1c1ccccc1)c1ccc(O)c(O)c1 |
| 1.576 | 2.933 | 0.689 | 362.381 | O=C(OC1Cc2ccc(O)cc2OC1c1ccccc1)c1ccc(O)cc1 |
| 1.568 | 2.944 | 0.735 | 348.398 | Oc1ccc(COC2Cc3ccc(O)cc3OC2c2ccccc2)cc1 |
