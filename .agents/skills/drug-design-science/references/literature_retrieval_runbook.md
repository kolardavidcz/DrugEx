# Literature discovery and method consultation

Use `consult-literature` for DrugEx method decisions and the installed `literature-research` skill for database selection, free access, critical appraisal, and citation provenance. Reuse its maintained PubMed, Europe PMC, arXiv, bioRxiv, and OpenAlex helpers instead of building another search client. Resolve helper paths for the active host.

The suite's `literature` command is a small Europe PMC metadata search. Europe PMC includes preprints as well as peer-reviewed articles; an indexed hit is not evidence of peer review. Metadata or an OA flag does not mean the full text was inspected. Follow up using actual PMID/PMCID/DOI records and label access limits. The maintained Europe PMC helper discovers all metadata by default; use `--open-access-only` for a deliberately restricted search. Neither an empty restricted search nor unavailable full text establishes absence from the broader literature.

## Verified starting records

The following records were resolved through Europe PMC on 2026-10-27. They are starting points, not a sufficient review of a new research question:

- Liu et al. (2021), *DrugEx v2: de novo design of drug molecules by Pareto-based multi-objective reinforcement learning in polypharmacology*. [DOI 10.1186/s13321-021-00561-9](https://doi.org/10.1186/s13321-021-00561-9), PMID 34772471, [PMC8588612](https://pmc.ncbi.nlm.nih.gov/articles/PMC8588612/). Full text retrieved and the exploration/reward methods inspected for this migration audit. Its RNN/Pareto workflow is relevant, but its predictor objectives are not a validation of the thesis's ROCS/SA/QED reward.
- Liu et al. (2023), *DrugEx v3: scaffold-constrained drug design with graph transformer-based reinforcement learning*. [DOI 10.1186/s13321-023-00694-z](https://doi.org/10.1186/s13321-023-00694-z), PMID 36803659, PMCID PMC9940339. Record/abstract checked here; inspect full methods before using it to support an architectural decision. This is a different generator/input representation from the active unprompted SequenceRNN.
- Šícho et al. (2023), *DrugEx: Deep Learning Models and Tools for Exploration of Drug-Like Chemical Space*. [DOI 10.1021/acs.jcim.3c00434](https://doi.org/10.1021/acs.jcim.3c00434), PMID 37272707, PMCID PMC10306259. Record/abstract checked here; this is the software paper, not the v3 graph-transformer paper.

The old runbook mixed up titles/authorship and listed an unverified 2026 metric paper without identifiers. Do not copy that bibliography. Resolve new references from primary records, distinguish software versions and publication versions, and preserve source links. Do not use citation counts or author h-index as a proxy for methodological validity.
