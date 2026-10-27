# Antigravity science-skill migration audit

Checked 2026-10-27 in local WSL, repository baseline `ed5b32b`, with unrelated in-progress conformer/visualization work in both checkouts. This audit concerns agent guidance and the retrieval helper; it does not validate or change research algorithms or approve experimental settings.

## Findings and corrections

| Earlier claim/behavior | Evidence and correction |
| --- | --- |
| CHEMBL2016 is DNMT1 | Live [ChEMBL target record](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL2016.json) identifies human coagulation factor IX. Human DNMT1 search resolves to [CHEMBL1993](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL1993.json). Corrected examples; helper now displays target identity. |
| Confidence >=7 means direct single-protein binding | [ChEMBL definitions](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions) assign 7 to direct complex subunits, 8 to homologous single proteins, and 9 to direct single proteins. Activity filtering alone does not join confidence metadata. |
| IC50 and Ki can be presented as an unlabeled affinity/inhibitor set | The old helper lost relation/validity/document context and did not filter censored values. It now requests exact positive observations and retains provenance. Output remains uncurated; cutoff choice, assays, mechanism, and completeness need explicit review. |
| Descriptor profiling performs standardization/pH 7.4 preparation | Code only parses the supplied structure and computes descriptors. Removed unsupported preprocessing claims. Average molecular weight and monoisotopic exact mass are now separate fields. |
| A `.tsv` output name produces TSV | Code always serialized JSON. CLI now accepts JSON output explicitly and rejects a misleading extension. |
| Never use vacuum; use epsilon 6.5/78.4 and K_ref=3/K_gen=10 | No source/raw benchmark in the skill establishes this universal rule. [RDKit API](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html) documents dielectric controls, not universal physiological parameters. Replaced prescriptions with consultation and validation requirements. |
| `timeout=0` means unlimited RL conformer generation | Inspected `create_rocs_scorer()` and core generator: timeout is not forwarded and the generator default is 20 s. Documented as an existing code issue; no runtime change here. |
| 4DA4 is human DNMT1 at 2.48 Å | Live [RCSB entry](https://data.rcsb.org/rest/v1/core/entry/4DA4) identifies **mouse** DNMT1 and 2.60 Å. Corrected and removed unverified pocket-residue claims. |
| Codex can invoke Antigravity `/science` sidecars | No such capability is supplied by these repository files. Removed mandatory unavailable tool calls; use tools actually available. |
| DrugEx paper list is verified | Europe PMC records distinguish the 2021 v2, 2023 v3 (Liu et al.), and 2023 software (Šícho et al.) papers. Corrected titles/attribution; omitted the unverified metric-paper entry. |

## Retrieval evidence and limits

Used keyless public APIs: ChEMBL target/search, RCSB entry, PubChem PUG REST, and the installed Europe PMC skill. Search: `TITLE:DrugEx` with the helper's `OPEN_ACCESS:y` filter; three records returned. Retrieved PMC8588612 full text and inspected exploration/reward sections. V3/software entries were checked at record/abstract level only. Both the existing multi-property PubChem URL and the stereochemistry-preserving SMILES endpoint returned EGCG successfully; no endpoint failure was inferred.

The helper is deliberately a bounded retrieval utility. It does not join assay confidence, paginate complete datasets, resolve selectivity, prepare pH-dependent protonation states, or perform a full-text evidence review. Use maintained literature skills and an explicit dataset workflow for those tasks. The repaired metadata path does not establish scientific validity of any retrieved assay or generated compound.

For future methodological choices, invoke `consult-literature`; reuse this audit only for claims whose code and external records still match. Review current uncommitted work before drawing implementation conclusions.

## Validation

The helper's 11 regression tests passed locally (RDKit 2026.09.1) and on dichtator using its existing project environment. Ruff passed for the helper and tests on both hosts. Both skill entrypoints passed the bundled skill validator; all nine local Markdown links resolved. Live smoke checks passed for PubChem EGCG profiling, ChEMBL DNMT1 resolution and a two-observation query, Europe PMC discovery, and RCSB entry metadata. The 3PTA coordinate download failed with `TimeoutError: The read operation timed out` at `resp.read()` after the configured 20-second socket read timeout; a successful live coordinate download is not established. A regression test verifies that a failed read preserves an existing coordinate file. Temporary API outputs and downloaded article text remain outside Git.

Reproduce the focused offline checks from the repository root using the host's Python and Ruff:

```bash
.venv/bin/python -B -m unittest discover -s .agents/skills/drug-design-science/tests -p 'test_*.py' -v
ruff check --no-cache .agents/skills/drug-design-science/scripts .agents/skills/drug-design-science/tests
```

These checks establish helper behavior and source retrieval, not scientific validation of conformer settings or RL outcomes. Method-selection behavior is specified in the skill and root AGENTS.md; no independent agent behavioral evaluation was run.
