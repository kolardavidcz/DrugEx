# Activity retrieval and dataset curation

Before adopting a scientific curation rule, use `consult-literature`. Define the endpoint, target species, intended use, and inclusion/exclusion policy with the user; keep those choices explicit.

## Verify biological identity

Resolve the ChEMBL target record and check preferred name, organism, target type, and protein accession before querying activities. On 2026-10-27, human DNMT1 resolved to [CHEMBL1993](https://www.ebi.ac.uk/chembl/api/data/target/CHEMBL1993.json). The previous example CHEMBL2016 resolves to coagulation factor IX and must not be labeled DNMT1. Verify identifiers again for a new dataset.

The helper's `chembl` command returns one bounded page of exact positive IC50/Ki observations in nM up to the supplied cutoff. It retains relation, target, assay, document, and validity/duplicate fields. It does not join assay metadata, enumerate all pages, or establish mechanism, selectivity, direct binding, or a complete active set. A small `--limit` is a discovery sample, not an unbiased dataset.

## Keep assay meaning and provenance

Read the [official ChEMBL assay/activity definitions](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions). Confidence **9** denotes a direct single-protein assignment; **8** a homologous single protein; **7** direct protein-complex subunits. Confidence concerns target assignment, not proof of an inhibition mechanism. Join the assay record when such filtering is needed.

Keep IC50, Ki, and Kd distinct. IC50 is assay-dependent potency and is not generally a binding constant. Retain standard relation and units, assay type/conditions, species, confidence, source document, data-validity comments, and potential duplicates. Decide explicitly how censored observations, replicates, mixtures, missing structures, and contradictory assays enter the dataset. A 10 micromolar threshold is a possible task-specific rule, not a universal definition of activity.

For a reusable dataset, paginate with the API's page metadata; record query, date/release, exclusion counts, and raw record identifiers. Split compounds/scaffolds before fitting preprocessing or predictors to prevent leakage. Do not collapse all assay types into one unlabeled affinity column.

## Chemical representation

The `profile` command computes descriptors on the supplied RDKit molecule and preserves specified stereochemistry. It does not desalt, select a parent, assign pH 7.4 protonation, or curate stereochemistry. Choose these transformations separately and record them. Largest-fragment selection can discard relevant components; uncharging is not pH-dependent protonation prediction. Prefer existing DrugEx/RDKit standardization code and inspect its exact behavior.

Export the format actually implemented: the helper writes JSON only. A curated TSV requires an explicit schema retaining endpoint, relation, units, and provenance. Keep average molecular weight separate from monoisotopic exact mass.
