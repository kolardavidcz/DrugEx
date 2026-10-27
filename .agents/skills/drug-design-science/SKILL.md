---
name: drug-design-science
description: Inspect ligands, target records, assay data, experimental structures, and conformer/scoring protocols for DrugEx RNN ligand-based de novo design. Use for ligand profiling, ChEMBL/PubChem retrieval, structure inspection, or conformer diagnostics. Route scientific method choices through consult-literature before recommending or changing them.
---

# DrugEx scientific support

The active thesis uses unprompted SequenceRNN generation; EGCG is a scoring reference. Read root and relevant nested AGENTS.md and inspect current code. DNMT1 and historical benchmarks are examples, not a standing target selection or approved experimental plan.

Before choosing or changing research methods, use [consult-literature](../consult-literature/SKILL.md), which routes evidence searches through the installed `literature-research` skill. Keep existing research choices intact while investigating alternatives. No fixed dielectric, conformer count, potency cutoff, or structure identifier in an old report is automatically authoritative.

## Choose the relevant workflow

- Ligand/assay retrieval and dataset preparation: [active ligand mining](references/active_ligand_mining_runbook.md).
- Protein identity, experimental structures, and visual inspection: [target profiling](references/target_profiling_runbook.md).
- Conformer generation, dielectric parameters, alignment, and validation: [conformer and alignment](references/conformer_and_alignment_runbook.md).
- Primary papers and bibliography: [literature retrieval](references/literature_retrieval_runbook.md).

Read only the references needed for the task. Use currently available tools or public APIs. Antigravity `/science` tools and `sidecar://` links are not Codex capabilities; do not invent viewer availability. An interactive structure viewer is optional when available; report what was actually inspected.

## Small retrieval helper

[scripts/drug_design_suite.py](scripts/drug_design_suite.py) provides RDKit descriptors, PubChem name resolution, PDB metadata/downloads, a bounded ChEMBL activity query, and Europe PMC discovery metadata. It does not curate a training set, model physiological protonation, or perform a literature review.

From the repository root, use the active host's project Python, for example:

```bash
.venv/bin/python .agents/skills/drug-design-science/scripts/drug_design_suite.py profile --name EGCG
.venv/bin/python .agents/skills/drug-design-science/scripts/drug_design_suite.py pdb --id 3PTA
.venv/bin/python .agents/skills/drug-design-science/scripts/drug_design_suite.py chembl --target CHEMBL1993 --limit 5
```

The ChEMBL command prints the resolved target name/species and returns activity observations, not verified inhibitors. Its potency cutoff is configurable and illustrative. Check assays and source publications before interpreting results. Literature metadata needs full-text follow-up using the literature skills.

The [migration audit](references/legacy_audit.md) records verified problems in the earlier skill and its remaining scope limits. Do not apply historical quantitative claims without revisiting their underlying data.
