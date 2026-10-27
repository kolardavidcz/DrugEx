---
name: consult-literature
description: Consult primary literature before choosing, recommending, or changing a DrugEx research algorithm, objective, molecular representation, conformer protocol, scientifically consequential hyperparameter, dataset curation rule, or evaluation method. Also use to challenge proposed DrugEx methods. Routine software fixes that preserve scientific behavior do not need a new review.
---

# Consult literature before DrugEx method decisions

Use this skill for DrugEx research decisions, including question-shaped proposals such as "maybe use this reward" or "should we set K to 10?". The user owns the scientific decisions. Research discussion authorizes inspection and consultation; it does not authorize a new experiment or a change to the research method.

## Establish the decision

Read repository guidance and inspect the relevant implementation, configuration, and available raw diagnostics. State the specific decision, intended benefit, existing baseline, and assumptions. Check existing DrugEx and RDKit implementations before proposing new code. Distinguish the standalone conformer study from the conformer path actually used by RL.

Consult before making the recommendation or implementing a methodological choice. Do not turn every refactor, typo correction, logging fix, or equivalent optimization into a literature exercise. Reuse an inspected, applicable evidence note when its question, code assumptions, source versions, and limitations still match; refresh missing or outdated evidence.

## Consult and challenge

Use the installed `literature-research` skill and its free database routes. Resolve its current location on the execution host; on Linux the managed skill is normally `~/.agents/skills/literature-research/SKILL.md`. Read the chosen database skill before using its helpers. Keep access keyless/free where available; use no paid plans or trials.

- Search for the original method, relevant comparisons, and evidence against the proposed assumption. Use official version-appropriate library documentation to check API behavior.
- Inspect accessible methods/results, not just titles or abstracts. Label abstract-only evidence and preprints. Retain DOI/PMID/arXiv ID, exact source link, version/date, supporting section/page, and access level for material claims.
- Compare molecular representation, objectives, data, baselines, leakage controls, score scales, sampling denominators, and compute budget with this project. A paper's benchmark or one EGCG self-recovery test does not establish performance here.
- Separate reported findings, verified code behavior, your inference, and untested hypotheses. A dielectric parameter is not evidence of a validated solvent model; a scalar threshold is not an established active-set definition.
- If useful evidence is unavailable, report the gap. Offer a bounded comparison that could resolve it; do not invent a citation, a universal constant, or successful retrieval.

## Give the user a decision they can assess

Lead with a qualified recommendation. Explain the current method and the most relevant alternative, supporting and opposing evidence, practical trade-offs, and what would change the conclusion. Scale the search and detail to the decision; a small API question does not need a full review.

For a substantial decision, keep a concise dated evidence note in the project's existing research-notes location, or alongside the affected method documentation if no such location exists. Record search terms, source access, code revision plus relevant uncommitted changes, assumptions, and unresolved questions. Do not store credentials or bulk downloaded papers in Git.

Before a proposed experimental comparison, specify the baseline, configurations, seeds, actual retained counts, evaluation denominator, raw-versus-modified score scales, and success/failure criteria. Include invalid/duplicate samples, latency, shrinkage, and diversity when relevant. Put meaningful diagnostics in `visualization.py` when implementing an approved experiment.

If the user requested discussion, present the evidence and stop before changing methods. If implementation of the specific method is already authorized, consult first and proceed within that authorization. If evidence calls for a materially different scientific choice, explain it and ask for that choice while continuing independent work.

## Project science support

Use [drug-design-science](../drug-design-science/SKILL.md) for ligand, assay, structure, and conformer inspection. It provides domain workflows, not a source of approved research settings. Its [legacy audit](../drug-design-science/references/legacy_audit.md) records errors found during migration from Antigravity; old reports and their action lists remain historical evidence.
