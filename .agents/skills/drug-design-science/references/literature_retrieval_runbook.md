# Academic Literature & SOTA Discovery Runbook

This runbook standardizes discovery, downloading, and citation tracking of peer-reviewed articles and preprints in generative molecular machine learning, DrugEx algorithms, and target pharmacology.

---

## 1. Europe PMC & PubMed (Peer-Reviewed Pharmacology)

Search Europe PMC for peer-reviewed studies, biological activity assays, and open-access PDFs:

```bash
# Query DrugEx and molecular generative reinforcement learning
python scripts/drug_design_suite.py literature --query "DrugEx de novo drug design" --limit 5

# Query target pharmacology (EGCG epigenetic mechanism or DNMT1 inhibition)
python scripts/drug_design_suite.py literature --query "Epigallocatechin gallate DNMT1 inhibition" --limit 5
```

### Search Syntax Tips:
- Use quotes for exact keyphrases: `"reinforcement learning"`, `"de novo drug design"`.
- Filter for Open Access full text: `OPEN_ACCESS:y`.
- Retrieve full-text XML/PDF using PMCID when available.

---

## 2. arXiv & bioRxiv (Machine Learning & Generative Chemistry Preprints)

For frontier algorithms, graph transformers, and multi-objective reinforcement learning (MORL):

- **bioRxiv Categories**: `Bioinformatics`, `Biophysics`, `Pharmacology and Toxicology`.
- **arXiv Categories**: `cs.LG` (Machine Learning), `q-bio.BM` (Biomolecules), `stat.ML`.

### Canonical Key Papers for the Thesis:
1. Liu et al. (2019): *DrugEx v1: De Novo Design of Drug Molecules by Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.*
2. Liu et al. (2021): *DrugEx v2: De Novo Design of Drug-like Molecules by Multi-Objective Deep Reinforcement Learning.*
3. Šícho et al. (2023): *DrugEx v3: Scaffold-Constrained De Novo Design using Multi-Objective Reinforcement Learning.*
4. Fil et al. (2026): *Scaffold-based evaluation metrics for fair comparison of molecular generators.*

---

## 3. OpenAlex Citation Graph & BibTeX Generation

Use OpenAlex (`literature_search_openalex`) to:
- Resolve DOIs to standardized BibTeX entries for your thesis bibliography (`.bib`).
- Retrieve citation counts, h-index, and author co-citation networks.
- Verify whether benchmark competitor models (e.g. REINVENT, ChemGE, GraphAF) are cited accurately.

---

## 4. Undermind MCP & Tiered Literature Strategy

We employ a two-tiered literature discovery protocol designed to maximize search depth while managing API quotas:

```
                      [ Scientific Query ]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    【Tier 1: Primary Depth】       【Tier 2: Unmetered Fallback】
    Undermind MCP (AI agentic)      Europe PMC + arXiv + OpenAlex
    - Complex thesis questions       - Routine citation verification
    - Obscure synthesis routes       - Free, unmetered full text
    - Cross-domain correlations      - Active when Undermind quota exhausted
```

### Tier 1: Undermind MCP (Primary)
- **Endpoint**: `https://mcp.undermind.ai/mcp` (OAuth Client: `antigravity`)
- **Strengths**: Conducts autonomous multi-step searches across millions of scientific papers, reading and analyzing full texts to answer nuanced questions.
- **Trigger**: Complex mechanism-of-action questions, obscure polyphenol binding kinetics, novel RL exploration strategies.

### Tier 2: Open Science Fallback (Secondary / Zero-Cost)
Whenever free Undermind query limits are reached or for quick title/DOI lookups:
1. **Europe PMC**: Run `python scripts/drug_design_suite.py literature --query "..."` for open-access XML/PDFs.
2. **arXiv / bioRxiv**: For computational chemistry and generative ML preprints.
3. **OpenAlex**: For automated BibTeX exports and citation metrics.

