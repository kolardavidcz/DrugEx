# Target identity and structural inspection

Resolve the biological question before fetching structures. The active pipeline is ligand-based; structure retrieval does not authorize a switch to docking or a target-conditioned generator.

Use current UniProt/RCSB records to verify species, accession, construct boundaries, sequence numbering, mutations, experimental method, resolution, ligands, and biological assembly. Fetch polymer/nonpolymer entity metadata or inspect mmCIF where necessary: the helper's entry summary alone does not enumerate ligands or establish pocket suitability. Map residue numbers through the actual construct rather than copying an old residue list.

Examples checked against the RCSB entry API on 2026-10-27:

| Entry | Verified entry description | Resolution |
| --- | --- | --- |
| [3PTA](https://www.rcsb.org/structure/3PTA) | Human DNMT1(646–1600) in complex with DNA | 3.60 Å |
| [4WXX](https://www.rcsb.org/structure/4WXX) | Human DNMT1(351–1600) | 2.622 Å |
| [4DA4](https://www.rcsb.org/structure/4DA4) | **Mouse** DNMT1(731–1602) bound to hemimethylated CpG DNA | 2.60 Å |

These are identity checks, not a recommendation to use a particular binding pocket. Verify bound small molecules and relevance to the intended mechanism separately. Do not call a cofactor-bound or DNA-bound structure a co-crystal of the candidate ligand.

From the repository root, retrieve metadata or coordinates with the active host's Python:

```bash
.venv/bin/python .agents/skills/drug-design-science/scripts/drug_design_suite.py pdb --id 3PTA
.venv/bin/python .agents/skills/drug-design-science/scripts/drug_design_suite.py pdb --id 3PTA --download /tmp/drugex-structures
```

Use an available viewer such as PyMOL or Mol* when visual inspection is useful, and preserve the original coordinate file separately from prepared derivatives. Report the file/entry, chain, residues/ligands, and what was actually inspected. Codex does not inherit Antigravity's `/science` MCP or `sidecar://` viewer links; do not promise an unavailable tool. Distinguish experimental coordinates from predicted structures and docking poses.
