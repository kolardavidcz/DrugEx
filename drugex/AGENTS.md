# DrugEx core boundary

For this thesis, core source is read-only. Implement approved adaptations under
`_david/`, reusing DrugEx and RDKit APIs where possible. The authoritative project
instructions are [../_david/AGENTS.md](../_david/AGENTS.md); verified issues and
research status are in
[../_david/code/conformer/results.md](../_david/code/conformer/results.md).

Inspect core implementations when tracing behavior: molecular converters are in
`molecules/converters/`; conformer generation and ROCS scoring in
`training/scorers/`; RL generation/exploration in `training/generators/` and
`training/explorers/`. Molecular identity and parameter forwarding must be
verified from the actual path used by training, rather than inferred from names
or standalone-study results. No molecule-specific thesis behavior belongs here.
