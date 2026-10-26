# Previous-agent handoff: source register

Imported on 2026-10-26 from the user's Desktop export. The original export is preserved in Git at commit fb1d1e4, path _david/GEMINI_HANDOFF.md. This revision removes obsolete tool instructions, incorrect metrics, and unsupported operational recommendations. Read [CODEX_ONBOARDING.md](CODEX_ONBOARDING.md) and [AGENTS.md](AGENTS.md) for current guidance.

## Research context carried forward

- Thesis: RNN ligand-based de novo design in DrugEx, with EGCG as the default scoring reference.
- Current model: unprompted Papyrus SMILES SequenceRNN and an independent frozen mutation prior.
- Objectives under investigation: ROCS shape/color similarity, SA, QED, exploration schedules, and policy regularization.
- The standalone conformer study compares force-field/dielectric variants and requested reference/probe ensemble sizes.
- The stopping point was discussion of how that study should translate into defensible training settings, including missing candidate latency measurements.
- The user wants candid academic critique, readable class-oriented code, preservation of config annotations, and unfavorable diagnostics.

These are inherited context, not permission to select parameters or start experiments. Supervisor quotations in the original export have not been independently verified.

## Verified state and corrections

Code baseline at handoff was 9261af8. Onboarding documentation was committed afterward as fb1d1e4. Current source and live Git state take precedence over those snapshots.

- Run 17 and Run 18 have 50 recorded epochs and saved baseline/reinforced evaluation CSVs.
- Run 19 has 18 recorded epochs and a checkpoint, with no completed reinforced evaluation found. No Run 20 output directory was found in the SSH survey.
- Original handoff validity claims of 99.8-100% do not match saved evaluation rows: Run 17 reinforced validity is 88.3%; Run 18 is 84.5%.
- ROCS report values cannot be compared without tracking scale: current reporting can apply the factor-of-two conversion twice.
- Config timeout_seconds=0 is not wired into the core generator constructor; it still has its default timeout.
- The standalone pocket variant is not automatically used by the RL scoring path.
- MetaCentrum SSH authentication failed at inspection. Its grid CLI has no submit action; old command examples were invalid.
- Suggested K=10, a 100-epoch production matrix, resource requests, and specific benchmark sample sizes remain proposals.

## Source reports

- [Conformer study](onboarding/gemini-reports/conformer_convergence_and_dielectric_study.md)
- [Run 17](onboarding/gemini-reports/run17_50epoch_baseline_evaluation_report.md)
- [Run 18](onboarding/gemini-reports/run18_50epoch_pure_reinforce_evaluation_report.md)
- [Historical model pivot](THESIS_PIVOT_REVIEW.md)

For future literature claims, use primary papers actually inspected, state full-text versus abstract-only access, and distinguish preprints from peer-reviewed articles. Old agent explanations and citation counts are not evidence of methodological correctness.
