# Historical model pivot

This note summarizes the transition recorded on 2026-10-25 from fragment-conditioned GraphTransformer experiments to unprompted SequenceRNN work. The original narrative is preserved in Git at fb1d1e4. It is historical context, not a cleanup or training instruction.

## Reason recorded for the change

Supplying fragments from EGCG constrained generated candidates around the supplied substructures. The thesis direction shifted toward sampling from the Papyrus SMILES prior without query-derived fragment prompts, using the query through the scoring environment instead.

This distinguishes the experimental setups. It does not establish that fragment-conditioned design is generally invalid, that RNN output is unbiased, or that a learned prior has no structural preferences. Supervisor feedback in earlier notes is an unverified paraphrase unless confirmed by the user.

## Retained comparisons

Historical GraphTransformer examples are under [legacy/stupid_graph_transformer](legacy/stupid_graph_transformer/): run01_baseline, run14_harrington_decay, and run15_harrington_deep_sampling. Earlier references to legacy/runs are obsolete.

Those runs used different model conditioning, scoring implementations, and configurations. Treat their metrics as historical records; comparisons with current SequenceRNN results require matched protocols and score scales.

## Current implementation

The active source is code/config.py, environment.py, models.py, pipeline.py, and visualization.py. The current default reward is weighted_sum; Harrington is an available alternative, not the deployed default merely because it was discussed in old notes.

Conformer curation, alternative scalarization, exploration schedules, and diagnostics are implemented work. Their presence does not establish novelty, physical accuracy, or scientific superiority. Such claims need code evidence, controlled comparisons, and primary literature.

Read [AGENTS.md](AGENTS.md) and [CODEX_ONBOARDING.md](CODEX_ONBOARDING.md) for current operation. Do not repeat historical instructions to purge outputs, remove checkpoints, or launch a matrix.
