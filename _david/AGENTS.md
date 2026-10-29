# Thesis implementation guide

## Ownership and scope

The user owns the bachelor's thesis and its scientific decisions. Debate methods
candidly, challenge unsupported assumptions, and implement the choices the user
makes. Discuss new scientifically consequential settings before adding them to a
study. Research discussion alone does not authorize experiments.

Keep all thesis implementation under `_david/`. DrugEx core and other source
paths are read-only. Keep code simple, readable and easy to review; refactor only
as necessary for the requested change. Preserve the user's annotations in
`code/config.py`. Before adding functionality, inspect DrugEx, then established
libraries such as RDKit.

## Molecule-agnostic pipeline

References and candidates are inputs, not named special cases. EGCG is the
current configurable default reference for unprompted Papyrus SMILES SequenceRNN
generation, not a fragment prompt. Do not specialize preparation, conformer
budgets, identity checks, visualization or success criteria to EGCG or another
named molecule. A single-reference experiment cannot establish generality.

Apply the same explicit chemistry policy to every supported input. Preserve
specified stereochemistry; distinguish unresolved stereochemistry from a
preparation error. Report unsupported chemistry, missing force-field parameters,
embedding failures and nonconvergence explicitly. Molecule-agnostic does not
promise support for every chemical structure.

The conformer study must use the same candidate-preparation/scoring path and
isomer policy as training. Deep reference optimization and candidate preparation
serve different roles; document that distinction. Any alternative candidate path
is a labelled comparison, not evidence that training already uses it. Keep
isomer and conformer limits independently configurable; coupling them is a
proposal until selected. Report actual enumerated/embedded/retained counts.

For self-recovery, every parent has its own independent reference ensemble.
Different candidates scored against one common query answer a different question;
make that comparison explicit before interpreting scores or launching a large run.

## Science and reporting

Conformer calibration serves a dependable, affordable ligand-based RL reward.
Do not expand it into physical-pose or solvent-population simulation unless the
user chooses that research question.

Use [consult-literature](../.agents/skills/consult-literature/SKILL.md) before
choosing, recommending or changing research methods, consequential parameters,
datasets or evaluation. Use free primary literature and inspect the actual
implementation. Routine software fixes preserving scientific behavior do not
need a new literature review.

Measure unfavorable outcomes as well as promising ones. Include relevant
diagnostics in `code/visualization.py`. State score definitions and transformations
explicitly; never infer their scale from observed maxima. Scoring/conformer
experiments do not establish RNN training improvements.

When a consequential result or method remains unclear to the user, provide a
readable Markdown explanation with a concrete molecular comparison, a diagram or
table when useful, and a clear distinction between measured facts and hypotheses.
Keep results.md as the maintained entry point and link requested dated artifacts.
Use consistent terms for an input molecule, stereoisomer, conformer, embedding
request and retained reference. Explain the practical difference between similar
technical names before asking for a choice; do not infer a method change from
a spelling variation when an existing choice is already recorded.
Lead method discussions with one evidence-based recommendation and its reasons.
Before proposing a parameter grid, identify the deployed quantity and how often
its cost is paid. Reference preparation is a one-time cost; retained reference
count controls repeated alignment work. Embedding requests are not a substitute
for the retained count. Reuse saved measurements before launching more geometry
generation, and explain what a comparison can actually decide.
Ask for a choice when a material scientific trade-off needs the user's judgment;
do not hand back a list of unexplained technical options.

This file is the single source of current project instructions.
[code/conformer/results.md](code/conformer/results.md) is the single maintained findings/limitations/decision
record. Mark proposals, selected methods and measured results separately. Keep
the conformer delivery compact: one score/time figure, reproducible scripts and
raw data referenced from results.md. Frozen historical evidence, raw data and
Git history are provenance, not additional live guidance. Do not revive their
action lists or settings.

## Code and execution

- `code/config.py`: run configuration and checkpoint resolution.
- `code/environment.py`: reference preparation, conformer/scorer construction,
  modifiers and rewards.
- `code/models.py`, `code/pipeline.py`: agent/frozen-prior loading and RL workflow.
- `code/schema.py`, `code/utils.py`: score columns and runtime configuration.
- `code/visualization.py`: analysis and figures; `tests/`: executable assertions.
- `code/conformer/`: conformer study and maintained results; generated evidence
  belongs under `output/YYYYMMDD_HHMM_short_description/`, separate from source.
  Put the report and final figures at the experiment root, measurements and
  intermediate data in `raw_data/`, and execution logs in `logs/`.
- `legacy/`: fragment-conditioned historical comparisons; `output/`: generated
  artifacts. Ignored data/checkpoints are not transferred by Git.

Edit in local WSL `/home/kolar/DrugEx`; execute computation with explicit SSH
commands in `/home/kolarv/DrugEx` through alias `dichtator`. All thesis source
must remain in the main checkout's `_david/code/`, and all generated work under
`_david/output/`. Do not create execution worktrees, source copies or experiment
data in `.cache`, another checkout, or another home-directory path. Preserve
provenance with Git revisions, source hashes and archived source under the
experiment's `raw_data/`; check relevant hashes before and after computation.
Set temporary directories inside the experiment's `raw_data/` as well.
Follow the personal `drugex-dichtator` skill within these project constraints.
Inspect both hosts before synchronization and preserve unrelated/user edits.
Commit only task files and synchronize through private `dichtator`; never push
to GitHub `origin` or force push. Do not commit credentials, checkpoints or bulky
output. Use LF endings and the existing remote `.venv`.

Validate the relevant source, including imports from the main checkout.
Record host, revision/dirty state, exact command and result for substantive
validation. When relevant source is dirty or concurrent edits are possible,
compare its hashes before claiming remote validation. Reuse existing experiment
manifests; record configuration, input identities, dependency versions, seeds,
sample denominators, failures and score scales. No additional validation framework.
