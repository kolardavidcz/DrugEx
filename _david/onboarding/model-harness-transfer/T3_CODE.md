# DrugEx in T3 Code

Selected by the user on 29 October 2026. T3 Code now owns ongoing research
conversation and continuation; the existing WSL/lab checkouts remain in place.

## Project and execution

- Registered T3 project: **DrugEx**, ID `mcp:b9836007-9c06-4418-aea4-96651c3f638d`.
- Project folder: `C:\Ubuntu\home\kolar\DrugEx`, the Windows bridge to
  `/home/kolar/DrugEx` in Ubuntu WSL.
- Default thread environment: `local`, using the existing checkout.
  Automatic pulling is off. No execution worktree was created.
- Compute stays at `/home/kolarv/DrugEx` through the existing `dichtator` SSH
  alias. Use explicit WSL commands from the Windows harness.
- Current provider/model settings are preserved; no new API account or model
  selection was introduced for the migration.

Read `AGENTS.md`, `_david/AGENTS.md` and current `results.md` first. Personal skills
are already available to the Codex provider; project science skills remain in
`.agents/skills/`. Explicitly read a needed skill if automatic discovery differs
between providers. Do not copy old Antigravity global rules over this guidance.

## Existing conversation

The takeover thread is `ff45f87b-9ad4-4e00-a8e1-322669be0217`. It originally
started in T3's **No project** scratch workspace. The exposed thread API cannot
rebind an existing thread's project/workspace. Its commands explicitly target
the real WSL checkout, so work and scheduled continuation can proceed there.
Open future DrugEx conversations in the registered **DrugEx** project. If the
app UI offers moving this conversation, choose that project and its existing
checkout; do not create a worktree to simulate the move.

## Continuation and artifacts

The old Codex hourly automation was paused by the user. The native T3 task
**DrugEx calibration continuation (T3 takeover)** posts into the takeover thread
every 30 minutes. Its ID is
`scheduled-task:command:mcp:eeaa35c7-79a1-49e4-b8b0-6a87183f3b94:schedule-task:drugex-takeover-01a17a3a-20261029-v1`.
Read its live state through T3 tools; do not infer delivery from this dated note.

Latest steering holds **all new experiments, including the pending RL timing
illustration**, while the input-stereochemistry proposal is discussed. Finish
validation/analysis/reporting of the already-running calibration with its recorded
stereo-preserving inputs. Do not silently relabel that result as stereo-masked.
Pause the follow-up after the current delivery, or when further work requires
the user's scientific choice.

Use ordinary Markdown, PNG/SVG, CSV and SDF artifacts in the existing experiment
folder. Antigravity carousel/sidecar URLs are historical UI references. Retrieve
literature through existing free tools/APIs and retain access/source evidence.
Native T3 scheduling replaces the former app's automation for this task; browser
sessions, credentials and old pending timers are not copied.

At every compute launch, inspect live CPU/RAM and both GPUs' utilization,
processes and free memory. Prefer an available idle GPU for the RNN, recording
physical-to-logical CUDA mapping. RDKit/ROCS retains its CPU path. Current
hardware availability is an observation, not a permanent allocation.
