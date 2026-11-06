# 05 — Resources: cores, memory, walltime, queues

This workload is unusual for a cluster: the neural network is tiny, while **ROCS scoring and
conformer generation are CPU-bound and dominate the wall clock**. GPUs help the RNN only, so
a GPU allocation is usually a waste of a scarce queue slot unless the user asks for it.

## 1. Current default

`submit.py` sends, unless told otherwise:

```
select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0
walltime=24:00:00
(no -q: 24 h routes to q_1d, the default and only normal lane)
ngpus=0
```

That is the configuration under which the 200-epoch, 30-conformer, unminimised arms
(`sprint_h1`…`h4`) completed in under 24 h. Treat any deviation as a deliberate,
documented change.

## 2. Queues and routing

**You cannot name an execution queue.** Verified 2026-11-04 from `tarkil` (the host behind
`ssh meta`):

| Attempt | Result |
| :--- | :--- |
| `qsub -q p2e_1d …` | `qsub: Access from host not allowed, or unknown host` (`acl_hosts = pbs-m1.metacentrum.cz`) |
| `qsub -q q_2h` / `q_4h` / `q_1d` / `q_2d` | `qsub: Access to queue is denied` (`from_route_only = True`) |
| `qsub …` with no `-q` | accepted, then routed by walltime |
| `qsub -q default …` | accepted, same routing |

So the queue is a **function of `--walltime`**, through the `default` router
(`route_destinations = q_2h,q_4h,q_1d,q_2d,q_4d,q_1w,q_2w,q_2w_plus,gpu,gpu_long`):

| `--walltime` up to | Routed queue | Observed depth (2026-11-04) |
| :--- | :--- | :--- |
| 2 h | `q_2h` | shallow |
| 4 h | `q_4h` | 471 running / 28 queued |
| 24 h | `q_1d` | 1581 running / 3873 queued |
| 48 h | `q_2d` | 267 running / 4986 queued |
| 1 week | `q_1w` | deep |
| 2 weeks | `q_2w` | deepest |

`p2e_1d` (705 running / 4 queued at the same time) is a fast lane we cannot address
directly. Do not try to force it: a direct `-q p2e_1d` fails at submission.

Practical consequences:

- **One lane: 24 h → `q_1d`.** Every normal job asks for a full day and waits there. The
  queue is deep; that is the price of a budget that does not truncate the run, and
  `submit status` is how you measure the wait.
- **Walltime is a budget, not a queue knob.** Do not request 2 h to reach `q_2h`: the job is
  killed at the limit with partial results, and the time already spent queueing is wasted.
  Ask for the compute the scientific plan needs (24 h default, `48:00:00` → `q_2d` only for
  minimised 200-epoch sets that genuinely exceed a day).
- `--queue` remains as an escape hatch for a route queue or a special queue an administrator
  grants (`--queue gpu_long` for GPU work). It is never needed for a normal CPU run.

Queue depths move hourly; treat the numbers as context, not a contract:

```bash
ssh meta 'qstat -q | grep -E "^(Queue|q_2h|q_4h|q_1d|q_2d|p2e_1d|gpu_long)"'
ssh meta 'qstat -Qf default | grep route_destinations'
```

`runone` (interactive, one job at a time) and `interactive` exist but are not granted to
this account; the attempt returns `Access to queue is denied`.

## 3. Cores

`PipelineConfig.cpu_ratio = 0.99` means the pipeline derives its conformer worker count from
the cores PBS granted, so `--ncpus` is the only knob that matters.

| Situation | `--ncpus` |
| :--- | ---: |
| Default ROCS run | 16 |
| Minimised candidates, large ligand, or many conformers | 32 |
| Anything above 32 | Rarely worth it: ROCS is per-molecule serial, and the straggler molecule bounds each epoch. |

More cores shorten an epoch sub-linearly once the worker pool is saturated by a few
pathological molecules. They also increase memory pressure (§4) without improving the RNN.

## 4. Memory

Workers each hold an RDKit molecule plus conformers; minimisation holds force-field state.
Measured practice:

| Request | Fits |
| :--- | :--- |
| `--mem 32` | 16 cores, 20-30 conformers, EGCG-scale ligands |
| `--mem 64` | 32 cores, or ligands with many rotatable bonds |
| `--mem 128` | very large ligands / minimised runs with many workers |

Nodes range from small legacy machines to 128-core, ~1 TB nodes; a 32 GB request is easy to
place, a 128 GB request narrows the pool. PBS kill by the OOM killer is `SIGKILL`
(`Exit_status = -9`), with no chance for `job.py`'s graceful message — chapter 04 §6.

## 5. Scratch

`scratch_local=15gb` holds:

- the copied release (~23 MB),
- the run directory (per-epoch/reinforced SDFs for up to 5000 evaluation molecules, figures),
- per-molecule temporary conformer files during embedding.

Rules of thumb:

| Workload | `--scratch` |
| :--- | ---: |
| Default EGCG run, C=20, 5000 eval | 15 |
| Minimised, C=40, or large ligand | 30 |
| Several thousand minimised conformers | 50 |

Scratch is local disk on the worker, wiped by `job.py` after the copy-back. It is **not**
a place to keep results; only `$STORAGE` survives the job.

## 6. Walltime arithmetic

Measure the epoch rate, then multiply:

```
epochs_that_fit = floor(walltime_hours * 60 / minutes_per_epoch)
```

Measured rates for a minimised 500-sample epoch (MMFF94s, dielectric 78.4, C=20):
**~10.4 min at 16 cores** and ~5.1 min on a 48-thread workstation. So the 150-epoch default
needs ~26 h at 16 cores and does not fit a 24 h slot; the 2026-11-04/05 arms that asked for
200 were killed at epoch 132/137, i.e. the cap was never reachable. Three honest fixes, in
order of preference:

1. **More cores, same science** (`--ncpus 32 --mem 64`): the scoring step is the parallel
   part, and the wall clock halves while the molecule scores stay identical.
2. **A deeper queue** (`--walltime 48:00:00` → `q_2d`): wait longer, spend less allocation.
3. **Fewer epochs**: only if the horizon is a decided design choice, never just to reach a
   shallower queue.

Practical guidance:

- `q_1d` drains roughly twice as fast as `q_2d` (1435 vs 764 running jobs on 2026-11-06) and
  a submitted job waited ~6 h for a slot there, so buying cores usually beats buying days.
- `--mem` scales with `--ncpus` at ~2 GB per core; `JobSpec` rejects a starved pairing. The
  16-core arms reached their 32 GB ceiling, so 32 cores starts at 64 GB.
- Start from the default 24 h. If `monitor_live_epochs.py` shows an ETA beyond the limit,
  kill early and resubmit with the right budget — a killed job still consumes its walltime.
- `--epochs 150` is a **cap**: early stopping (`min_epochs=50`, `patience=30`) usually ends a
  converged run long before it.
- Walltime above the routed queue's maximum is rejected at submission; the router would have
  to pick a deeper queue anyway.

## 7. `spec` and GPU accounting

`spec=8.0` is a minimum SPEC CPU score per core. Its only effect is node filtering:

- `spec=10` restricts to `fobos`/`bee` → short list, long waits.
- `spec=8` keeps modern nodes and drops ancient ones.
- Omit `--spec` only when debugging hardware-specific behaviour.

For GPU jobs the same `spec` filters CPU generation, not GPU generation; use `--gpu-cap`
when a specific compute capability matters (e.g. `sm_89`). Remember `--ngpus 1` implies a
GPU queue; `submit.py` does not switch queues for you.

## 8. What a job costs

A job occupies `ncpus` cores for its whole walltime, so a 16-core, 20-hour run is 320
core-hours of a shared national resource. Consequences for how we operate:

- Do not leave a converged run queued for days; kill it.
- Do not submit four near-identical arms "just in case"; submit the one the user asked for.
- There is no rehearsal job. The launcher's environment check and `job.py`'s spec, release and
  scratch validation fail within seconds of the job starting, and `logs <name>` shows that
  immediately; a separate 1-epoch probe adds a queue wait without telling you anything the
  real job's first epoch does not.
