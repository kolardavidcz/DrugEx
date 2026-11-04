# 05 — Resources: cores, memory, walltime, queues

This workload is unusual for a cluster: the neural network is tiny, while **ROCS scoring and
conformer generation are CPU-bound and dominate the wall clock**. GPUs help the RNN only, so
a GPU allocation is usually a waste of a scarce queue slot unless the user asks for it.

## 1. Current default

`submit.py` sends, unless told otherwise:

```
select=1:ncpus=16:mem=32gb:scratch_local=15gb:spec=8.0
walltime=24:00:00
queue=p2e_1d, ngpus=0
```

That is the configuration under which the 200-epoch, 30-conformer, unminimised arms
(`sprint_h1`…`h4`) completed in under 24 h. Treat any deviation as a deliberate,
documented change.

## 2. Queues (observed 2026-11-04)

| Queue | Max walltime | Running | Queued | Use for |
| :--- | ---: | ---: | ---: | :--- |
| `p2e_1d` | 24 h | 705 | 4 | **Default.** Fast admission, 24 h budget. |
| `q_4h` | 4 h | 471 | 28 | Smoke tests and short parameter checks. |
| `q_1d` | 24 h | 1581 | 3873 | Alternative to `p2e_1d`; much deeper queue. |
| `q_2d` | 48 h | 267 | 4986 | Only when 24 h is provably insufficient. |
| `runone` | 24 h | 0 | 22 | Interactive/debug single job (1 per user). |
| `interactive` | 48 h | 18 | 1 | Debug shells; not for batch runs. |
| `gpu` | 48 h | 160 | 2010 | Avoid: multi-thousand queue. |
| `gpu_long` | 336 h | 44 | 43 | The only sane GPU queue. |
| `large_mem` | 168 h | 10 | 55 | >1 TB RAM nodes. |
| `default` | 720 h | 0 | 9 | Not ours; do not use. |

Queue depths move hourly; the numbers above are context, not a contract. Check before
choosing:

```bash
ssh meta 'qstat -q | grep -E "^(Queue|p2e_1d|q_4h|q_1d|q_2d|gpu_long|runone)"'
```

`p2e_1d` is preferred over `q_1d` for the same walltime because its queue is two to three
orders of magnitude shorter. `runone` is useful for a 1-epoch smoke test: one job at a time,
starts almost immediately, but it will not accept a second concurrent job.

## 3. Cores

`PipelineConfig.cpu_ratio = 0.99` means the pipeline derives its conformer worker count from
the cores PBS granted, so `--ncpus` is the only knob that matters.

| Situation | `--ncpus` |
| :--- | ---: |
| Smoke test (`--epochs 1 --samples 10`) | 4 |
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
| `--mem 8` | smoke tests, 4 cores |
| `--mem 32` | 16 cores, 20-30 conformers, EGCG-scale ligands |
| `--mem 64` | 32 cores, or ligands with many rotatable bonds |
| `--mem 128` | very large ligands / minimised runs with many workers |

Nodes range from small legacy machines to 128-core, ~1 TB nodes; a 32 GB request is easy to
place, a 128 GB request narrows the pool. PBS kill by the OOM killer is `SIGKILL`
(`Exit_status = -9`), with no chance for the EXIT trap's graceful message — chapter 04 §6.

## 5. Scratch

`scratch_local=15gb` holds:

- the copied release (~23 MB),
- the run directory (per-epoch/reinforced SDFs for up to 5000 evaluation molecules, figures),
- per-molecule temporary conformer files during embedding.

Rules of thumb:

| Workload | `--scratch` |
| :--- | ---: |
| Smoke test | 5 |
| Default EGCG run, C=20, 5000 eval | 15 |
| Minimised, C=40, or large ligand | 30 |
| Several thousand minimised conformers | 50 |

Scratch is local disk on the worker, wiped by the template after the copy-back. It is **not**
a place to keep results; only `$STORAGE` survives the job.

## 6. Walltime arithmetic

Measure the epoch rate, then multiply:

```
epochs_that_fit = floor(walltime_hours * 60 / minutes_per_epoch)
```

From the historical runs: 200 epochs × 500 samples with `--no-minimize-candidates`,
C=30, 16 cores finished inside 24 h. Minimisation costs roughly **4×** more (conformer
embedding with MMFF94s, dielectric 78.4), so a minimised 200-epoch run does not fit in
`p2e_1d`; use `--walltime 48:00:00 --queue q_2d` or reduce `--epochs`.

Practical guidance:

- Start from the default 24 h. If `monitor_live_epochs.py` shows an ETA beyond the limit,
  kill early and resubmit with the right budget — a killed job still consumes its walltime.
- `--epochs 200` is a **cap**: early stopping (`min_epochs=50`, `patience=50`) usually ends a
  converged run before 200.
- Never request more walltime than the queue's maximum; `qsub` rejects it at submission.

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
- A 1-epoch smoke test before a 24-hour production run is cheap insurance and is the
  recommended pattern after any code change.
