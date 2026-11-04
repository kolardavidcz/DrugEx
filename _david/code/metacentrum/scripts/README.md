# Historical PBS runners (`_david/code/metacentrum/scripts/`)

Frozen, one-off submission scripts from the H1–H9 sprint, the EGCG K=4/C=20
comparisons and the supervisor replication. They are kept because each one
records the exact flags and the PBS job id that produced a thesis run, not
because they should be executed again.

Submit through `grid.py` instead:

```bash
python -m _david.code.metacentrum.grid sync
python -m _david.code.metacentrum.grid submit --name <unique-name> --args "<pipeline flags>"
```

Why not re-run them as-is:

- they duplicate `job_template.sh` and have drifted from it (including their own
  `SOURCE_DIR`/`RESULTS_DIR` resolution);
- they write to `_david/output/<exp_name>` without the `YYYYmmdd_HHMM_` prefix;
- they pin historical defaults that the gold standard has since changed.

If one idea is revived, copy its `--args` into `grid.py submit`, not the script.
