# 04 · Molecule preparation and the canonical data cache

**Question.** What exactly is in the input set, what does standardization do to
it, and how many conformers survive each expensive stage?

**Deliverable.** Tables, not a figure. This component is the shared substrate the
other three read: the molecule registry, the conformer-survival audit, and the
stereoisomer-truncation audit.

**Kernel.** `prepare/standardize.py` (`prepare_molecule`) and
`analysis/preparation.py` (`build_registry`, `reference_count_summary`,
`isomer_audit`).

---

## `molecule_registry.csv` — 527 inputs, one row each

Produced by `python -m _david.code.conformer.cli registry` (also rebuilt by
`cli figures`, which caches it). Every row records the input SMILES, the
standardized SMILES, the enumerated stereoisomers, descriptor values, and any
warning raised while preparing it.

| Property | Value |
|---|---|
| Molecules | 527 (512 DrugCentral: 128 development + 384 confirmation; 15 EGCG stress-panel members) |
| Preparation failures | **0** |
| Heavy atoms | 3–51, median 23 |
| Molecular weight | 44–1140 Da, median 337 Da |
| Formal charge | 512 neutral, 14 at +1, 1 at +2 |
| Molecules with an unassigned stereocentre | 98 |
| Enumeration cap used for the audit | 32 |
| Inputs hitting that cap | **0** |

Reachable stereoisomer identities per input (only unassigned centres vary):

| Identities | 1 | 2 | 3 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| Molecules | 416 | 92 | 1 | 14 | 4 |

Nothing is silently dropped. Five molecules below needed a disclosed
intervention; the rest prepared cleanly:

| Warning | Count | Meaning |
|---|---:|---|
| `tautomer-or-protonation-state-changed` | 73 | Standardization changed the chemical state; the row keeps the standardized structure |
| `chemical-state-fallback: … loses assigned stereochemistry` | 4 | Tautomer/protomer normalization would have destroyed an assigned centre, so the input state was kept instead |
| `symmetry-collapsed-4-to-2` / `-to-1` / `-to-3` | 5 | RDKit's naive upper bound ignored symmetry; enumeration correctly produced fewer distinct identities |
| `no-shape-color-features` | 4 | No N/O atoms, so the colour term has nothing to match |

The distinction between the last two matters: a collapse is *correct chemistry*,
truncation is *a budget cutting the set*. Only truncation gets an
`isomer-cap-N-truncated-M` warning, and none occurred at cap 32.

### What standardization actually does

1. **SMILES → RDKit mol**, rejecting anything RDKit cannot parse.
2. **Standardize** (the same routine the RL pipeline uses), then
   **tautomer/protomer canonicalization** at physiological pH.
3. **Refuse to lose stereochemistry.** If step 2 would drop an assigned centre,
   the input state is restored and a fallback warning is recorded. Silently
   scoring the wrong stereoisomer is a worse outcome than scoring the input's.
4. **Enumerate only unassigned centres** (`onlyUnassigned=True`), so a fully
   specified drug stays one identity and an ambiguous input becomes 2/4/8.
5. **Record descriptors** on the heavy-atom graph: HAC, MW, rotatable bonds,
   aromatic rings, HBD/HBA, stereo centres (assigned/unspecified), formal charge,
   QED, SA, fraction Csp3.

Preparation is cheap and needs no 3D conformer, so registering a new molecule is
safe to repeat and is cached by `(standardized smiles, EmbedSpec)`.

---

## `reference_counts.csv` and `reference_count_summary.csv`

Per molecule × phase × method: how many conformers were generated, how many a
force field converged, and how many survived the energy window.

| Phase | Method | Generated | Survived | Survival |
|---|---|---:|---:|---:|
| development (128) | ETKDG | 65.6 | 65.6 | 1.000 |
| development | MMFF94s ε=78.4 | 65.6 | 54.1 | 0.824 |
| development | MMFF94s ε=1.0 | 65.6 | 48.7 | 0.743 |
| development | UFF | 65.6 | 50.4 | 0.769 |
| confirmation (384) | MMFF94s ε=78.4 | 62.6 | 53.8 | 0.859 |
| confirmation | MMFF94s ε=1.0 | 62.6 | 48.1 | 0.769 |

Means per molecule over 100 embedding attempts. Force-field convergence is
essentially 1.0 — the loss is entirely the energy window, which is why the
window is the dial that actually controls bank size (component 1, side quest).
The `preparation_failures` column separates chemical failures (5 and 8) from
conformer failures so neither disappears into a mean.

---

## `isomer_audit.csv` — where enumeration would explode

512 rows (128 development + 384 confirmation), 124 flagged `truncated` under the
historical `I=8` protocol.

The audit reports both bounds because they answer different questions:

- `original_upper_bound` — identities for the input **as specified** (usually 1:
  a drug SMILES already names its stereoisomer).
- `masked_upper_bound` — identities if **every** centre were unassigned, i.e. the
  worst case an RNN candidate presents. Aminoglycosides and macrolides reach
  10⁵–10⁶ here, which is exactly why an enumeration cap is mandatory rather than
  optional.

The component-1 stratification figure bins on `masked_upper_bound` because the
historical runs used the masked protocol; new study tables record
`isomer_upper_bound` for whatever the study actually enumerated. Both feed the
same six bins (`core/taxonomy.py`), so the two remain comparable.

---

## Where the reusable data lives

```
_david/data/conformers/
├── cache/                      content-addressed: prepare, embed, minimize, select, score
└── canonical/
    ├── tables/                 10 registered tables — the source of every figure
    ├── panels/                 drug_panel_development (128), drug_panel_confirmation (384), egcg (16)
    └── migration_sources.json  which historical run each table came from, with SHA-256
```

The cache key is
`sha256(ALGORITHM_VERSION ‖ namespace ‖ canonical_json(spec))`, so:

- re-running a study reuses conformers instead of re-embedding;
- a changed force field, seed, cap or score spec cannot silently hit an old
  entry;
- deleting the cache costs time, never correctness.

`python -m _david.code.conformer.cli cache-stats` prints occupancy per namespace.
Registry entries are the only artifacts kept indefinitely: they are the audit
trail for what the pipeline thinks each input molecule is.

---

## Regenerate

```bash
# registry only (fast; no conformers)
python -m _david.code.conformer.cli registry

# registry + all component tables and figures
python -m _david.code.conformer.cli figures --canonical-only
```

A run that succeeds deletes any stale `registry_errors.csv` from a previous
failure, so the presence of that file always means a real, current failure.
