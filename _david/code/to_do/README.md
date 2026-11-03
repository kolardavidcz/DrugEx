# Experimental & Archived Modules (To-Do / Staging)

This directory stores experimental, exploratory, and non-authoritative prototypes that are preserved for research and comparison, but kept out of the production training pipeline.

---

### Contents

- **`rocs_multimetric.py`**: Experimental 162-combination multi-metric ROCS scoring system.
  - Contains experimental Tversky scoring variants (`tversky_combo`, `tversky_subshape`, `tversky_ref`, `shape_only`, `color_only`), custom Gaussian re-weighters, and `ConfigurableRDKitROCSScorer`.
  - Archived here so that the production baseline in `_david/code/rocs.py` remains lean, elegant, and strictly focused on canonical **`TanimotoCombo`** ($S + C \in [0, 2]$).
