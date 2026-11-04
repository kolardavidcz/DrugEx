# Archived Legacy Tests (`_david/code/tests/archive/`)

Historical tests from previous iterations of the codebase. They are **not collected** by
`python -m unittest discover -s _david/code/tests` (there is no `__init__.py` here) and they
are **not maintained**: several import modules that no longer exist, including
`_david.code.conformer.sampling` (conformer v1) and `_david.code.metacentrum.grid`
(deleted with the grid/sprint machinery). Keep them as provenance, do not "fix" them to make
them importable again.

Active, maintained tests live directly in `_david/code/tests/`.
