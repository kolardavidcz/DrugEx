"""Ligand-based de novo design on top of DrugEx: the thesis pipeline.

DrugEx core under ``drugex/`` is read-only. This package adds exactly two things to it:

1. force-field relaxation of 3D conformers (``conformers``), which the native generators
   do not do, applied identically to the reference bank and to RL candidates;
2. an explorer that keeps two checkpoints instead of one (``explorer``).

Everything else -- conformer embedding, protonation, stereoisomer enumeration, ROCS
shape/colour scoring, Pareto rewards, the policy gradient -- is native DrugEx, reached
through its own constructors and configured from ``config.Config``.
"""
