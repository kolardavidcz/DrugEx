"""Everything that reads a finished run directory and writes figures or a report.

Nothing in here is imported by the pipeline while it trains, so a missing matplotlib can
never fail a run.
"""

from _david.code.analysis.report import analyse  # noqa: E402,F401
