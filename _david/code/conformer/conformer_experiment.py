"""Compatibility entry point for the current conformer study.

Use python -m _david.code.conformer.conformer_experiment --help.
The implementation and supported arguments live in study.py.
"""

from _david.code.conformer.study import main


if __name__ == "__main__":
    main()
