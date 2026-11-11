"""Every setting of a self-similarity study, in one place.

House style follows ``_david/code/config.py``: one frozen dataclass, one line of comment per
field, alternatives written as ``# ^[a, b, c]``. Override on the command line with
``--set name=value``; the resolved config is written to the study directory as
``config.json``.

The question this study answers is narrow and worth stating: **given a molecule, how well
does a conformer ensemble generated from its SMILES recover its own shape, and what does
that cost?** It is self-recovery -- ensemble reproducibility -- and it is explicitly *not*
agreement with a bound or crystal conformation. Nothing here measures biological accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from _david.code.config import EGCG, PROJECT_ROOT


#: A generation method: which force field relaxes the embedded conformers, and under what
#: electrostatics. ``force_field=None`` is ETKDG alone, with no minimisation at all -- the
#: control the other rows are measured against.
@dataclass(frozen=True)
class Method:
    force_field: str | None  # ^[None, UFF, MMFF94, MMFF94s]
    dielectric: float = 1.0  # ignored by UFF, which has no dielectric term
    dielectric_model: int = 1  # ^[1, 2] 1 = constant, 2 = distance-dependent

    @property
    def label(self) -> str:
        if self.force_field is None:
            return "ETKDG"
        if self.force_field == "UFF":
            return "UFF"
        return f"{self.force_field} e{self.dielectric:g}" + (
            "r" if self.dielectric_model == 2 else ""
        )


#: The methods the study compares. MMFF94s at eps=78.4 is the pipeline's own setting
#: (`_david/code/scorers/conformers.py`), so it is the row every other row is read against.
#: eps=1 is the gas-phase control that `DECISIONS.md` cites [Gur17] against; MMFF94 is the
#: unplanarised variant MMFF94s replaced; UFF is the force field [Kan17] ranked worst.
METHODS: dict[str, Method] = {
    "ETKDG": Method(force_field=None),
    "UFF": Method(force_field="UFF"),
    "MMFF94_e1": Method(force_field="MMFF94", dielectric=1.0),
    "MMFF94_e78.4": Method(force_field="MMFF94", dielectric=78.4),
    "MMFF94s_e1": Method(force_field="MMFF94s", dielectric=1.0),
    "MMFF94s_e78.4": Method(force_field="MMFF94s", dielectric=78.4),
    "MMFF94s_e78.4r": Method(force_field="MMFF94s", dielectric=78.4, dielectric_model=2),
}


#: The force field a reference pool is energy-ranked with when the method has none of its
#: own. ``_david/code/scorers/conformers._force_field`` is not gated on ``cfg.minimize``, so the
#: pipeline ranks even an *unminimised* bank by MMFF94s energy. Ranking the ETKDG row by
#: nothing anchored its bank on embedding order instead, and moved the headline number.
RANKING_FALLBACK = METHODS["MMFF94s_e78.4"]


@dataclass(frozen=True)
class StudyConfig:
    smiles: str = EGCG  # the molecule recovered against itself; any SMILES
    name: str = "egcg"  # label for the study directory

    # the grid
    methods: tuple[str, ...] = (
        "ETKDG",
        "UFF",
        "MMFF94_e1",
        "MMFF94s_e1",
        "MMFF94s_e78.4",
    )  # ^ keys of METHODS
    reference_caps: tuple[int, ...] = (1, 2, 4, 8, 16)  # K: conformers kept in the bank
    candidate_counts: tuple[int, ...] = (10, 20, 25, 30, 40)  # C: candidate embed attempts
    repeats: int = 10  # independent seeds per cell; the spread across them is the result

    # reference bank construction, matched to the pipeline's own rules
    reference_attempts: int = 200  # conformers embedded before K are kept
    reference_rmsd: float = 0.5  # heavy-atom RMSD below which a pose is not a new one
    max_isomers: int = 4  # cap on enumerated unassigned stereocentres
    max_iterations: int = 500  # force-field iteration cap per conformer
    prune_rms: float = 0.5  # ETKDG's own pruning, native's value

    # scoring, matched to the pipeline
    alignment_shape_weight: float = 0.5  # AlignMol opt_param
    color_ff: str = "fdef"  # ^[fdef, oe_mills_dean, default] colour typing

    # execution. There is deliberately no `seed` field: the repeat seeds are a fixed,
    # documented list of primes in `study.SEEDS`, chosen so the streams cannot collide.
    # A knob recorded in config.json that changed nothing is worse than no knob.
    n_jobs: int = -1  # -1 = all cores; one cell per worker, RDKit single-threaded inside
    output_root: Path = PROJECT_ROOT / "_david" / "output" / "conformer"

    def method(self, key: str) -> Method:
        try:
            return METHODS[key]
        except KeyError:
            raise ValueError(
                f"unknown method {key!r}; known methods are {', '.join(sorted(METHODS))}"
            ) from None

    def __post_init__(self) -> None:
        for key in self.methods:
            self.method(key)
        if not self.methods:
            raise ValueError("at least one method is needed")
        if any(k < 1 for k in self.reference_caps):
            raise ValueError(f"reference_caps must all be >= 1, got {self.reference_caps!r}")
        if any(c < 1 for c in self.candidate_counts):
            raise ValueError(f"candidate_counts must all be >= 1, got {self.candidate_counts!r}")
        if self.repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {self.repeats!r}")
        if max(self.reference_caps) > self.reference_attempts:
            raise ValueError(
                f"reference_caps asks for {max(self.reference_caps)} conformers but only "
                f"{self.reference_attempts} are embedded"
            )
        if not 0.0 <= self.alignment_shape_weight <= 1.0:
            raise ValueError(
                f"alignment_shape_weight must lie in [0, 1], got {self.alignment_shape_weight!r}"
            )

    @property
    def cells(self) -> int:
        """How many measurements the grid asks for, before anything is run."""
        return (
            len(self.methods) * len(self.reference_caps) * len(self.candidate_counts) * self.repeats
        )
