"""Every hyperparameter of a run, in one place.

House style: one frozen dataclass, one line of comment per field, alternatives written as
``# ^[a, b, c]``. No derived properties, no aliases, no auto-calibration that invents a
value, and no re-validation of anything DrugEx already validates -- two copies of a rule
drift. Override on the command line with ``--set name=value``; the resolved config is
written to the run directory as ``config.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# parents[2]: this file is _david/code/config.py, so the repo root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Epigallocatechin gallate. A default example, not a special case: the pipeline applies
#: the same preparation and scoring to any reference molecule.
EGCG = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"


@dataclass(frozen=True)
class Config:
    reference_smiles: str = EGCG  # the reference; its 3D conformer bank is generated from this

    # reinforcement learning
    epochs: int = 70  # hard cap; patience normally ends the run first
    patience: int = 20  # epochs without improvement before a criterion is done
    min_epochs: int = 40  # neither criterion may freeze or stop below this epoch
    n_samples: int = 1000  # molecules generated and scored per epoch for the policy gradient
    n_eval: int = 50  # held-out molecules scored each epoch to measure the stop condition
    batch_size: int = 128  # ^[64, 128] molecules per policy-gradient step
    epsilon: float = 0.0  # per-token probability of sampling from the frozen prior instead
    beta: float = 0.0  # reward baseline subtracted in the policy gradient
    scheme: str = "PRCD"  # ^[PRCD, PRTD, WS] DrugEx reward scheme; WS turns thresholds into weights

    # scoring
    alignment_shape_weight: float = 0.5  # ^[0.0, 0.25, 0.5, 0.75, 1.0] pose objective only
    color_ff: str = "fdef"  # ^[fdef, oe_mills_dean, default] -- "default" is NOT "fdef"
    include_qed: bool = False  # ^[False, True] adds QED as a third objective
    sa_reward_zero: float = 5.0  # raw SA at or above this scores 0.0
    sa_reward_one: float = 3.0  # raw SA at or below this scores 1.0
    fixed_thresholds: tuple[
        float, ...
    ] = ()  # ^[(), (1.0, 1.0)] per objective; overrides the percentile
    threshold_percentile: float = 85.0  # prior p85 for stop; WS also derives reward weights from it
    threshold_from_reference: tuple[str, ...] = ("SA", "QED")  # ^[(), ("SA",)] use EGCG's own value
    n_prior: int = 5000  # molecules sampled from the prior to calibrate the thresholds
    n_final: int = 5000  # molecules sampled from each checkpoint once the run ends

    # conformers and force field
    minimize: bool = True  # ^[True, False] relax reference AND candidates with MMFF94
    dielectric: float = 78.4  # ^[78.4, 6.5, 1.0] 78.4 = water
    dielectric_model: int = 1  # ^[1, 2] 1 = constant, 2 = distance-dependent
    max_iterations: int = 500  # force-field iteration cap per conformer
    max_conformers: int = 20  # ^[20, 30, 40] candidate conformers per stereoisomer
    max_isomers: int = 4  # cap on enumerated unassigned stereocentres
    max_heavy_atoms: int = 45  # candidates above this are skipped, scoring 0
    max_rotatable_bonds: int = 15  # candidates above this are skipped, scoring 0
    timeout: int = 30  # per-isomer ETKDG cap in seconds; 0 = unlimited
    reference_attempts: int = 200  # conformers embedded for the reference; DrugEx clamps above 200
    reference_conformers: int = 4  # ^[3, 4, 5] kept from the pool by geometric diversity
    reference_rmsd: float = 0.5  # angstrom; heavy-atom RMSD below this is a duplicate pose

    # execution
    seed: int = 42  # Python/NumPy/torch seed, recorded per run
    n_jobs: int = -1  # -1 = all cores; DRUGEX_SCORE_NJOBS caps it further
    output_root: Path = PROJECT_ROOT / "_david" / "output"

    def __post_init__(self) -> None:
        # Only rules nothing downstream enforces. color_ff is validated by
        # RDKitROCSScorer's constructor, so they are deliberately not checked again here.
        if self.dielectric_model not in (1, 2):
            raise ValueError(f"dielectric_model must be 1 or 2, got {self.dielectric_model!r}")
        if self.fixed_thresholds and any(
            not math.isfinite(value) for value in self.fixed_thresholds
        ):
            raise ValueError(f"fixed_thresholds must all be finite, got {self.fixed_thresholds!r}")
        if not 0.0 <= self.alignment_shape_weight <= 1.0:
            raise ValueError(
                f"alignment_shape_weight must lie in [0, 1], got {self.alignment_shape_weight!r}"
            )
        if not 0.0 < self.threshold_percentile < 100.0:
            raise ValueError(
                f"threshold_percentile must lie in (0, 100), got {self.threshold_percentile!r}"
            )
        if self.n_eval < 1 or self.n_samples < 1:
            raise ValueError(
                f"n_samples ({self.n_samples}) and n_eval ({self.n_eval}) must both be "
                "at least 1: n_eval is the arm the stop condition is measured on"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs!r}")
        if self.min_epochs > self.epochs:
            raise ValueError(
                f"min_epochs ({self.min_epochs}) exceeds epochs ({self.epochs}): the run could "
                "never reach its own early-stopping floor"
            )
        # Lower SA is better, so the modifier descends. ClippedScore only asserts
        # low_score < high_score, so an accidentally ascending pair would silently invert
        # the objective and reward unsynthesizable molecules.
        if self.sa_reward_one >= self.sa_reward_zero:
            raise ValueError(
                f"sa_reward_one ({self.sa_reward_one}) must be below sa_reward_zero "
                f"({self.sa_reward_zero}): lower SA is better, so the modifier descends"
            )

    def pretrained(self) -> tuple[Path, Path]:
        """Checkpoint and vocabulary of the pretrained SMILES RNN."""
        base = PROJECT_ROOT / "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        pkg, voc = base / f"{base.name}.pkg", base / f"{base.name}.vocab"
        missing = [str(p) for p in (pkg, voc) if not p.is_file()]
        if missing:
            raise FileNotFoundError("pretrained model not found: " + ", ".join(missing))
        return pkg, voc
