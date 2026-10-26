from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ForceFieldVariant:
    """Specification for conformer optimization force field and solvent dielectric environment.

    Attributes:
        name: Unique programmatic identifier (e.g., 'MMFF94s_Pocket').
        display_title: Publication-ready title for plots and visualizers.
        ff_type: Core force field engine ('mmff' or 'uff').
        dielectric_model: MMFF dielectric model (1 = constant macroscopic, 2 = distance-dependent screening).
        dielectric_constant: Relative dielectric permittivity (epsilon_r).
            1.0 = gas-phase / vacuum
            6.5 = protein binding pocket consensus (Warshel / Simonson)
            78.4 = bulk aqueous solution at 298 K
        energy_window_kcal: Maximum Delta E above lowest energy conformer. None disables window.
        rmsd_cutoff: Heavy-atom pairwise RMSD clustering threshold in Angstroms (default 1.25 A).
    """

    name: str
    display_title: str
    ff_type: str
    dielectric_model: int = 1
    dielectric_constant: float = 1.0
    energy_window_kcal: float | None = 8.0
    rmsd_cutoff: float = 1.25

    def is_solvent_shielded(self) -> bool:
        """Return True if the variant incorporates electrostatic charge screening."""
        return self.dielectric_constant > 1.0 or self.dielectric_model == 2


class PhysicalVariantFactory:
    """Factory for standard physiological and computational force field variants."""

    @staticmethod
    def get_benchmark_suite() -> Sequence[ForceFieldVariant]:
        """Return the authoritative 5-variant benchmark suite for conformer convergence analysis."""
        return (
            ForceFieldVariant(
                name="MMFF94s_Vacuum",
                display_title="MMFF94s (Vacuum, eps=1.0)",
                ff_type="mmff",
                dielectric_model=1,
                dielectric_constant=1.0,
                energy_window_kcal=8.0,
                rmsd_cutoff=1.25,
            ),
            ForceFieldVariant(
                name="UFF_Vacuum",
                display_title="UFF (Vacuum)",
                ff_type="uff",
                dielectric_model=1,
                dielectric_constant=1.0,
                energy_window_kcal=8.0,
                rmsd_cutoff=1.25,
            ),
            ForceFieldVariant(
                name="MMFF94s_Pocket",
                display_title="MMFF94s (Pocket, eps=6.5)",
                ff_type="mmff",
                dielectric_model=1,
                dielectric_constant=6.5,
                energy_window_kcal=None,
                rmsd_cutoff=1.25,
            ),
            ForceFieldVariant(
                name="MMFF94s_Water",
                display_title="MMFF94s (Aqueous, eps=78.4)",
                ff_type="mmff",
                dielectric_model=1,
                dielectric_constant=78.4,
                energy_window_kcal=None,
                rmsd_cutoff=1.25,
            ),
            ForceFieldVariant(
                name="MMFF94s_DistanceDep",
                display_title="MMFF94s (Distance-Dep eps(r)=r)",
                ff_type="mmff",
                dielectric_model=2,
                dielectric_constant=1.0,
                energy_window_kcal=None,
                rmsd_cutoff=1.25,
            ),
        )
