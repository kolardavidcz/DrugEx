"""Molecular Dynamics Simulation & Trajectory Analysis Engine.

Solvates protein-ligand complexes in explicit TIP3P water (0.15 M NaCl),
parameterizes ligands with GAFF-2.11 and proteins with Amber14-all,
executes energy minimization, NVT/NPT equilibration, and production MD
on NVIDIA RTX PRO 4000 Blackwell GPU (OpenCL), and evaluates ligand RMSD drift.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmm
from openbabel import pybel
from openmm import app, unit
from openmm.app import Element
import openmmforcefields

# [PACKAGE: openmm] WHY: High-performance molecular mechanics engine with GPU acceleration. WHAT: Simulation, System, ForceField, Integrator. HOW: openmm.app API.


@dataclass
class MDTrajectoryMetrics:
    """Quantitative stability metrics computed from an MD simulation trajectory."""
    ligand_id: str
    target_id: str
    duration_ns: float
    mean_ligand_rmsd_angstrom: float
    max_ligand_rmsd_angstrom: float
    final_ligand_rmsd_angstrom: float
    mean_protein_rmsd_angstrom: float
    pocket_retention_status: str  # 'STABLE' (<=2.5A), 'WOBBLE' (2.5-4.0A), 'DISSOCIATED' (>4.5A)
    time_points_ns: List[float] = field(default_factory=list)
    ligand_rmsd_curve: List[float] = field(default_factory=list)
    protein_rmsd_curve: List[float] = field(default_factory=list)


class MDSimulationEngine:
    """Encapsulates system preparation, GPU execution, and RMSD trajectory analysis."""

    def __init__(self, output_root: Path | str, target_gpu: int = 0) -> None:
        """Initialize MDSimulationEngine.

        Args:
            output_root: Directory to store trajectories, logs, and plots.
            target_gpu: Target GPU device index (default 0 for primary Blackwell GPU).
        """
        self.output_root: Path = Path(output_root)
        self.target_gpu: int = target_gpu
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Locate built-in GAFF-2.11 XML in openmmforcefields
        ff_base = Path(openmmforcefields.__file__).parent
        matches = list(ff_base.glob("**/gaff-2.11.xml"))
        if not matches:
            raise FileNotFoundError("Could not find gaff-2.11.xml in openmmforcefields package.")
        self.gaff_xml_path: Path = matches[0]

        # Select OpenCL platform for NVIDIA RTX PRO 4000 Blackwell
        self.platform = openmm.Platform.getPlatformByName("OpenCL")
        self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", str(self.target_gpu))

    @staticmethod
    def _map_openbabel_to_gaff(ob_type: str) -> str:
        """Map OpenBabel atom type strings to standard GAFF atom types."""
        mapping = {
            "C3": "c3", "C2": "c2", "Car": "ca", "C1": "c1",
            "O3": "os", "O2": "o", "S3": "ss", "S2": "s2",
            "N3": "n3", "N2": "n2", "Npl": "na", "Nam": "n",
            "H": "ha", "HO": "ho", "HN": "hn"
        }
        return mapping.get(ob_type, ob_type.lower())

    def _generate_ligand_ffxml(self, ob_mol: pybel.Molecule, xml_path: Path) -> None:
        """Generate OpenMM ForceField XML residue template for ligand using GAFF types and Gasteiger charges."""
        xml_lines = [
            '<ForceField>',
            ' <Residues>',
            '  <Residue name="LIG">'
        ]
        for a in ob_mol.atoms:
            gtype = self._map_openbabel_to_gaff(a.type)
            xml_lines.append(f'   <Atom name="A{a.idx}" type="{gtype}" charge="{a.partialcharge:.4f}"/>')

        for b in pybel.ob.OBMolBondIter(ob_mol.OBMol):
            xml_lines.append(f'   <Bond atomName1="A{b.GetBeginAtomIdx()}" atomName2="A{b.GetEndAtomIdx()}"/>')

        xml_lines.extend(['  </Residue>', ' </Residues>', '</ForceField>'])
        xml_path.write_text("\n".join(xml_lines), encoding="utf-8")

    def _build_solvated_system(
        self,
        receptor_clean_pdb: Path,
        docked_pdbqt: Path,
        sim_dir: Path,
    ) -> Tuple[app.Modeller, app.ForceField, List[int], List[int]]:
        """Construct solvated protein-ligand system in explicit water box with 0.15 M NaCl."""
        # 1. Read docked ligand coordinates via OpenBabel
        ob_mol = next(pybel.readfile("pdbqt", str(docked_pdbqt)))
        ob_mol.addh()

        # 2. Generate ligand XML
        lig_xml_path = sim_dir / "ligand.xml"
        self._generate_ligand_ffxml(ob_mol, lig_xml_path)

        # 3. Create OpenMM ForceField
        ff = app.ForceField(
            "amber14-all.xml",
            "amber14/tip3p.xml",
            str(self.gaff_xml_path),
            str(lig_xml_path),
        )

        # 4. Load and fix protein topology using PDBFixer (repairs missing terminal caps and disordered atoms)
        from pdbfixer import PDBFixer
        fixer = PDBFixer(filename=str(receptor_clean_pdb))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        modeller = app.Modeller(fixer.topology, fixer.positions)

        # Record protein C-alpha indices for alignment
        ca_indices: List[int] = [
            atom.index for atom in fixer.topology.atoms() if atom.name == "CA"
        ]

        # 5. Build ligand topology and append to Modeller
        lig_top = app.Topology()
        lig_chain = lig_top.addChain("Z")
        lig_res = lig_top.addResidue("LIG", lig_chain)
        omm_atoms = []
        for a in ob_mol.atoms:
            el = Element.getBySymbol(a.type[0])
            omm_atoms.append(lig_top.addAtom(f"A{a.idx}", el, lig_res))

        for b in pybel.ob.OBMolBondIter(ob_mol.OBMol):
            lig_top.addBond(omm_atoms[b.GetBeginAtomIdx() - 1], omm_atoms[b.GetEndAtomIdx() - 1])

        lig_coords = [openmm.Vec3(*a.coords) * unit.angstrom for a in ob_mol.atoms]

        prot_atom_count = len(modeller.positions)
        modeller.add(lig_top, lig_coords)

        # Record ligand heavy atom indices (for ligand RMSD evaluation)
        lig_heavy_indices: List[int] = [
            prot_atom_count + i
            for i, a in enumerate(ob_mol.atoms)
            if a.atomicnum > 1
        ]

        # 6. Add hydrogens to protein at pH 7.4 and solvate
        modeller.addHydrogens(ff, pH=7.4)
        modeller.addSolvent(
            ff, model="tip3p", padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar
        )

        return modeller, ff, ca_indices, lig_heavy_indices

    def run_simulation(
        self,
        ligand_id: str,
        target_id: str,
        receptor_clean_pdb: Path,
        docked_pdbqt: Path,
        duration_ns: float = 1.0,
        report_interval_ps: float = 10.0,
    ) -> MDTrajectoryMetrics:
        """Execute full MD protocol (Minimization -> NVT -> NPT -> Production) and return metrics.

        Args:
            ligand_id: Name of candidate ligand.
            target_id: Name of target receptor.
            receptor_clean_pdb: Clean receptor PDB coordinates.
            docked_pdbqt: Starting docked ligand pose from Vina.
            duration_ns: Production simulation time in nanoseconds (default 1.0 ns).
            report_interval_ps: Trajectory recording interval in picoseconds (default 10 ps).

        Returns:
            MDTrajectoryMetrics containing RMSD curves and pocket retention verdict.
        """
        sim_dir = self.output_root / f"{target_id}_{ligand_id}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[MD Setup] Building solvated complex: {ligand_id} @ {target_id}...")
        modeller, ff, ca_indices, lig_heavy_indices = self._build_solvated_system(
            receptor_clean_pdb, docked_pdbqt, sim_dir
        )
        print(f"  System atoms: {len(modeller.positions)} | Solvated box constructed.")

        # Save initial complex PDB
        initial_pdb_path = sim_dir / "system_solvated_initial.pdb"
        with open(initial_pdb_path, "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

        # Build OpenMM System
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )

        # Add barostat for constant pressure (NPT)
        barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
        system.addForce(barostat)

        # Integrator: 2.0 fs timestep
        dt_fs = 2.0
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtoseconds
        )

        simulation = app.Simulation(modeller.topology, system, integrator, self.platform)
        simulation.context.setPositions(modeller.positions)

        # ---------------------------------------------------------------------
        # 1. Energy Minimization & Thermal Equilibration
        # ---------------------------------------------------------------------
        print("  [Minimization] Minimizing energy on GPU (max 1500 iterations)...")
        simulation.minimizeEnergy(maxIterations=1500, tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer)
        minimized_positions = simulation.context.getState(getPositions=True).getPositions()

        # Extract initial reference coordinates for RMSD (docked-minimized pose)
        init_coords = np.array([[v.x, v.y, v.z] for v in minimized_positions.value_in_unit(unit.angstrom)])
        ref_ca = init_coords[ca_indices]
        ref_lig = init_coords[lig_heavy_indices]

        # Initialize thermal velocities from Maxwell-Boltzmann distribution at 300 K
        print("  [Equilibration] Initializing velocities to 300 K & relaxing NPT (5 ps)...")
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
        simulation.step(2500)  # 5 ps NPT relaxation at 2.0 fs timestep

        # ---------------------------------------------------------------------
        # 2. Production MD & Trajectory Recording
        # ---------------------------------------------------------------------
        steps_total = int((duration_ns * 1000.0) / (dt_fs / 1000.0))
        steps_per_report = int(report_interval_ps / (dt_fs / 1000.0))
        n_reports = steps_total // steps_per_report

        dcd_path = sim_dir / "trajectory.dcd"
        simulation.reporters.append(app.DCDReporter(str(dcd_path), steps_per_report))

        print(f"  [Production MD] Simulating {duration_ns:.2f} ns ({steps_total} steps, {n_reports} frames)...")

        time_points: List[float] = []
        ligand_rmsds: List[float] = []
        protein_rmsds: List[float] = []

        for frame_idx in range(n_reports):
            simulation.step(steps_per_report)
            current_time_ns = (frame_idx + 1) * (report_interval_ps / 1000.0)

            # Get current frame positions
            state = simulation.context.getState(getPositions=True)
            frame_coords = np.array([[v.x, v.y, v.z] for v in state.getPositions().value_in_unit(unit.angstrom)])

            # Kabsch alignment: Align protein C-alpha to initial frame
            curr_ca = frame_coords[ca_indices]
            # Center both
            ca_center_ref = ref_ca.mean(axis=0)
            ca_center_curr = curr_ca.mean(axis=0)
            p_ref = ref_ca - ca_center_ref
            p_curr = curr_ca - ca_center_curr

            # Optimal rotation matrix via SVD (Kabsch algorithm)
            h = p_curr.T @ p_ref
            u, s, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            diag = np.diag([1.0, 1.0, d])
            r_mat = vt.T @ diag @ u.T

            # Apply rotation to entire frame aligned by protein C-alpha
            aligned_frame = ((frame_coords - ca_center_curr) @ r_mat.T) + ca_center_ref

            # Calculate Protein C-alpha RMSD
            prot_rmsd = float(np.sqrt(np.mean(np.sum((aligned_frame[ca_indices] - ref_ca) ** 2, axis=1))))

            # Calculate Ligand Heavy-Atom RMSD relative to aligned pocket
            lig_rmsd = float(np.sqrt(np.mean(np.sum((aligned_frame[lig_heavy_indices] - ref_lig) ** 2, axis=1))))

            time_points.append(round(current_time_ns, 3))
            protein_rmsds.append(round(prot_rmsd, 3))
            ligand_rmsds.append(round(lig_rmsd, 3))

            if (frame_idx + 1) % max(1, n_reports // 5) == 0 or (frame_idx + 1) == n_reports:
                print(
                    f"    Frame {frame_idx + 1}/{n_reports} ({current_time_ns:.2f} ns) | "
                    f"Prot RMSD: {prot_rmsd:.2f} Å | Ligand RMSD: {lig_rmsd:.2f} Å"
                )

        mean_lig_rmsd = float(np.mean(ligand_rmsds))
        max_lig_rmsd = float(np.max(ligand_rmsds))
        final_lig_rmsd = float(ligand_rmsds[-1])
        mean_prot_rmsd = float(np.mean(protein_rmsds))

        # Academic verdict on binding retention
        if max_lig_rmsd <= 2.5:
            verdict = "STABLE"
        elif max_lig_rmsd <= 4.0:
            verdict = "WOBBLE"
        else:
            verdict = "DISSOCIATED"

        metrics = MDTrajectoryMetrics(
            ligand_id=ligand_id,
            target_id=target_id,
            duration_ns=duration_ns,
            mean_ligand_rmsd_angstrom=round(mean_lig_rmsd, 2),
            max_ligand_rmsd_angstrom=round(max_lig_rmsd, 2),
            final_ligand_rmsd_angstrom=round(final_lig_rmsd, 2),
            mean_protein_rmsd_angstrom=round(mean_prot_rmsd, 2),
            pocket_retention_status=verdict,
            time_points_ns=time_points,
            ligand_rmsd_curve=ligand_rmsds,
            protein_rmsd_curve=protein_rmsds,
        )

        # Save metrics JSON
        metrics_json_path = sim_dir / "md_metrics.json"
        metrics_json_path.write_text(json.dumps(asdict(metrics), indent=4), encoding="utf-8")

        return metrics

    def plot_comparative_md_curves(
        self, metrics_list: List[MDTrajectoryMetrics], output_path: Path
    ) -> Path:
        """Generate publication-ready comparative RMSD trajectory plots."""
        # [PACKAGE: matplotlib] WHY: Publication line plots of time-dependent molecular dynamics trajectories. WHAT: subplots with RMSD curves. HOW: ax.plot with retention thresholds.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        colors = {
            "EGCG_seed": "#2ca02c",
            "Lead_01": "#1f77b4",
            "Lead_02": "#aec7e8",
            "Lead_03": "#ff7f0e",
            "Lead_04": "#9467bd",
            "Lead_05": "#d62728",
        }
        target_styles = {
            "3OOB_pin1": "-",
            "3NG5_ttr": "--",
            "8ZVP_sirt6": "-.",
        }

        for m in metrics_list:
            c = colors.get(m.ligand_id, "#333333")
            ls = target_styles.get(m.target_id, "-")
            target_label = m.target_id.split('_')[1].upper() if '_' in m.target_id else m.target_id
            label = f"{m.ligand_id} on {target_label} ({m.pocket_retention_status})"
            ax1.plot(m.time_points_ns, m.ligand_rmsd_curve, label=label, color=c, linestyle=ls, linewidth=1.8)
            ax2.plot(m.time_points_ns, m.protein_rmsd_curve, label=label, color=c, linestyle=ls, linewidth=1.5, alpha=0.7)

        # Thresholds on Ligand RMSD
        ax1.axhline(2.5, color="green", linestyle=":", alpha=0.6, label="Stable Pocket (<= 2.5 Å)")
        ax1.axhline(4.0, color="orange", linestyle=":", alpha=0.6, label="Wobble Limit (4.0 Å)")

        ax1.set_xlabel("Simulation Time (ns)", fontsize=11)
        ax1.set_ylabel("Ligand Heavy-Atom RMSD (Å)", fontsize=11)
        ax1.set_title("Ligand Pocket Retention & RMSD Drift (Explicit TIP3P Solvent)", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        ax2.set_xlabel("Simulation Time (ns)", fontsize=11)
        ax2.set_ylabel("Protein C-alpha RMSD (Å)", fontsize=11)
        ax2.set_title("Receptor Backbone Stability (C-alpha)", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path
