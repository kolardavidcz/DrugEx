"""Unified Runner for GPU-Accelerated Molecular Dynamics Simulations.

Executes explicit solvent MD (TIP3P + 0.15 M NaCl) on NVIDIA RTX PRO 4000 Blackwell GPU
to evaluate dynamic pocket residence and ligand RMSD stability:
- Reference Seed: EGCG (8 HBDs, native hydrogen bonding network)
- De Novo Lead 01: Top M4 lead (0 HBDs, pure hydrophobic packing)
- De Novo Lead 05: Highest affinity lead (1 HBD, amide contact)
Across Pin1 (3OOB) and SIRT6 (8ZVP).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.validation.code.md_engine import MDSimulationEngine, MDTrajectoryMetrics


def run_comparative_md_cohort(
    base_dir: Path | str | None = None,
    duration_ns: float = 0.5,
    target_gpu: int = 0,
) -> Dict[str, str]:
    """Execute comparative MD simulations across key benchmark complexes.

    Args:
        base_dir: Base validation directory. Defaults to _david/validation.
        duration_ns: Simulation length per complex in nanoseconds (default 0.5 ns).
        target_gpu: GPU index (default 0).

    Returns:
        Dictionary of generated artifact paths.
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "_david" / "validation"
    base_dir = Path(base_dir)

    md_out_dir = base_dir / "output" / "md"
    md_out_dir.mkdir(parents=True, exist_ok=True)

    receptors_dir = base_dir / "receptors"
    poses_dir = base_dir / "output" / "poses"

    # Curate key benchmark comparison pairs
    experiments: List[Tuple[str, str]] = [
        ("3OOB_pin1", "EGCG_seed"),  # Ground-truth reference control
        ("3OOB_pin1", "Lead_01"),    # Top M4 de novo lead (0 HBD)
        ("3OOB_pin1", "Lead_05"),    # Top affinity de novo lead (1 HBD)
        ("3NG5_ttr", "EGCG_seed"),   # Ground-truth TTR reference
        ("3NG5_ttr", "Lead_01"),     # Top M4 de novo lead (0 HBD)
        ("3NG5_ttr", "Lead_05"),     # Top affinity de novo lead (1 HBD)
        ("8ZVP_sirt6", "EGCG_seed"), # Ground-truth SIRT6 reference
        ("8ZVP_sirt6", "Lead_01"),   # Top SIRT6 binder (-9.89 kcal/mol)
        ("8ZVP_sirt6", "Lead_05"),   # Highest affinity SIRT6 binder (-10.95 kcal/mol)
    ]

    print("=" * 80)
    print(f"MOLECULAR DYNAMICS SIMULATION COHORT (GPU {target_gpu}: RTX PRO 4000 Blackwell)")
    print(f"Duration per complex: {duration_ns:.2f} ns | Solvent: Explicit TIP3P + 0.15 M NaCl")
    print("=" * 80)

    engine = MDSimulationEngine(output_root=md_out_dir, target_gpu=target_gpu)
    all_metrics: List[MDTrajectoryMetrics] = []

    for target_id, lig_id in experiments:
        rec_clean_pdb = receptors_dir / target_id / "receptor_clean.pdb"
        docked_pdbqt = poses_dir / f"{target_id}_{lig_id}_docked.pdbqt"

        if not rec_clean_pdb.exists() or not docked_pdbqt.exists():
            print(f"  [Skipping] Missing files for {target_id} - {lig_id}")
            continue

        metrics = engine.run_simulation(
            ligand_id=lig_id,
            target_id=target_id,
            receptor_clean_pdb=rec_clean_pdb,
            docked_pdbqt=docked_pdbqt,
            duration_ns=duration_ns,
            report_interval_ps=10.0,
        )
        all_metrics.append(metrics)

    # Generate comparative plots
    plot_path = md_out_dir / "md_comparative_rmsd.png"
    engine.plot_comparative_md_curves(all_metrics, plot_path)

    # Compile TSV report
    tsv_path = md_out_dir / "md_summary_table.tsv"
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Target_ID", "Ligand_ID", "Duration_ns", "Mean_Ligand_RMSD_A",
            "Max_Ligand_RMSD_A", "Final_Ligand_RMSD_A", "Mean_Protein_RMSD_A",
            "Pocket_Retention_Status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({
                "Target_ID": m.target_id,
                "Ligand_ID": m.ligand_id,
                "Duration_ns": f"{m.duration_ns:.2f}",
                "Mean_Ligand_RMSD_A": f"{m.mean_ligand_rmsd_angstrom:.2f}",
                "Max_Ligand_RMSD_A": f"{m.max_ligand_rmsd_angstrom:.2f}",
                "Final_Ligand_RMSD_A": f"{m.final_ligand_rmsd_angstrom:.2f}",
                "Mean_Protein_RMSD_A": f"{m.mean_protein_rmsd_angstrom:.2f}",
                "Pocket_Retention_Status": m.pocket_retention_status,
            })

    # Compile JSON report
    json_path = md_out_dir / "md_cohort_summary.json"
    json_dict = {
        "duration_ns": duration_ns,
        "gpu": target_gpu,
        "total_simulations": len(all_metrics),
        "results": [asdict(m) for m in all_metrics],
        "plot_path": str(plot_path),
        "tsv_path": str(tsv_path),
    }
    json_path.write_text(json.dumps(json_dict, indent=4), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MD SIMULATION COHORT EXECUTION COMPLETE")
    print(f"  Artifacts saved in: {md_out_dir}")
    print(f"  Plot: {plot_path.name}")
    print(f"  TSV: {tsv_path.name}")
    print(f"  JSON: {json_path.name}")
    print("=" * 80)

    return {
        "plot": str(plot_path),
        "tsv": str(tsv_path),
        "json": str(json_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MD simulation cohort")
    parser.add_argument("--duration-ns", type=float, default=0.2, help="Simulation duration in ns per complex")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU device index")
    args = parser.parse_args()

    run_comparative_md_cohort(duration_ns=args.duration_ns, target_gpu=args.gpu)
