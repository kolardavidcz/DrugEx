"""Unified Runner for Multi-Target Docking Validation and Interaction Analysis.

Executes the complete end-to-end validation across 6 molecules (5 DrugEx M4 leads + 1 EGCG seed)
and 3 experimentally verified targets (Pin1, Transthyretin, SIRT6):
1. Ligand preparation (ETKDGv3 conformers + MMFF94 + Meeko PDBQT).
2. Receptor cleaning and grid envelope calibration from co-crystallized KDH.
3. AutoDock Vina cross-docking (18 experiments).
4. PLIP biophysical contact characterization (H-bond donor/acceptor tracking, hydrophobic packing).
5. Generation of publication artifacts, heatmaps, and empirical HBD collapse analysis.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.validation.code.docking_engine import DockingResult, VinaDockingEngine
from _david.validation.code.prepare_ligands import LigandPreparer, LigandRecord
from _david.validation.code.prepare_targets import ReceptorPreparer, TargetMetadata
from _david.validation.code.visualizer import ValidationVisualizer


def run_full_validation_pipeline(
    base_dir: Path | str | None = None, exhaustiveness: int = 4, cpu_cores: int = 4
) -> Dict[str, str]:
    """Execute the end-to-end multi-target validation pipeline.

    Args:
        base_dir: Base validation root directory. Defaults to _david/validation.
        exhaustiveness: AutoDock Vina search exhaustiveness.
        cpu_cores: Number of parallel CPU threads for Vina.

    Returns:
        Dictionary of generated artifact paths.
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "_david" / "validation"
    base_dir = Path(base_dir)

    receptors_dir = base_dir / "receptors"
    ligands_dir = base_dir / "ligands"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1: PREPARING 6 LIGANDS (5 M4 LEADS + 1 EGCG SEED REFERENCE)")
    print("=" * 80)
    lig_preparer = LigandPreparer(ligands_dir)
    ligands = lig_preparer.prepare_all_ligands()
    print(f"Generated 3D conformers and PDBQT files for {len(ligands)} ligands in {ligands_dir}\n")

    print("=" * 80)
    print("STAGE 2: PREPARING 3 TARGET RECEPTORS (PIN1, TTR, SIRT6)")
    print("=" * 80)
    rec_preparer = ReceptorPreparer(receptors_dir)
    targets = rec_preparer.prepare_all_targets()
    for tid, meta in targets.items():
        print(f"  Target: {tid} ({meta.common_name})")
        print(f"    Class: {meta.biological_class} | Resolution: {meta.resolution_angstrom} A")
        print(f"    Grid Envelope: Center={meta.box_center}, Size={meta.box_size}")
    print()

    print("=" * 80)
    print("STAGE 3: CROSS-DOCKING & PLIP CONTACT PROFILING (18 EXPERIMENTS)")
    print("=" * 80)
    engine = VinaDockingEngine(output_dir, exhaustiveness=exhaustiveness, cpu_cores=cpu_cores)
    docking_results = engine.run_full_validation_matrix(ligands, targets, receptors_dir)
    print(f"Successfully evaluated {len(docking_results)} docking complexes.\n")

    print("=" * 80)
    print("STAGE 4: GENERATING PUBLICATION METRICS & CHARTS")
    print("=" * 80)
    visualizer = ValidationVisualizer(output_dir)
    heatmap_path = visualizer.plot_affinity_heatmap(docking_results, targets)
    hbond_path = visualizer.plot_hbond_collapse_profile(docking_results)
    radar_path = visualizer.plot_interaction_radar_and_breakdown(docking_results)

    # Compile comprehensive TSV report
    tsv_report_path = output_dir / "cross_target_validation_report.tsv"
    with open(tsv_report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Ligand_ID", "Is_Reference", "Target_ID", "Target_Name", "Biological_Class",
            "Affinity_kcal_mol", "Prot_Donor_HBonds", "Lig_Donor_HBonds",
            "Hydrophobic_Contacts", "Pi_Stacking", "Interacting_Residues"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in docking_results:
            writer.writerow({
                "Ligand_ID": r.ligand_id,
                "Is_Reference": r.is_reference,
                "Target_ID": r.target_id,
                "Target_Name": r.target_name,
                "Biological_Class": r.biological_class,
                "Affinity_kcal_mol": f"{r.affinity_kcal_mol:.2f}",
                "Prot_Donor_HBonds": r.interactions.num_hbonds_prot_donor,
                "Lig_Donor_HBonds": r.interactions.num_hbonds_lig_donor,
                "Hydrophobic_Contacts": r.interactions.num_hydrophobic,
                "Pi_Stacking": r.interactions.num_pi_stacking,
                "Interacting_Residues": ", ".join(r.interactions.interacting_residues),
            })

    # Compile JSON summary
    summary_json_path = output_dir / "validation_summary.json"
    summary_data = {
        "num_ligands": len(ligands),
        "num_targets": len(targets),
        "total_experiments": len(docking_results),
        "artifacts": {
            "heatmap": str(heatmap_path),
            "hbond_collapse": str(hbond_path),
            "mechanism_shift": str(radar_path),
            "tsv_report": str(tsv_report_path),
        },
        "target_classifications": {
            tid: {
                "common_name": m.common_name,
                "class": m.biological_class,
                "ec_number": m.ec_number,
                "functional_role": m.functional_role,
            }
            for tid, m in targets.items()
        },
    }
    summary_json_path.write_text(json.dumps(summary_data, indent=4), encoding="utf-8")

    print("\nValidation execution complete!")
    print(f"Artifacts generated in: {output_dir}")
    print(f"  Heatmap: {heatmap_path.name}")
    print(f"  H-bond Analysis: {hbond_path.name}")
    print(f"  Mechanism Shift: {radar_path.name}")
    print(f"  TSV Report: {tsv_report_path.name}")

    return {
        "heatmap": str(heatmap_path),
        "hbond_collapse": str(hbond_path),
        "mechanism_shift": str(radar_path),
        "tsv_report": str(tsv_report_path),
        "summary_json": str(summary_json_path),
    }


if __name__ == "__main__":
    run_full_validation_pipeline()
