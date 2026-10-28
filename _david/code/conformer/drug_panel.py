"""Freeze a score-independent panel from the downloaded DrugCentral source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Lipinski
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

from _david.code.chemistry import standardize_smiles


def build_panel(source: Path, output: Path, size: int = 512) -> None:
    """Select before embedding, convergence or scoring; preserve the source audit."""
    source_manifest = json.loads((source / "source_manifest.json").read_text())
    for filename, info in source_manifest["sources"].items():
        actual = hashlib.sha256((source / "source" / filename).read_bytes()).hexdigest()
        if actual != info["sha256"]:
            raise ValueError(f"Source hash mismatch: {filename}")
    with (source / "source/fda_approved.csv").open() as handle:
        approved = {row[0] for row in csv.reader(handle)}
    audit, seen = [], set()
    with (
        (source / "source/structures_20210901.tsv").open() as handle,
        rdBase.BlockLogs(),
    ):
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["ID"] not in approved:
                continue
            item = dict(
                molecule_id=int(row["ID"]),
                name=row["INN"],
                source_smiles=row["SMILES"],
                status="",
                error="",
            )
            try:
                smiles = standardize_smiles(row["SMILES"])
                mol = Chem.MolFromSmiles(smiles)
                heavy = mol.GetNumHeavyAtoms()
                rotors = Lipinski.NumRotatableBonds(mol)
                item.update(
                    smiles=smiles,
                    heavy_atoms=heavy,
                    rotatable_bonds=rotors,
                    stereo_upper_bound=int(GetStereoisomerCount(mol)),
                )
                if heavy > 45 or rotors > 15:
                    item["status"] = "outside_size_rotor_domain"
                elif smiles in seen:
                    item["status"] = "duplicate_standardized_input"
                else:
                    item["status"] = "eligible_before_preparation"
                    seen.add(smiles)
            except (ValueError, RuntimeError) as exc:
                item.update(status="standardization_failure", error=str(exc))
            audit.append(item)
    frame = pd.DataFrame(audit)
    eligible = frame[frame.status.eq("eligible_before_preparation")].sort_values(
        "molecule_id"
    )
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible inputs for requested {size}")
    selected = eligible.iloc[
        np.random.default_rng(42).choice(len(eligible), size, replace=False)
    ]
    selected = selected.copy().reset_index(drop=True)
    # A frozen development/confirmation label supports later bounded refinement.
    selected["split"] = [
        "development" if i < size // 4 else "confirmation" for i in range(size)
    ]
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "source_audit.csv", index=False)
    selected.to_csv(output / "panel.csv", index=False)
    manifest = dict(
        state="panel_frozen_before_geometry",
        seed=42,
        size=size,
        source=source_manifest,
        rdkit=rdBase.rdkitVersion,
        eligible_inputs=len(eligible),
        status_counts=dict(Counter(frame.status)),
        selection="Uniform without replacement from unique standardized inputs with <=45 heavy atoms and <=15 rotatable bonds; no convergence, score or reference-count filtering",
        stereo="Source assignments preserved; no stereo-masked variants",
        split_counts=dict(Counter(selected.split)),
        panel_sha256=hashlib.sha256((output / "panel.csv").read_bytes()).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("size", "eligible_inputs", "status_counts", "split_counts")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    build_panel(args.source, args.output, args.size)
