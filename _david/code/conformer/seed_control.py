"""Audit exact raw coordinate reuse in the superseded 43/42-seed study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from _david.code.conformer.self_recovery import atomic_json, digest


def coordinate_keys(path):
    return [
        hashlib.sha256(mol.GetConformer().GetPositions().tobytes()).hexdigest()
        for mol in Chem.SDMolSupplier(str(path), removeHs=False)
    ]


def audit(output):
    raw = output / "raw_data/development"
    provenance = output / "raw_data/provenance"
    provenance.mkdir(exist_ok=True)
    rows, hashes = [], {}
    for directory in sorted((raw / "checkpoints").iterdir()):
        candidates = list(directory.glob("candidate_etkdg_c50_*.sdf"))
        reference = directory / "reference_etkdg.sdf"
        if not candidates or not reference.exists():
            continue
        reference_keys = coordinate_keys(reference)
        for path in candidates:
            candidate_keys = coordinate_keys(path)
            matches = [
                (ci, ri)
                for ci, key in enumerate(candidate_keys)
                for ri, rkey in enumerate(reference_keys)
                if key == rkey
            ]
            rows.append(
                dict(
                    molecule_id=int(directory.name),
                    candidate_file=path.name,
                    reference_conformers=len(reference_keys),
                    candidate_conformers=len(candidate_keys),
                    exact_matching_pairs=len(matches),
                    indices=json.dumps(matches),
                )
            )
            hashes[str(path.relative_to(output))] = digest(path)
        hashes[str(reference.relative_to(output))] = digest(reference)
    frame = pd.DataFrame(rows)
    frame.to_csv(provenance / "seed_overlap.csv", index=False)
    control = dict(
        rdkit_effective_seed="base_seed*(conformer_index+1), sequential flag false",
        original_reference_base=43,
        original_candidate_base=42,
        reference_ordinal=42,
        candidate_ordinal=43,
        shared_effective_seed=1806,
        source="https://github.com/rdkit/rdkit/blob/Release_2026_03_6/Code/GraphMol/DistGeomHelpers/Embedder.cpp#L1593-L1601",
        audited_inputs=frame.molecule_id.nunique(),
        inputs_with_exact_reuse=frame.loc[
            frame.exact_matching_pairs.gt(0), "molecule_id"
        ].nunique(),
        exact_pairs=int(frame.exact_matching_pairs.sum()),
        files_sha256=hashes,
        corrected_reference_base=104729,
        corrected_candidate_base=42,
        corrected_bounded_overlap=sorted(
            set(104729 * i for i in range(1, 51)) & set(42 * i for i in range(1, 76))
        ),
        script_sha256=digest(__file__),
    )
    atomic_json(provenance / "seed_overlap.json", control)
    print(
        json.dumps(
            {key: value for key, value in control.items() if key != "files_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
