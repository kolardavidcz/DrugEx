"""Validate saved geometry identity, full grids and independent score reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from _david.code.chemistry import check_identity
from _david.code.conformer.reward_calibration import KEYS, source_hashes
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.environment import create_rocs_scorer
from _david.code.rocs import score_pair


class SavedGenerator:
    def __init__(self, path):
        self.path = path

    def genConformers(self, smiles, destination):
        return str(self.path)


def validate(output: Path, phase: str, replays: int = 12):
    raw = output / "raw_data" / phase
    path = raw / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Measurement writer must finish before validation")
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Current measurement imports differ from recorded source")
    frame = pd.read_csv(raw / "data.csv.gz")
    smiles_by_id = (
        pd.read_csv(raw / "panel.csv").set_index("molecule_id").smiles.to_dict()
    )
    if len(frame) != manifest["expected_rows"] or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicated grid cells")
    expected_settings = {
        tuple(setting[key] for key in KEYS[1:]) for setting in manifest["settings"]
    }
    for mid, group in frame.groupby("molecule_id"):
        actual = {
            tuple(row[key] for key in KEYS[1:]) for row in group.to_dict("records")
        }
        if actual != expected_settings:
            raise ValueError(f"Incomplete grid for {mid}")
    geometry_checks, hash_checks = 0, 0
    for mid in manifest["input_ids"]:
        directory = raw / "checkpoints" / str(mid)
        saved = json.loads((directory / "complete.json").read_text())
        if saved["source_sha256"] != manifest["source_sha256"]:
            raise ValueError(f"Mixed source at input {mid}")
        for name, expected in saved["files_sha256"].items():
            file = directory / name
            if digest(file) != expected:
                raise ValueError(f"Changed measurement: {file}")
            hash_checks += 1
            if file.suffix != ".sdf" or file.stat().st_size == 0:
                continue
            for record in Chem.SDMolSupplier(str(file), removeHs=False):
                if record is None:
                    raise ValueError(f"Invalid geometry file: {file}")
                if not record.HasProp("EnumeratedIsomerSMILES"):
                    raise ValueError(f"Missing stereo identity: {file}")
                check_identity(
                    record, Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
                )
                geometry_checks += 1
    eligible = frame[
        ~frame.color_excluded
        & ~frame.reference_failure
        & ~frame.candidate_failure
        & frame.candidate_attempted
    ]
    if not frame.loc[~frame.color_excluded, "score"].between(0, 1.0001).all():
        raise ValueError("Invalid native normalized score")
    # Small-cap cells permit an exhaustive independent component/max reduction.
    selected = (
        eligible[eligible.reference_cap.eq(1) & eligible.candidate_attempts.eq(10)]
        .groupby("method", group_keys=False)
        .head(max(1, replays // 4))
        .head(replays)
    )
    # Larger native overshoots require their own exact public/component replay;
    # retain the original values instead of clipping the reward or table.
    selected = pd.concat(
        [selected, eligible[eligible.score.gt(1.00001)]]
    ).drop_duplicates(KEYS)
    replay_rows = []
    for row in selected.to_dict("records"):
        directory = raw / "checkpoints" / str(row["molecule_id"])
        refs = list(
            Chem.SDMolSupplier(
                str(directory / f"reference_{row['method']}.sdf"), removeHs=False
            )
        )
        if row["reference_cap"]:
            refs = refs[: row["reference_cap"]]
        candidates = list(
            Chem.SDMolSupplier(str(directory / row["candidate_file"]), removeHs=False)
        )
        best = max(
            score_pair(candidate, ref) for candidate in candidates for ref in refs
        )
        scorer = create_rocs_scorer(refs, n_jobs=1, conformer_threads=1)

        scorer.conformer_generator = SavedGenerator(directory / row["candidate_file"])
        public = float(scorer.getScores([smiles_by_id[row["molecule_id"]]])[0, 0]) / 2
        if max(abs(best[0] - row["score"]), abs(public - row["score"])) > 1e-8:
            raise ValueError(f"Score replay mismatch for {row['molecule_id']}")
        if abs(best[0] - (best[1] + best[2]) / 2) > 1e-12:
            raise ValueError("Component normalization mismatch")
        replay_rows.append(
            {key: row[key] for key in KEYS}
            | dict(
                measured=row["score"],
                public_replay=public,
                normalized_combo=best[0],
                shape_tanimoto=best[1],
                color_tanimoto=best[2],
            )
        )
    pd.DataFrame(replay_rows).to_csv(raw / "score_replays.csv", index=False)
    manifest["validation"] = dict(
        grid_rows=len(frame),
        inputs=frame.molecule_id.nunique(),
        file_hash_checks=hash_checks,
        geometry_identity_checks=geometry_checks,
        public_and_component_replays=len(replay_rows),
        score_tolerance=1e-8,
        max_public_difference=max(
            (abs(row["public_replay"] - row["measured"]) for row in replay_rows),
            default=0.0,
        ),
        min_score=float(eligible.score.min()),
        max_score=float(eligible.score.max()),
        script_sha256=digest(__file__),
    )
    archive = output / "raw_data/provenance/measurement_source.tar.gz"
    if archive.exists():
        manifest["source_archive_sha256"] = digest(archive)
    atomic_json(path, manifest)
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="development")
    parser.add_argument("--replays", type=int, default=12)
    args = parser.parse_args()
    validate(args.output, args.phase, args.replays)
