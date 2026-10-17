from __future__ import annotations


class std:
    """Canonical single source of truth for tabular column naming and schema constants."""

    # Core Structural & Tracking Columns
    SMILES: str = "SMILES"
    FRAGS: str = "FRAGS"
    VALID: str = "Valid"
    ACCURATE: str = "Accurate"
    DESIRED: str = "Desired"
    PARETO_RANK: str = "Pareto_Rank"
    CANONICAL_SMILES: str = "Canonical_SMILES"
    EPOCH: str = "Epoch"
    SCAFFOLD: str = "Scaffold"

    # Multi-Objective & Physicochemical Properties
    ROCS: str = "ROCS"
    SA: str = "SA"
    QED: str = "QED"
    MW: str = "MW"
    LOGP: str = "LogP"
    TPSA: str = "TPSA"
    HBD: str = "HBD"
    HBA: str = "HBA"
    HAC: str = "HAC"
    ROT_BONDS: str = "RotBonds"
    TANIMOTO_TO_SEED: str = "Tanimoto_to_Seed"

    # Training Telemetry Columns
    LOSS_TRAIN: str = "loss_train"
    VALID_RATIO: str = "valid_ratio"
    ACCURATE_RATIO: str = "accurate_ratio"
    UNIQUE_RATIO: str = "unique_ratio"
    DESIRED_RATIO: str = "desired_ratio"
    AVG_AMEAN: str = "avg_amean"
    AVG_GMEAN: str = "avg_gmean"
    BEST_EPOCH: str = "best_epoch"
