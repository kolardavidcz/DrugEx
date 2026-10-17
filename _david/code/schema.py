from __future__ import annotations

from typing import Sequence
import pandas as pd


class std:
    """Canonical Single Source of Truth for column naming and tabular standardization.

    WHY:
    Eliminates fragile ad-hoc column name lookups across pipeline, environment, analysis,
    and visualization modules where external scorers (e.g. DrugEx RDKitROCSScorer) produce dynamic keys
    like 'RDKit_Supermol_TanimotoCombo_oe' or 'RDKit_Aggregate_8refs_TanimotoCombo_oe', and environments
    output 'Desired'.

    WHAT:
    Defines canonical column constants and provides a deterministic standardize() method that
    normalizes any raw DataFrame into the canonical schema.

    HOW:
    Maps recognized variants of ROCS, SA, Desired, and Pareto columns to single authoritative constants.
    """

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

    @classmethod
    def find_rocs_column(cls, columns: Sequence[str]) -> str | None:
        """Identify any ROCS / 3D shape overlap column dynamically."""
        for col in columns:
            if col == cls.ROCS:
                return col
        for col in columns:
            c_up = col.upper()
            if any(
                k in c_up
                for k in ("ROCS", "TANIMOTOCOMBO", "SHAPE_SIMILARITY", "SIMILARITY_3D")
            ):
                return col
        return None

    @classmethod
    def find_sa_column(cls, columns: Sequence[str]) -> str | None:
        """Identify any SA synthesizability column dynamically."""
        for col in columns:
            if col == cls.SA:
                return col
        for col in columns:
            c_up = col.upper()
            if c_up in ("SA", "SASCORE", "SA_SCORE", "SYNTHETIC_ACCESSIBILITY"):
                return col
        for col in columns:
            c_up = col.upper()
            if "TPSA" in c_up:
                continue
            if c_up.startswith("SA_") or c_up.endswith("_SA") or "SASCORE" in c_up:
                return col
        return None

    @classmethod
    def find_desired_column(cls, columns: Sequence[str]) -> str | None:
        """Identify desirability / multi-objective pass indicator column."""
        for col in columns:
            if col == cls.DESIRED:
                return col
        for col in columns:
            c_up = col.upper()
            if c_up in ("DESIRED", "DESIRABILITY", "DES", "WEIGHTEDSUM"):
                return col
        return None

    @classmethod
    def standardize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize any raw or historical DataFrame to canonical column names."""
        if df.empty:
            return df.copy()

        df_out = df.copy()
        rename_map: dict[str, str] = {}

        # 1. Normalize ROCS column and drop redundant ROCS variants if ROCS exists
        if cls.ROCS in df_out.columns:
            drop_rocs = [
                c
                for c in df_out.columns
                if c != cls.ROCS
                and any(
                    k in c.upper()
                    for k in ("TANIMOTOCOMBO", "SHAPE_SIMILARITY", "SIMILARITY_3D")
                )
            ]
            if drop_rocs:
                df_out = df_out.drop(columns=drop_rocs)
        else:
            rocs_col = cls.find_rocs_column(df_out.columns)
            if rocs_col and rocs_col != cls.ROCS:
                rename_map[rocs_col] = cls.ROCS

        # 2. Normalize SA column and drop duplicate SAscore columns if SA exists
        if cls.SA in df_out.columns:
            drop_sa = [
                c
                for c in df_out.columns
                if c != cls.SA
                and c.upper() in ("SASCORE", "SA_SCORE", "SYNTHETIC_ACCESSIBILITY")
            ]
            if drop_sa:
                df_out = df_out.drop(columns=drop_sa)
        else:
            sa_col = cls.find_sa_column(df_out.columns)
            if sa_col and sa_col != cls.SA:
                rename_map[sa_col] = cls.SA

        # 3. Normalize Desired column and drop duplicate Desirability columns if Desired exists
        if cls.DESIRED in df_out.columns:
            drop_des = [
                c
                for c in df_out.columns
                if c != cls.DESIRED and c.upper() in ("DESIRABILITY", "DES")
            ]
            if drop_des:
                df_out = df_out.drop(columns=drop_des)
        else:
            des_col = cls.find_desired_column(df_out.columns)
            if des_col and des_col != cls.DESIRED:
                rename_map[des_col] = cls.DESIRED

        # 4. Normalize structural & physicochemical conventions
        if "Frags" in df_out.columns:
            if cls.FRAGS in df_out.columns:
                df_out = df_out.drop(columns=["Frags"])
            else:
                rename_map["Frags"] = cls.FRAGS

        if "logP" in df_out.columns:
            if cls.LOGP in df_out.columns:
                df_out = df_out.drop(columns=["logP"])
            else:
                rename_map["logP"] = cls.LOGP

        if rename_map:
            df_out = df_out.rename(columns=rename_map)

        return df_out

    @classmethod
    def unscale_to_physical_metrics(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Unscale ROCS [0, 1] to [0, 2] TanimotoCombo and calculate real raw SA [1, 10] if clipped.

        WHY:
        DrugEx environment records modified desirability scores ([0, 1]) in _smiles.tsv.
        For academic integrity, medicinal chemistry presentation, and Pareto analysis,
        we must report physical observables: TanimotoCombo in [0, 2.0] and raw SAscore in [1.0, 10.0].

        WHAT:
        1. If ROCS column max <= 1.05 and non-empty, scales by * 2.0.
        2. If SA column max <= 1.05 and non-empty, calculates canonical Ertl & Schuffenhauer (2009)
           raw SAscore in [1.0, 10.0] from SMILES.
        """
        if df.empty:
            return df.copy()

        df_out = df.copy()

        # 1. Unscale ROCS: [0, 1] desirability -> [0, 2.0] TanimotoCombo
        # Threshold 1.05 provides a 5% margin against float serialization/rounding noise
        # (e.g. 1.0000000002) to reliably distinguish normalized [0, 1] from physical [0, 2.0]
        if cls.ROCS in df_out.columns:
            vals = df_out[cls.ROCS].dropna()
            if not vals.empty and vals.max() <= 1.05:
                df_out[cls.ROCS] = df_out[cls.ROCS] * 2.0

        # 2. Unscale SA: [0, 1] desirability -> [1.0, 10.0] raw SAscore
        # Same 1.05 margin distinguishes [0, 1] desirability from physical SA in [1.0, 10.0]
        if cls.SA in df_out.columns and cls.SMILES in df_out.columns:
            vals = df_out[cls.SA].dropna()
            if not vals.empty and vals.max() <= 1.05:
                from drugex.training.scorers.sascorer import (
                    calculateScore as calculate_sa_score,
                )
                from rdkit import Chem

                mols = [
                    Chem.MolFromSmiles(s) if isinstance(s, str) else None
                    for s in df_out[cls.SMILES]
                ]
                df_out[cls.SA] = [
                    float(calculate_sa_score(m)) if m is not None else 10.0
                    for m in mols
                ]

        return df_out

