from __future__ import annotations

from typing import Sequence
import pandas as pd


class DataSchema:
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

    # Multi-Objective & Physicochemical Properties
    ROCS: str = "ROCS"
    SA: str = "SA"
    QED: str = "QED"
    MW: str = "MW"
    LOGP: str = "LogP"
    TPSA: str = "TPSA"
    HBD: str = "HBD"
    ROT_BONDS: str = "RotBonds"
    TANIMOTO_TO_SEED: str = "Tanimoto_to_Seed"

    @classmethod
    def find_rocs_column(cls, columns: Sequence[str]) -> str | None:
        """Identify any ROCS / 3D shape overlap column dynamically."""
        for col in columns:
            if col == cls.ROCS:
                return col
        for col in columns:
            c_up = col.upper()
            if any(k in c_up for k in ("ROCS", "TANIMOTOCOMBO", "SHAPE_SIMILARITY", "SIMILARITY_3D")):
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

        # 1. Normalize ROCS column
        rocs_col = cls.find_rocs_column(df_out.columns)
        if rocs_col and rocs_col != cls.ROCS:
            rename_map[rocs_col] = cls.ROCS

        # 2. Normalize SA column
        sa_col = cls.find_sa_column(df_out.columns)
        if sa_col and sa_col != cls.SA:
            rename_map[sa_col] = cls.SA

        # 3. Normalize Desired column
        des_col = cls.find_desired_column(df_out.columns)
        if des_col and des_col != cls.DESIRED:
            rename_map[des_col] = cls.DESIRED

        if rename_map:
            df_out = df_out.rename(columns=rename_map)

        return df_out
