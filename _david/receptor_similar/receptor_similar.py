"""End-to-End Molecular Bioactivity & Target Deconvolution Pipeline (Papyrus-Only).

This module automates the extraction and curation of shared-target ligand spaces:
1. Standardizes a query molecule using DrugEx DefaultStandardizer (SMILES -> canonical SMILES, InChIKey, connectivity).
2. Identifies all biological receptors (targets / UniProt accessions) binding the molecule via Papyrus (with ChEMBL API fallback).
3. Harvests all compounds active against one or more of these identified receptors exclusively from Papyrus chemogenomics.
4. Checks and filters candidate molecules by properties that support DrugEx Scorers (pAffinity, QED, SAScore, MW, LogP, TPSA).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from drugex.data.processing import Standardization
from drugex.molecules.converters.standardizers import DefaultStandardizer, StandardizationException
from drugex.training.scorers import sascorer


class NormalizedMolecule(NamedTuple):
    """Container for standardized chemical identifier representations.

    Attributes
    ----------
    canonical_smiles : str
        Standardized canonical SMILES string without salts or solvates.
    inchikey : str
        Full 27-character IUPAC InChIKey hash.
    connectivity : str
        First 14-character block of the InChIKey (invariant to stereochemistry and protonation).
    """

    canonical_smiles: str
    inchikey: str
    connectivity: str


class TargetRecord(TypedDict):
    """Structured record representing an identified biological receptor target.

    Attributes
    ----------
    target_chembl_id : str
        ChEMBL target identifier (e.g. ``'CHEMBL2111414'``).
    uniprot_acc : str
        Primary UniProt accession code (e.g. ``'P00519'``).
    pref_name : str
        Preferred target protein name (e.g. ``'Tyrosine-protein kinase ABL1'``).
    organism : str
        Source organism taxonomy (e.g. ``'Homo sapiens'``).
    target_type : str
        Classification type (e.g. ``'SINGLE PROTEIN'``).
    pchembl_query : float
        Query molecule binding affinity expressed as standard -log10(molar) value.
    """

    target_chembl_id: str
    uniprot_acc: str
    pref_name: str
    organism: str
    target_type: str
    pchembl_query: float


@dataclass
class ScorerPropertyFilter:
    """Evaluates and filters molecules against property constraints supporting DrugEx Scorers.

    Parameters
    ----------
    min_paffinity : Optional[float]
        Minimum consensus pAffinity (-log10 M) threshold (default: 6.0, corresponding to pIC50 >= 6.0).
    min_qed : Optional[float]
        Minimum quantitative drug-likeness score (0.0 to 1.0).
    max_sascore : Optional[float]
        Maximum synthetic accessibility score (1.0 to 10.0, lower is easier to synthesize).
    mw_range : Optional[Tuple[float, float]]
        Acceptable molecular weight range in Daltons (e.g. (200.0, 600.0)).
    logp_range : Optional[Tuple[float, float]]
        Acceptable octanol-water partition coefficient range (e.g. (0.0, 5.0)).
    tpsa_range : Optional[Tuple[float, float]]
        Acceptable topological polar surface area range in Å² (e.g. (0.0, 140.0)).
    require_valid_rdkit : bool
        If True, drops any molecules failing RDKit parsing or sanitization (default: True).
    max_molecules : Optional[int]
        Optional upper cap on the number of candidate molecules returned (e.g. 1000 for fine-tuning).
    """

    min_paffinity: Optional[float] = 6.0
    min_qed: Optional[float] = None
    max_sascore: Optional[float] = None
    mw_range: Optional[Tuple[float, float]] = None
    logp_range: Optional[Tuple[float, float]] = None
    tpsa_range: Optional[Tuple[float, float]] = None
    require_valid_rdkit: bool = True
    max_molecules: Optional[int] = None

    @staticmethod
    def compute_molecular_properties(mol: Chem.Mol) -> Dict[str, float]:
        """Compute chemical properties corresponding to DrugEx Scorers.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        props : Dict[str, float]
            Dictionary containing MW, logP, TPSA, QED, and SAScore.
        """
        props: Dict[str, float] = {
            "MW": float(Descriptors.MolWt(mol)),
            "logP": float(Crippen.MolLogP(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "QED": float(QED.qed(mol)),
        }
        try:
            props["SAScore"] = float(sascorer.calculateScore(mol))
        except Exception:
            props["SAScore"] = float("nan")
        return props

    def filter(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Filter a DataFrame of molecules and annotate computed property columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a 'smiles' or 'SMILES' column.
        verbose : bool, optional
            Whether to log filtering summary statistics (default: False).

        Returns
        -------
        filtered_df : pd.DataFrame
            Filtered DataFrame with annotated property columns ('MW', 'logP', 'TPSA', 'QED', 'SAScore').
        """
        if df.empty:
            return df.copy()

        smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
        if smiles_col not in df.columns:
            return df.copy()

        n_initial = len(df)
        records = df.to_dict(orient="records")
        accepted_records: List[Dict[str, Any]] = []

        for rec in records:
            smi = rec.get(smiles_col)
            if not isinstance(smi, str) or len(smi) < 2:
                continue

            # Affinity check
            if self.min_paffinity is not None:
                p_aff = rec.get("pAffinity") or rec.get("pchembl_value_Mean")
                if p_aff is not None:
                    try:
                        if float(p_aff) < self.min_paffinity:
                            continue
                    except (ValueError, TypeError):
                        pass

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                if self.require_valid_rdkit:
                    continue
                else:
                    accepted_records.append(rec)
                    continue

            # Calculate Scorer properties
            props = self.compute_molecular_properties(mol)
            mw = props["MW"]
            logp = props["logP"]
            tpsa = props["TPSA"]
            qed_val = props["QED"]
            sa_val = props["SAScore"]

            # Filter bounds
            if self.mw_range is not None:
                if not (self.mw_range[0] <= mw <= self.mw_range[1]):
                    continue

            if self.logp_range is not None:
                if not (self.logp_range[0] <= logp <= self.logp_range[1]):
                    continue

            if self.tpsa_range is not None:
                if not (self.tpsa_range[0] <= tpsa <= self.tpsa_range[1]):
                    continue

            if self.min_qed is not None:
                if qed_val < self.min_qed:
                    continue

            if self.max_sascore is not None and not np.isnan(sa_val):
                if sa_val > self.max_sascore:
                    continue

            # Annotate properties into record
            rec["MW"] = mw
            rec["logP"] = logp
            rec["TPSA"] = tpsa
            rec["QED"] = qed_val
            rec["SAScore"] = sa_val
            accepted_records.append(rec)

        if not accepted_records:
            cols = list(df.columns)
            for c in ["MW", "logP", "TPSA", "QED", "SAScore"]:
                if c not in cols:
                    cols.append(c)
            return pd.DataFrame(columns=cols)

        out_df = pd.DataFrame(accepted_records)

        # Sort by pAffinity descending
        if "pAffinity" in out_df.columns:
            out_df = out_df.sort_values(by="pAffinity", ascending=False)

        # Cap max molecules if requested
        if self.max_molecules is not None and len(out_df) > self.max_molecules:
            out_df = out_df.iloc[: self.max_molecules]

        out_df = out_df.reset_index(drop=True)

        if verbose:
            print(
                f"[ScorerPropertyFilter] Kept {len(out_df)} / {n_initial} molecules "
                f"matching Scorer constraints."
            )

        return out_df


@dataclass(frozen=True)
class PipelineResult:
    """Consolidated outputs from an end-to-end bioactivity query pipeline run.

    Attributes
    ----------
    query_molecule : NormalizedMolecule
        Standardized query structure and associated InChIKey hashes.
    targets : List[TargetRecord]
        List of identified target receptors interacting with the query compound.
    papyrus_curated : pd.DataFrame
        Curated, quality-filtered benchmark subset harvested from Papyrus chemogenomics.
    """

    query_molecule: NormalizedMolecule
    targets: List[TargetRecord]
    papyrus_curated: pd.DataFrame

    @property
    def molecules(self) -> pd.DataFrame:
        """Primary DataFrame of harvested molecules."""
        return self.papyrus_curated

    @property
    def outcome_2(self) -> pd.DataFrame:
        """Alias for `papyrus_curated` maintaining backward compatibility."""
        return self.papyrus_curated

    @property
    def public_bioactivities(self) -> pd.DataFrame:
        """Backward compatibility alias returning `papyrus_curated`."""
        return self.papyrus_curated

    @property
    def outcome_1(self) -> pd.DataFrame:
        """Backward compatibility alias returning `papyrus_curated`."""
        return self.papyrus_curated

    def get_unique_smiles(
        self,
        source: str = "papyrus",
        min_paffinity: Optional[float] = None
    ) -> List[str]:
        """Extract a clean, deduplicated list of candidate ligand SMILES strings.

        Parameters
        ----------
        source : str, optional
            Candidate pool (defaults to ``'papyrus'``). Kept for backward compatibility.
        min_paffinity : float, optional
            Optional minimum pAffinity cutoff.

        Returns
        -------
        List[str]
            Deduplicated list of canonical SMILES strings.
        """
        if self.papyrus_curated.empty:
            return []

        df = self.papyrus_curated
        if min_paffinity is not None and "pAffinity" in df.columns:
            df = df[df["pAffinity"] >= min_paffinity]

        col = "SMILES" if "SMILES" in df.columns else "smiles"
        if col not in df.columns:
            return []

        candidates = df[col].dropna().tolist()
        return list(dict.fromkeys(s for s in candidates if isinstance(s, str) and len(s) > 3))

    @property
    def unique_smiles(self) -> List[str]:
        """Deduplicated list of all candidate ligand SMILES from Papyrus."""
        return self.get_unique_smiles()


def ensure_papyrus_downloaded(
    version: str = "latest",
    source_path: Optional[str] = None,
    nostereo: bool = True,
    auto_download: bool = True
) -> Optional[str]:
    """Check if local Papyrus dataset files are available, and download if missing.

    Parameters
    ----------
    version : str, optional
        Version tag of Papyrus (default: ``'latest'``).
    source_path : str, optional
        Path to Papyrus root directory. If None, auto-resolved.
    nostereo : bool, optional
        Whether to check/download non-stereochemical dataset (default: True).
    auto_download : bool, optional
        If True, downloads missing dataset via ``papyrus_scripts.download_papyrus`` (default: True).

    Returns
    -------
    resolved_path : str or None
        Path to available Papyrus repository, or None if unavailable.
    """
    resolved = source_path or MoleculeBioactivityPipeline._resolve_papyrus_path()
    if resolved and Path(resolved).exists():
        return resolved

    if not auto_download:
        warnings.warn("Papyrus data files not found and auto_download is disabled.")
        return None

    try:
        from papyrus_scripts import download_papyrus
        target_dir = Path(source_path) if source_path else Path.home() / ".Papyrus"
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Papyrus] Downloading Papyrus dataset version '{version}' to {target_dir}...")
        download_papyrus(
            outdir=target_dir,
            version=version,
            nostereo=nostereo,
            descriptors=None
        )
        return str(target_dir)
    except Exception as err:
        warnings.warn(f"Failed to auto-download Papyrus: {err}")
        return None


class MoleculeBioactivityPipeline:
    """Automated multi-target bioactivity extraction pipeline using exclusively Papyrus chemogenomics.

    Standardizes a query ligand, deconvolutes its biological receptor targets,
    and extracts all shared-target active molecules directly from Papyrus with optional
    Scorer property filtering (pAffinity, QED, SAScore, MW, LogP, TPSA).

    Parameters
    ----------
    papyrus_version : str, optional
        Version tag of the Papyrus database to read (default: ``'latest'``).
    papyrus_source_path : str, optional
        Explicit path to Papyrus repository folder. If None, automatically detected.
    timeout : int, optional
        HTTP network request timeout in seconds for fallback ChEMBL target queries (default: 15).
    session : requests.Session, optional
        Custom HTTP session for connection pooling. If None, a default session is created.
    auto_download_papyrus : bool, optional
        Whether to automatically download Papyrus dataset if missing on disk (default: True).

    Examples
    --------
    >>> from receptor_similar import MoleculeBioactivityPipeline, ScorerPropertyFilter
    >>> pipeline = MoleculeBioactivityPipeline()
    >>> smi = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(C)CC5"
    >>> prop_filter = ScorerPropertyFilter(min_paffinity=6.5, min_qed=0.4, max_sascore=4.5)
    >>> result = pipeline.run(smi, property_filter=prop_filter)
    >>> print(f"Found {len(result.targets)} targets and {len(result.papyrus_curated)} ligands.")
    """

    @staticmethod
    def _resolve_papyrus_path() -> Optional[str]:
        """Automatically locate local Papyrus repository cache if present on disk.

        Returns
        -------
        path : str or None
            Absolute path to the local Papyrus root directory, or None if not found.
        """
        candidate_roots = [
            Path(__file__).resolve().parents[2] / "tutorial" / "data" / "data" / ".Papyrus",
            Path(__file__).resolve().parents[2] / "data" / "data" / ".Papyrus",
            Path(__file__).resolve().parents[2] / ".Papyrus",
            Path.home() / ".Papyrus",
            Path.home() / ".data",
        ]
        for candidate in candidate_roots:
            if candidate.exists() and (candidate / "papyrus" / "versions.json").exists():
                return str(candidate)
            if candidate.exists() and (candidate / "versions.json").exists():
                return str(candidate.parent)
        return None

    def __init__(
        self,
        papyrus_version: str = "latest",
        papyrus_source_path: Optional[str] = None,
        timeout: int = 15,
        session: Optional[requests.Session] = None,
        auto_download_papyrus: bool = True
    ) -> None:
        self.papyrus_version: str = papyrus_version
        self.timeout: int = timeout
        self.chembl_api: str = "https://www.ebi.ac.uk/chembl/api/data"
        self.session: requests.Session = session if session is not None else requests.Session()
        self.standardizer: DefaultStandardizer = DefaultStandardizer()

        # Resolve or ensure Papyrus repository
        resolved = papyrus_source_path or self._resolve_papyrus_path()
        if not resolved and auto_download_papyrus:
            resolved = ensure_papyrus_downloaded(version=papyrus_version, source_path=papyrus_source_path)
        self.papyrus_source_path: Optional[str] = resolved

    def standardize_molecule(self, smiles: str) -> NormalizedMolecule:
        """Normalize chemical structure using DrugEx DefaultStandardizer and generate InChIKeys.

        Applies DrugEx standardizer routines (metal disconnection, normalization,
        largest fragment selection, charge neutralization) prior to database querying.

        Parameters
        ----------
        smiles : str
            Input raw SMILES representation of the query molecule.

        Returns
        -------
        normalized : NormalizedMolecule
            Named tuple containing ``(canonical_smiles, inchikey, connectivity)``.

        Raises
        ------
        ValueError
            If the supplied SMILES string cannot be parsed or standardized.
        """
        try:
            can_smiles = self.standardizer(smiles)
        except Exception as err:
            raise ValueError(f"DrugEx DefaultStandardizer failed on SMILES '{smiles}': {err}") from err

        if not can_smiles:
            raise ValueError(f"Unable to standardize SMILES string: '{smiles}'")

        mol = Chem.MolFromSmiles(can_smiles)
        if mol is None:
            raise ValueError(f"Unable to parse standardized SMILES into RDKit molecule: '{can_smiles}'")

        inchikey = Chem.MolToInchiKey(mol)
        connectivity = inchikey.split("-")[0]

        return NormalizedMolecule(
            canonical_smiles=can_smiles,
            inchikey=inchikey,
            connectivity=connectivity
        )

    def find_targets_papyrus(
        self,
        inchikey: str,
        connectivity: str,
        min_paffinity: float = 6.0,
        organism: Optional[str] = "Homo sapiens"
    ) -> List[TargetRecord]:
        """Search local Papyrus dataset for targets binding the query compound.

        Parameters
        ----------
        inchikey : str
            Full 27-character InChIKey.
        connectivity : str
            First 14-character connectivity block.
        min_paffinity : float, optional
            Minimum binding affinity threshold (default: 6.0).
        organism : str, optional
            Target source organism filter (default: 'Homo sapiens').

        Returns
        -------
        targets : List[TargetRecord]
            Resolved targets matching the query compound in Papyrus.
        """
        try:
            import polars as pl
            from papyrus_scripts.reader import read_papyrus
        except ImportError:
            return []

        try:
            papyrus_data = read_papyrus(
                version=self.papyrus_version,
                source_path=self.papyrus_source_path,
                plusplus=True,
                chunksize=1
            )
            lf = papyrus_data.lazy() if isinstance(papyrus_data, pl.DataFrame) else papyrus_data
        except Exception:
            return []

        # Filter by connectivity or inchikey
        q_filter = (pl.col("connectivity") == connectivity) | (pl.col("InChIKey") == inchikey)
        if min_paffinity is not None:
            q_filter = q_filter & (pl.col("pchembl_value_Mean") >= float(min_paffinity))

        matched = lf.filter(q_filter).select([
            pl.col("accession"),
            pl.col("pchembl_value_Mean"),
            pl.col("Protein_Type"),
            pl.col("target_id")
        ]).collect().to_pandas()

        if matched.empty:
            return []

        # Load protein target annotations if available
        targets_parquet = None
        if self.papyrus_source_path:
            p_targets = Path(self.papyrus_source_path) / "papyrus"
            for p in p_targets.glob("**/*protein_targets*.parquet"):
                targets_parquet = p
                break

        target_meta_map: Dict[str, Dict[str, str]] = {}
        if targets_parquet and targets_parquet.exists():
            try:
                t_df = pl.scan_parquet(targets_parquet).collect().to_pandas()
                for _, r in t_df.iterrows():
                    acc = str(r.get("UniProtID", ""))
                    if acc:
                        target_meta_map[acc] = {
                            "pref_name": str(r.get("HGNC_symbol") or r.get("UniProtID") or "Unknown"),
                            "organism": str(r.get("Organism") or "Unknown"),
                            "classification": str(r.get("Classification") or "Unknown"),
                        }
            except Exception:
                pass

        resolved: Dict[str, TargetRecord] = {}
        for _, row in matched.iterrows():
            acc = str(row["accession"])
            pchembl = float(row["pchembl_value_Mean"])
            meta = target_meta_map.get(acc, {})
            t_org = meta.get("organism", "Unknown")

            if organism and t_org != "Unknown" and organism.lower() not in t_org.lower():
                continue

            if acc not in resolved or pchembl > resolved[acc]["pchembl_query"]:
                resolved[acc] = {
                    "target_chembl_id": str(row.get("target_id", "Unknown")),
                    "uniprot_acc": acc,
                    "pref_name": meta.get("pref_name", acc),
                    "organism": t_org,
                    "target_type": str(row.get("Protein_Type") or meta.get("classification") or "SINGLE PROTEIN"),
                    "pchembl_query": pchembl
                }

        return list(resolved.values())

    def find_targets_chembl(
        self,
        inchikey: str,
        min_paffinity: float = 6.0,
        organism: Optional[str] = "Homo sapiens",
        target_type: str = "SINGLE PROTEIN"
    ) -> List[TargetRecord]:
        """Query ChEMBL REST API as fallback to identify biological receptors for a novel compound.

        Parameters
        ----------
        inchikey : str
            Full 27-character InChIKey.
        min_paffinity : float, optional
            Minimum binding affinity threshold expressed as pChEMBL (default: 6.0).
        organism : str, optional
            Target organism filter (default: 'Homo sapiens').
        target_type : str, optional
            Target classification filter (default: 'SINGLE PROTEIN').

        Returns
        -------
        targets : List[TargetRecord]
            Resolved targets from ChEMBL.
        """
        mol_url = f"{self.chembl_api}/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
        try:
            resp = self.session.get(mol_url, timeout=self.timeout)
            resp.raise_for_status()
            molecules = resp.json().get("molecules", [])
        except requests.RequestException:
            return []

        if not molecules:
            return []

        chembl_id = molecules[0].get("molecule_chembl_id")
        if not chembl_id:
            return []

        act_url = f"{self.chembl_api}/activity.json?molecule_chembl_id={chembl_id}&limit=500"
        try:
            act_resp = self.session.get(act_url, timeout=self.timeout)
            act_resp.raise_for_status()
            activities = act_resp.json().get("activities", [])
        except requests.RequestException:
            return []

        resolved_targets: Dict[str, TargetRecord] = {}
        for act in activities:
            pchembl_raw = act.get("pchembl_value")
            target_id = act.get("target_chembl_id")

            if pchembl_raw is None or not target_id:
                continue

            try:
                pchembl = float(pchembl_raw)
            except (ValueError, TypeError):
                continue

            if pchembl < min_paffinity:
                continue

            if target_id not in resolved_targets:
                t_url = f"{self.chembl_api}/target/{target_id}.json"
                try:
                    t_resp = self.session.get(t_url, timeout=self.timeout)
                    if t_resp.status_code != 200:
                        continue
                    t_data = t_resp.json()
                except requests.RequestException:
                    continue

                t_org = t_data.get("organism")
                t_cls = t_data.get("target_type")

                if organism and t_org != organism:
                    continue
                if target_type and t_cls != target_type:
                    continue

                components = t_data.get("target_components", [])
                for comp in components:
                    acc = comp.get("accession")
                    if acc and acc not in resolved_targets:
                        resolved_targets[acc] = {
                            "target_chembl_id": target_id,
                            "uniprot_acc": acc,
                            "pref_name": t_data.get("pref_name", "Unknown"),
                            "organism": t_org or "Unknown",
                            "target_type": t_cls or "Unknown",
                            "pchembl_query": pchembl
                        }

        return list(resolved_targets.values())

    def find_targets(
        self,
        inchikey: str,
        connectivity: Optional[str] = None,
        min_paffinity: float = 6.0,
        organism: Optional[str] = "Homo sapiens",
        target_type: str = "SINGLE PROTEIN"
    ) -> List[TargetRecord]:
        """Identify biological receptor targets binding the query compound.

        Prioritizes fast local Papyrus querying. If the query structure is not indexed
        in Papyrus, seamlessly falls back to the ChEMBL target resolution API.

        Parameters
        ----------
        inchikey : str
            Full 27-character InChIKey.
        connectivity : str, optional
            First 14-character connectivity hash. If None, derived from inchikey.
        min_paffinity : float, optional
            Minimum affinity threshold (default: 6.0).
        organism : str, optional
            Organism taxonomy filter (default: 'Homo sapiens').
        target_type : str, optional
            Target classification filter (default: 'SINGLE PROTEIN').

        Returns
        -------
        targets : List[TargetRecord]
            Identified target receptors.
        """
        conn = connectivity or inchikey.split("-")[0]

        # 1. Search Papyrus
        papyrus_targets = self.find_targets_papyrus(
            inchikey=inchikey,
            connectivity=conn,
            min_paffinity=min_paffinity,
            organism=organism
        )
        if papyrus_targets:
            return papyrus_targets

        # 2. Fallback to ChEMBL
        return self.find_targets_chembl(
            inchikey=inchikey,
            min_paffinity=min_paffinity,
            organism=organism,
            target_type=target_type
        )

    def get_papyrus_curated(
        self,
        target_accessions: Sequence[str],
        min_paffinity: Optional[float] = 6.0,
        quality_filter: Optional[str] = "High",
        deduplicate: bool = True,
        property_filter: Optional[ScorerPropertyFilter] = None
    ) -> pd.DataFrame:
        """Harvest ligands for specified targets exclusively from Papyrus chemogenomics.

        Parameters
        ----------
        target_accessions : sequence of str
            Target UniProt accession codes to extract from Papyrus.
        min_paffinity : float, optional
            Minimum consensus pAffinity (-log10 M) threshold (default: 6.0).
        quality_filter : str, optional
            Papyrus data quality filter (``'High'``, ``'Medium'``, or None, default: ``'High'``).
        deduplicate : bool, optional
            Whether to deduplicate molecules by SMILES (default: True).
        property_filter : ScorerPropertyFilter, optional
            Optional property filter matching DrugEx Scorers (QED, SAScore, MW, LogP, TPSA).

        Returns
        -------
        papyrus_curated : pd.DataFrame
            Curated Pandas DataFrame containing shared-target ligands.
        """
        if not target_accessions:
            return pd.DataFrame(columns=[
                "smiles",
                "SMILES",
                "InChIKey",
                "connectivity",
                "target_accession",
                "pAffinity",
                "pchembl_value_Mean",
                "Quality",
                "type",
                "source"
            ])

        try:
            import polars as pl
            from papyrus_scripts.reader import read_papyrus
        except ImportError:
            warnings.warn("papyrus_scripts (or polars) not installed. Returning empty Papyrus dataset.")
            return pd.DataFrame()

        try:
            papyrus_data = read_papyrus(
                version=self.papyrus_version,
                source_path=self.papyrus_source_path,
                plusplus=True,
                chunksize=1
            )
            papyrus_lf = papyrus_data.lazy() if isinstance(papyrus_data, pl.DataFrame) else papyrus_data
        except (OSError, ValueError, Exception) as err:
            warnings.warn(f"Unable to read local Papyrus database: {err}. Returning empty Papyrus dataset.")
            return pd.DataFrame()

        # Query filter
        query_filter = pl.col("accession").is_in(list(target_accessions))
        if quality_filter is not None:
            query_filter = query_filter & (pl.col("Quality") == quality_filter)
        if min_paffinity is not None:
            query_filter = query_filter & (pl.col("pchembl_value_Mean") >= float(min_paffinity))

        available_cols = papyrus_lf.collect_schema().names()
        select_exprs = [
            pl.col("SMILES").alias("smiles"),
            pl.col("InChIKey"),
            pl.col("connectivity"),
            pl.col("accession").alias("target_accession"),
            pl.col("pchembl_value_Mean").alias("pAffinity"),
            pl.col("pchembl_value_Mean"),
            pl.col("Quality"),
        ]
        if "Activity_class" in available_cols:
            select_exprs.append(pl.col("Activity_class").alias("type"))
        elif "type" in available_cols:
            select_exprs.append(pl.col("type"))
        else:
            select_exprs.append(pl.lit("Papyrus").alias("type"))

        if "source" in available_cols:
            select_exprs.append(pl.col("source"))
        else:
            select_exprs.append(pl.lit("Papyrus").alias("source"))

        filtered_papyrus = (
            papyrus_lf
            .filter(query_filter)
            .select(select_exprs)
            .collect()
            .to_pandas()
        )

        if filtered_papyrus.empty:
            return filtered_papyrus

        filtered_papyrus["SMILES"] = filtered_papyrus["smiles"]

        if deduplicate:
            filtered_papyrus = (
                filtered_papyrus.sort_values(by="pAffinity", ascending=False)
                .drop_duplicates(subset=["smiles"])
                .reset_index(drop=True)
            )

        # Apply Scorer property filter if configured
        if property_filter is not None:
            filtered_papyrus = property_filter.filter(filtered_papyrus)

        return filtered_papyrus

    # Backward compatibility alias
    get_outcome_2 = get_papyrus_curated

    @staticmethod
    def _print_progress(
        fraction: float,
        message: str = "",
        bar_len: int = 25,
        prefix: str = "[Papyrus Pipeline]"
    ) -> None:
        """Render an in-place updating terminal progress bar to sys.stdout."""
        fraction = min(max(fraction, 0.0), 1.0)
        filled = int(round(bar_len * fraction))
        if 0 < filled < bar_len:
            bar = "=" * (filled - 1) + ">"
        else:
            bar = "=" * filled
        bar = bar.ljust(bar_len)
        percent = int(fraction * 100)
        line = f"\r{prefix} [{bar}] {percent:3d}% | {message}"
        sys.stdout.write(line.ljust(95))
        sys.stdout.flush()
        if fraction >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def run(
        self,
        smiles: str,
        min_paffinity: float = 6.0,
        quality_filter: Optional[str] = "High",
        organism: Optional[str] = "Homo sapiens",
        min_ligands_threshold: int = 50,
        max_molecules: Optional[int] = 1000,
        auto_relax: bool = True,
        min_relax_paffinity: float = 5.0,
        deduplicate: bool = True,
        show_progress: bool = True,
        property_filter: Optional[ScorerPropertyFilter] = None,
        # Direct property filter convenience shortcuts:
        filter_properties: bool = False,
        min_qed: Optional[float] = None,
        max_sascore: Optional[float] = None,
        mw_range: Optional[Tuple[float, float]] = None,
        logp_range: Optional[Tuple[float, float]] = None,
        tpsa_range: Optional[Tuple[float, float]] = None
    ) -> PipelineResult:
        """Execute the Papyrus-only bioactivity deconvolution pipeline.

        Chains:
        1. Standardization via DrugEx DefaultStandardizer
        2. Target receptor discovery via Papyrus (with ChEMBL fallback)
        3. Extraction of shared-target ligands from Papyrus
        4. Optional Scorer property filtering (pAffinity, QED, SAScore, MW, LogP, TPSA)

        Parameters
        ----------
        smiles : str
            Raw SMILES string of the query molecule.
        min_paffinity : float, optional
            Minimum affinity threshold (default: 6.0, pIC50 >= 6.0).
        quality_filter : str, optional
            Papyrus curation quality filter (``'High'``, ``'Medium'``, or None, default: ``'High'``).
        organism : str, optional
            Target source taxonomy filter (default: ``'Homo sapiens'``).
        min_ligands_threshold : int, optional
            Minimum number of unique candidate ligands desired (default: 50).
        max_molecules : int, optional
            Upper cap on candidate molecules for training efficiency (default: 1000).
        auto_relax : bool, optional
            Whether to automatically relax pAffinity threshold if fewer candidates are found (default: True).
        min_relax_paffinity : float, optional
            Floor limit for automatic affinity relaxation (default: 5.0).
        show_progress : bool, optional
            Whether to stream in-place updating progress bar to stdout (default: True).
        property_filter : ScorerPropertyFilter, optional
            Configured property filter instance.
        filter_properties : bool, optional
            If True and `property_filter` is None, constructs a `ScorerPropertyFilter` from kwargs.
        min_qed : float, optional
            Minimum QED threshold when filter_properties is True.
        max_sascore : float, optional
            Maximum SAScore threshold when filter_properties is True.
        mw_range : tuple of (float, float), optional
            Molecular weight bounds when filter_properties is True.
        logp_range : tuple of (float, float), optional
            LogP bounds when filter_properties is True.
        tpsa_range : tuple of (float, float), optional
            TPSA bounds when filter_properties is True.

        Returns
        -------
        result : PipelineResult
            Consolidated dataclass containing `query_molecule`, `targets`, and `papyrus_curated`.
        """
        # Build property filter if requested
        active_prop_filter: Optional[ScorerPropertyFilter] = property_filter
        if active_prop_filter is None and filter_properties:
            active_prop_filter = ScorerPropertyFilter(
                min_paffinity=min_paffinity,
                min_qed=min_qed,
                max_sascore=max_sascore,
                mw_range=mw_range,
                logp_range=logp_range,
                tpsa_range=tpsa_range,
                max_molecules=max_molecules
            )
        elif active_prop_filter is not None and max_molecules is not None and active_prop_filter.max_molecules is None:
            active_prop_filter.max_molecules = max_molecules

        if show_progress:
            self._print_progress(0.10, "Step 1/3: Standardizing query structure with DrugEx...")

        # Step 1: Standardize with DrugEx DefaultStandardizer
        normalized = self.standardize_molecule(smiles)

        current_paffinity = float(min_paffinity)

        while True:
            if show_progress:
                smi_preview = normalized.canonical_smiles[:25] + ("..." if len(normalized.canonical_smiles) > 25 else "")
                self._print_progress(0.35, f"Step 2/3: Identifying targets for {smi_preview} (pAffinity >= {current_paffinity:.1f})...")

            # Step 2: Target Discovery (Papyrus first, ChEMBL fallback)
            targets = self.find_targets(
                inchikey=normalized.inchikey,
                connectivity=normalized.connectivity,
                min_paffinity=current_paffinity,
                organism=organism
            )

            target_accessions = [t["uniprot_acc"] for t in targets]

            if show_progress:
                self._print_progress(0.65, f"Step 3/3: Found {len(targets)} targets. Extracting ligands from Papyrus...")

            # Step 3: Extract from Papyrus
            papyrus_curated = self.get_papyrus_curated(
                target_accessions=target_accessions,
                min_paffinity=current_paffinity,
                quality_filter=quality_filter,
                deduplicate=deduplicate,
                property_filter=active_prop_filter
            )

            # Auto-relax affinity if below threshold
            n_candidates = len(papyrus_curated)
            if (
                auto_relax
                and (len(targets) == 0 or n_candidates < min_ligands_threshold)
                and current_paffinity > min_relax_paffinity
            ):
                next_paffinity = max(min_relax_paffinity, current_paffinity - 0.5)
                if show_progress:
                    msg = (
                        f"Found {n_candidates} candidates across {len(targets)} targets at "
                        f"pAffinity >= {current_paffinity:.1f}. Auto-relaxing to {next_paffinity:.1f}..."
                    )
                    self._print_progress(0.35, msg)
                current_paffinity = next_paffinity
                continue

            break

        if show_progress:
            self._print_progress(
                1.00,
                f"Done! {len(targets)} targets | {len(papyrus_curated)} Papyrus curated ligands"
            )

        return PipelineResult(
            query_molecule=normalized,
            targets=targets,
            papyrus_curated=papyrus_curated
        )
