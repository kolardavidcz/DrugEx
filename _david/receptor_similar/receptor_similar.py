"""End-to-End Molecular Bioactivity & Target Deconvolution Pipeline.

This module automates the extraction and curation of shared-target ligand spaces:
1. Standardizes a query molecule (SMILES -> canonical SMILES, InChIKey, connectivity).
2. Identifies all biological receptors (targets / UniProt accessions) binding the molecule via ChEMBL.
3. Harvests all compounds active against one or more of these identified receptors (polypharmacology).
4. Formats and exports a universal bioactivity matrix (`public_bioactivities` - broad coverage for AI).
5. Cross-references and enriches the dataset against Papyrus with standardized pAffinity,
   quality filters, and pre-computed structural features (`papyrus_curated` - curated benchmark for AI).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


class NormalizedMolecule(NamedTuple):
    """Container for standardized chemical identifier representations.

    Attributes
    ----------
    canonical_smiles : str
        Kekulized / standardized canonical SMILES string without salts or solvates.
    inchikey : str
        Full 27-character IUPAC International Chemical Identifier hash (capturing stereochemistry).
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
        ChEMBL classification type (e.g. ``'SINGLE PROTEIN'``).
    pchembl_query : float
        Query molecule binding affinity expressed as standard -log10(molar) value.
    """

    target_chembl_id: str
    uniprot_acc: str
    pref_name: str
    organism: str
    target_type: str
    pchembl_query: float


@dataclass(frozen=True)
class PipelineResult:
    """Consolidated outputs from an end-to-end bioactivity query pipeline run.

    Attributes
    ----------
    query_molecule : NormalizedMolecule
        Standardized query structure and associated InChIKey hashes.
    targets : List[TargetRecord]
        List of identified target receptors interacting with the query compound.
    public_bioactivities : pd.DataFrame
        Universal multi-target bioactivity matrix harvested across public repositories (Outcome 1).
    papyrus_curated : pd.DataFrame
        Curated, quality-filtered benchmark subset cross-referenced against Papyrus (Outcome 2).
    """

    query_molecule: NormalizedMolecule
    targets: List[TargetRecord]
    public_bioactivities: pd.DataFrame
    papyrus_curated: pd.DataFrame

    @property
    def outcome_1(self) -> pd.DataFrame:
        """Alias for `public_bioactivities` maintaining backward compatibility."""
        return self.public_bioactivities

    @property
    def outcome_2(self) -> pd.DataFrame:
        """Alias for `papyrus_curated` maintaining backward compatibility."""
        return self.papyrus_curated


class MoleculeBioactivityPipeline:
    """Automated multi-target bioactivity extraction and Papyrus enrichment pipeline.

    Connects ligand structural input to public bioactivity registries (ChEMBL REST API)
    and curated high-performance chemogenomic datasets (Papyrus chemogenomic engine).

    Parameters
    ----------
    papyrus_version : str, optional
        Version tag of the Papyrus database to read (default: ``'latest'``).
    timeout : int, optional
        HTTP network request timeout in seconds for API calls (default: 15).
    session : requests.Session, optional
        Custom HTTP session for connection pooling. If None, a default session is created.

    Examples
    --------
    >>> from receptor_similar import MoleculeBioactivityPipeline
    >>> pipeline = MoleculeBioactivityPipeline()
    >>> # Example using Imatinib
    >>> smi = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(C)CC5"
    >>> result = pipeline.run(smi, min_paffinity=6.5)
    >>> print(f"Identified {len(result.targets)} target receptors")
    >>> print(f"Public bioactivities (Outcome 1): {len(result.public_bioactivities)}")
    >>> print(f"Papyrus curated (Outcome 2): {len(result.papyrus_curated)}")
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
        session: Optional[requests.Session] = None
    ) -> None:
        self.papyrus_version: str = papyrus_version
        self.timeout: int = timeout
        self.chembl_api: str = "https://www.ebi.ac.uk/chembl/api/data"
        self.session: requests.Session = session if session is not None else requests.Session()
        self.papyrus_source_path: Optional[str] = (
            papyrus_source_path if papyrus_source_path is not None else self._resolve_papyrus_path()
        )

    def standardize_molecule(self, smiles: str) -> NormalizedMolecule:
        """Normalize chemical structure, remove counter-ions, and generate InChIKey hashes.

        Applies RDKit `rdMolStandardize` routines (cleanup, parent fragment selection,
        neutralization via `Uncharger`, and canonical tautomerization via `TautomerEnumerator`)
        prior to database querying.

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
            If the supplied SMILES string cannot be parsed into a valid RDKit molecule.

        Examples
        --------
        >>> pipeline = MoleculeBioactivityPipeline()
        >>> mol = pipeline.standardize_molecule("CC(=O)Oc1ccccc1C(=O)O")
        >>> mol.inchikey
        'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES string into RDKit molecule: '{smiles}'")

        # Desalt, neutralize, and canonicalize tautomer
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean_mol)
        uncharged = rdMolStandardize.Uncharger().uncharge(parent)
        can_mol = rdMolStandardize.TautomerEnumerator().Canonicalize(uncharged)

        can_smiles = Chem.MolToSmiles(can_mol, isomericSmiles=True, canonical=True)
        inchikey = Chem.MolToInchiKey(can_mol)
        connectivity = inchikey.split("-")[0]

        return NormalizedMolecule(
            canonical_smiles=can_smiles,
            inchikey=inchikey,
            connectivity=connectivity
        )

    def find_targets(
        self,
        inchikey: str,
        min_paffinity: float = 6.0,
        organism: Optional[str] = "Homo sapiens",
        target_type: str = "SINGLE PROTEIN"
    ) -> List[TargetRecord]:
        """Query ChEMBL for biological receptors that bind the specified molecule.

        Queries the ChEMBL REST API by InChIKey, retrieves all published bioactivity
        assays meeting the affinity cutoff, and resolves target metadata into primary
        UniProt accession codes.

        Parameters
        ----------
        inchikey : str
            Full 27-character InChIKey of the query compound.
        min_paffinity : float, optional
            Minimum binding affinity threshold expressed as pChEMBL (-log10 Molar, default: 6.0,
            corresponding to Ki/IC50 <= 1.0 uM).
        organism : str, optional
            Target source organism filter (default: ``'Homo sapiens'``). Set to None to include all.
        target_type : str, optional
            ChEMBL target classification filter (default: ``'SINGLE PROTEIN'``).

        Returns
        -------
        targets : list of TargetRecord
            List of unique target receptor dictionaries containing UniProt accessions,
            ChEMBL IDs, preferred names, and query binding affinities.

        Raises
        ------
        requests.RequestException
            If the remote ChEMBL API is unreachable or returns an HTTP error status.

        Notes
        -----
        pChEMBL values standardize varying bioactivity types (Ki, Kd, IC50, EC50) into
        a logarithmic molar scale: :math:`-\\log_{10}(\\text{Value in M})`.
        """
        mol_url = f"{self.chembl_api}/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
        resp = self.session.get(mol_url, timeout=self.timeout)
        resp.raise_for_status()
        molecules = resp.json().get("molecules", [])
        if not molecules:
            return []

        chembl_id = molecules[0].get("molecule_chembl_id")
        if not chembl_id:
            return []

        # Fetch bioactivities for this molecule
        act_url = f"{self.chembl_api}/activity.json?molecule_chembl_id={chembl_id}&limit=500"
        act_resp = self.session.get(act_url, timeout=self.timeout)
        act_resp.raise_for_status()
        activities = act_resp.json().get("activities", [])

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

            # Query target component details if not yet resolved
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

    def get_public_bioactivities(
        self,
        target_accessions: Sequence[str],
        limit_per_target: int = 500,
        min_paffinity: float = 5.0,
        progress_callback: Optional[Any] = None
    ) -> pd.DataFrame:
        """Harvest shared-target ligands across public assays (Universal AI Dataset - Outcome 1).

        Gathers all molecules documented to bind any of the target receptors, capturing
        polypharmacological cross-reactivity and maximal chemical space diversity.

        Parameters
        ----------
        target_accessions : sequence of str
            List of primary UniProt accessions identifying target receptors.
        limit_per_target : int, optional
            Maximum number of bioactivity records retrieved per target receptor (default: 500).
        min_paffinity : float, optional
            Minimum pChEMBL affinity cutoff for inclusion (default: 5.0, corresponding to 10 uM).
        progress_callback : callable, optional
            Optional callback invoked as ``callback(current_idx, total_targets, accession, records_count)``.

        Returns
        -------
        public_bioactivities : pd.DataFrame
            Pandas DataFrame containing:
            - ``'smiles'`` : Canonical SMILES representation.
            - ``'molecule_chembl_id'`` : ChEMBL compound identifier.
            - ``'target_accession'`` : Interacting UniProt accession code.
            - ``'pAffinity'`` : Standardized -log10(Molar) binding value.
            - ``'type'`` : Bioactivity measurement type (e.g. Ki, IC50, Kd).
            - ``'source'`` : Provenance registry tag (``'ChEMBL_API'``).

        Notes
        -----
        Outcome 1 prioritizes chemical diversity and dataset breadth. It includes edge cases,
        differing assay designs, and broad structural variations suited for self-supervised
        pre-training, graph representations, and exploratory ligand screening.
        """
        all_records: List[Dict[str, Any]] = []
        total_targets = len(target_accessions)

        for idx, acc in enumerate(target_accessions, 1):
            if progress_callback is not None:
                progress_callback(idx, total_targets, acc, len(all_records))

            url = (
                f"{self.chembl_api}/activity.json?target_components__accession={acc}"
                f"&pchembl_value__gte={min_paffinity}&limit={limit_per_target}"
            )
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code != 200:
                    continue
                acts = r.json().get("activities", [])
            except requests.RequestException:
                continue

            for a in acts:
                smi = a.get("canonical_smiles")
                val_raw = a.get("pchembl_value")
                mol_id = a.get("molecule_chembl_id")

                if not smi or val_raw is None:
                    continue

                try:
                    pchembl = float(val_raw)
                except (ValueError, TypeError):
                    continue

                if pchembl < min_paffinity:
                    continue

                all_records.append({
                    "smiles": smi,
                    "molecule_chembl_id": mol_id or "Unknown",
                    "target_accession": acc,
                    "pAffinity": pchembl,
                    "type": a.get("standard_type", "Unknown"),
                    "source": "ChEMBL_API"
                })

        if not all_records:
            return pd.DataFrame(columns=[
                "smiles",
                "SMILES",
                "molecule_chembl_id",
                "target_accession",
                "pAffinity",
                "type",
                "source"
            ])

        df = pd.DataFrame(all_records)
        df["SMILES"] = df["smiles"]
        return df

    # Backward compatibility alias
    get_outcome_1 = get_public_bioactivities

    def get_papyrus_curated(
        self,
        target_accessions: Sequence[str],
        min_paffinity: Optional[float] = 6.0,
        quality_filter: Optional[str] = "High",
        public_bioactivities_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Harvest and enrich dataset with curated Papyrus benchmark data (Outcome 2).

        Filters the shared-target ligands against the pre-curated Papyrus chemogenomic
        database using high-performance Polars lazy query execution. Enriches each record
        with consensus affinities, data quality flags, and standardized stereochemistry.

        Parameters
        ----------
        target_accessions : sequence of str
            Target UniProt accession codes to filter against in Papyrus.
        min_paffinity : float, optional
            Minimum consensus pAffinity (-log10 M) threshold (default: 6.0).
        quality_filter : str, optional
            Data curation reliability threshold (``'High'``, ``'Medium'``, or None).
            When ``'High'`` (default), only rigorously validated assays with confirmed
            chemical structures and non-ambiguous endpoints are retained.
        public_bioactivities_df : pd.DataFrame, optional
            Optional public bioactivity DataFrame for backward compatibility.

        Returns
        -------
        papyrus_curated : pd.DataFrame
            Curated Pandas DataFrame containing harmonized Papyrus bioactivity columns:
            - ``'smiles'`` : Canonical SMILES string.
            - ``'SMILES'`` : Canonical SMILES string (alias for DrugEx pipelines).
            - ``'InChIKey'`` : Fully defined stereochemical InChIKey.
            - ``'connectivity'`` : Flat InChIKey connectivity scaffold.
            - ``'target_accession'`` : Validated UniProt protein accession code.
            - ``'pAffinity'`` : Multi-assay consensus pAffinity (-log10 M).
            - ``'pchembl_value_Mean'`` : Multi-assay consensus pAffinity (-log10 M).
            - ``'Quality'`` : Papyrus curation tier (e.g. 'High').
            - ``'type'`` : Standardized measurement class.
            - ``'organism'`` : Validated organism species.
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
                "organism"
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

        # Apply target, affinity, and quality filtering lazily using Polars expression engine
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

        if not filtered_papyrus.empty and "smiles" in filtered_papyrus.columns:
            filtered_papyrus["SMILES"] = filtered_papyrus["smiles"]

        return filtered_papyrus

    # Backward compatibility alias
    get_outcome_2 = get_papyrus_curated

    @staticmethod
    def _print_progress(
        fraction: float,
        message: str = "",
        bar_len: int = 25,
        prefix: str = "[Pipeline]"
    ) -> None:
        """Render an in-place updating terminal progress bar to sys.stdout using carriage return (\\r).

        Parameters
        ----------
        fraction : float
            Current progress fraction between 0.0 and 1.0.
        message : str
            Status message to display beside the progress bar.
        bar_len : int
            Width in characters of the progress bar itself.
        prefix : str
            Prefix label for the bar.
        """
        fraction = min(max(fraction, 0.0), 1.0)
        filled = int(round(bar_len * fraction))
        if 0 < filled < bar_len:
            bar = "=" * (filled - 1) + ">"
        else:
            bar = "=" * filled
        bar = bar.ljust(bar_len)
        percent = int(fraction * 100)
        # Use carriage return \r to overwrite line in terminal stdout
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
        limit_per_target: int = 500,
        quality_filter: Optional[str] = "High",
        organism: Optional[str] = "Homo sapiens",
        min_ligands_threshold: int = 1000,
        auto_relax: bool = True,
        min_relax_paffinity: float = 5.0,
        show_progress: bool = True
    ) -> PipelineResult:
        """Execute the complete end-to-end bioactivity deconvolution pipeline.

        Chains Step 1 (standardization) -> Step 2 (target identification) ->
        Step 3 & 4 (shared-target ligand harvesting / public_bioactivities) ->
        Step 5 (Papyrus cross-referencing / papyrus_curated).

        If the affinity threshold is too strict to yield enough candidate ligands
        (fewer than `min_ligands_threshold`), the pipeline can automatically relax
        the affinity threshold progressively down to `min_relax_paffinity`.

        Parameters
        ----------
        smiles : str
            Raw SMILES string of the query molecule.
        min_paffinity : float, optional
            Minimum pChEMBL affinity cutoff for receptor interaction (default: 6.0).
        limit_per_target : int, optional
            Maximum number of ligands harvested per identified receptor (default: 500).
        quality_filter : str, optional
            Papyrus data quality filter (``'High'``, ``'Medium'``, or None, default: ``'High'``).
        organism : str, optional
            Target source taxonomy filter (default: ``'Homo sapiens'``).
        min_ligands_threshold : int, optional
            Minimum number of unique candidate ligands desired (default: 1000).
        auto_relax : bool, optional
            Whether to automatically relax pAffinity threshold if fewer candidates are found (default: True).
        min_relax_paffinity : float, optional
            Floor limit for automatic affinity relaxation (default: 5.0).
        show_progress : bool, optional
            Whether to stream in-place updating progress bar to stdout (default: True).

        Returns
        -------
        result : PipelineResult
            Consolidated dataclass containing:
            - ``query_molecule`` : Standardized molecule structure and InChIKeys.
            - ``targets`` : Resolved biological receptors with UniProt accessions.
            - ``public_bioactivities`` : Universal bioactivity DataFrame (Outcome 1 - broad coverage).
            - ``papyrus_curated`` : Curated Papyrus benchmark DataFrame (Outcome 2 - high-fidelity ML).
        """
        if show_progress:
            self._print_progress(0.05, "Step 1/5: Standardizing molecule structure...")

        # Step 1: Standardize
        normalized = self.standardize_molecule(smiles)

        current_paffinity = float(min_paffinity)

        while True:
            if show_progress:
                smi_preview = normalized.canonical_smiles[:25] + ("..." if len(normalized.canonical_smiles) > 25 else "")
                self._print_progress(0.20, f"Step 2/5: Querying ChEMBL for targets of {smi_preview} (pAffinity >= {current_paffinity:.1f})...")

            # Step 2: Discover Receptors
            targets = self.find_targets(
                inchikey=normalized.inchikey,
                min_paffinity=current_paffinity,
                organism=organism
            )

            target_accessions = [t["uniprot_acc"] for t in targets]

            if show_progress:
                self._print_progress(0.40, f"Step 3/5: Found {len(targets)} targets. Harvesting ligands from ChEMBL...")

            def _on_target_progress(idx: int, total: int, acc: str, n_records: int) -> None:
                if show_progress and total > 0:
                    frac = 0.40 + 0.35 * (idx / total)
                    self._print_progress(frac, f"Harvesting target {idx}/{total} ({acc}) - {n_records} ligands collected")

            # Step 3 & 4: Harvest Shared-Target Ligands (public_bioactivities)
            public_bioactivities = self.get_public_bioactivities(
                target_accessions=target_accessions,
                limit_per_target=limit_per_target,
                min_paffinity=max(current_paffinity - 1.0, 4.0),
                progress_callback=_on_target_progress if show_progress else None
            )

            if show_progress:
                self._print_progress(0.75, f"Step 4/5: Enriching {len(target_accessions)} targets from curated Papyrus chemogenomics...")

            # Step 5: Enrich via Papyrus (papyrus_curated)
            papyrus_curated = self.get_papyrus_curated(
                target_accessions=target_accessions,
                min_paffinity=current_paffinity,
                quality_filter=quality_filter,
                public_bioactivities_df=public_bioactivities
            )

            # Check total candidate yield
            unique_candidates: Set[str] = set()
            if not public_bioactivities.empty and "smiles" in public_bioactivities.columns:
                unique_candidates.update(public_bioactivities["smiles"].dropna())
            if not papyrus_curated.empty and "smiles" in papyrus_curated.columns:
                unique_candidates.update(papyrus_curated["smiles"].dropna())

            # If threshold was too strict to yield enough candidates, relax threshold
            if (
                auto_relax
                and (len(targets) == 0 or len(unique_candidates) < min_ligands_threshold)
                and current_paffinity > min_relax_paffinity
            ):
                next_paffinity = max(min_relax_paffinity, current_paffinity - 0.5)
                if show_progress:
                    msg = (
                        f"Retrieved {len(unique_candidates)} candidates across {len(targets)} targets at "
                        f"pAffinity >= {current_paffinity:.1f}. Auto-relaxing threshold to {next_paffinity:.1f}..."
                    )
                    self._print_progress(0.20, msg)
                current_paffinity = next_paffinity
                continue

            break

        if show_progress:
            self._print_progress(
                1.00,
                f"Step 5/5: Done! {len(targets)} targets | {len(public_bioactivities)} public ligands | {len(papyrus_curated)} Papyrus curated"
            )

        return PipelineResult(
            query_molecule=normalized,
            targets=targets,
            public_bioactivities=public_bioactivities,
            papyrus_curated=papyrus_curated
        )

