"""End-to-End Molecular Bioactivity & Target Deconvolution Pipeline (Papyrus-Only).

This module automates the extraction and curation of shared-target ligand spaces:
1. Standardizes query molecule(s) using DrugEx DefaultStandardizer (SMILES -> canonical SMILES, InChIKey, connectivity).
2. Derives an automatic MolecularPropertyProfile across input molecule(s) with adaptive CV-based weighting
   (conserved properties get tighter windows and higher ranking weights; variable properties get relaxed windows).
3. Deconvolutes biological receptors (UniProt accessions) directly within Papyrus (strictly no ChEMBL unless opted-in).
4. Harvests active ligands from Papyrus chemogenomics, filters via dynamic tolerance windows, and ranks candidates
   by weighted property similarity to the input profile.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BertzCT

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
        Target identifier (e.g. ``'CHEMBL2111414'``).
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
class PropertyProfileEntry:
    """Statistical summary and tolerance bounds for a single molecular property.

    Attributes
    ----------
    name : str
        Human-readable property name (e.g. 'MW', 'logP', 'HBA').
    mean : float
        Mean value across input molecule(s) (or exact value if single molecule).
    std : float
        Sample standard deviation across input molecule(s).
    cv : float
        Coefficient of variation (std / (|mean| + eps)).
    weight : float
        Normalized ranking weight in property similarity distance (sums to 1.0 across profile).
    is_conserved : bool
        True if property displays high conservation (low variance) across input molecules.
    tolerance_min : float
        Lower tolerance window bound for filtering.
    tolerance_max : float
        Upper tolerance window bound for filtering.
    """

    name: str
    mean: float
    std: float
    cv: float
    weight: float
    is_conserved: bool
    tolerance_min: float
    tolerance_max: float


@dataclass
class MolecularPropertyProfile:
    """Multidimensional physicochemical property profile derived from input molecule(s).

    Extracts core properties (MW, logP, TPSA, HBA, HBD, RotBonds, AromaticRings, FractionCSP3, QED),
    measures variation across input molecules, adaptively weights conserved vs. variable features,
    and calculates similarity distance to rank candidate ligands.
    """

    entries: Dict[str, PropertyProfileEntry] = field(default_factory=dict)
    n_inputs: int = 1
    tolerance_factor: float = 1.0

    # Default absolute buffer scales for properties
    _DEFAULT_SCALES: ClassVar[Dict[str, float]] = {
        "MW": 50.0,
        "logP": 1.0,
        "TPSA": 20.0,
        "HBA": 1.0,
        "HBD": 1.0,
        "RotBonds": 2.0,
        "AromaticRings": 1.0,
        "FractionCSP3": 0.15,
        "QED": 0.15,
    }

    @staticmethod
    def calculate_properties(mol: Chem.Mol) -> Dict[str, float]:
        """Compute core physicochemical descriptor properties for a single molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        props : Dict[str, float]
            Dictionary containing MW, logP, TPSA, HBA, HBD, RotBonds, AromaticRings, FractionCSP3, QED.
        """
        return {
            "MW": float(Descriptors.MolWt(mol)),
            "logP": float(Crippen.MolLogP(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "HBA": float(rdMolDescriptors.CalcNumLipinskiHBA(mol)),
            "HBD": float(rdMolDescriptors.CalcNumLipinskiHBD(mol)),
            "RotBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "QED": float(QED.qed(mol)),
        }

    @classmethod
    def from_molecules(
        cls,
        molecules: Sequence[NormalizedMolecule],
        tolerance_factor: float = 1.0
    ) -> MolecularPropertyProfile:
        """Derive property profile and adaptive weights from one or more input molecules.

        Parameters
        ----------
        molecules : sequence of NormalizedMolecule
            Standardized input molecules.
        tolerance_factor : float, optional
            Multiplier for tolerance window width (default: 1.0; <1.0 = stricter, >1.0 = looser).

        Returns
        -------
        profile : MolecularPropertyProfile
            Configured property profile instance.
        """
        all_props: List[Dict[str, float]] = []
        for norm in molecules:
            mol = Chem.MolFromSmiles(norm.canonical_smiles)
            if mol is not None:
                all_props.append(cls.calculate_properties(mol))

        if not all_props:
            raise ValueError("No valid molecules provided to derive property profile.")

        df_props = pd.DataFrame(all_props)
        prop_names = list(df_props.columns)
        n_mols = len(df_props)

        entries: Dict[str, PropertyProfileEntry] = {}
        raw_weights: Dict[str, float] = {}

        for col in prop_names:
            vals = df_props[col].values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals)) if n_mols > 1 else 0.0
            cv_val = std_val / (abs(mean_val) + 1e-4) if mean_val != 0 else std_val

            # Determine conservation status
            # Integer count features (HBA, HBD, AromaticRings) are conserved if std <= 0.5
            is_count = col in ("HBA", "HBD", "AromaticRings", "RotBonds")
            if n_mols == 1:
                is_conserved = True
                raw_weight = 1.0
            else:
                if is_count:
                    is_conserved = std_val <= 0.5 or cv_val <= 0.15
                else:
                    is_conserved = cv_val <= 0.15

                # Adaptive weighting: conserved properties get higher weight
                raw_weight = 1.0 / (cv_val + 0.10)

            raw_weights[col] = raw_weight

            # Calculate tolerance windows
            scale = cls._DEFAULT_SCALES.get(col, 1.0)
            if n_mols == 1:
                # Single molecule tolerance
                if col == "MW":
                    buffer = max(50.0, abs(mean_val) * 0.20) * tolerance_factor
                elif col == "logP":
                    buffer = 1.2 * tolerance_factor
                elif col == "TPSA":
                    buffer = max(25.0, abs(mean_val) * 0.25) * tolerance_factor
                elif is_count:
                    buffer = max(1.0, round(scale * tolerance_factor))
                elif col in ("FractionCSP3", "QED"):
                    buffer = 0.20 * tolerance_factor
                else:
                    buffer = scale * tolerance_factor

                t_min = max(0.0, mean_val - buffer) if col not in ("logP",) else mean_val - buffer
                t_max = mean_val + buffer
                if col in ("FractionCSP3", "QED"):
                    t_min = max(0.0, t_min)
                    t_max = min(1.0, t_max)
            else:
                # Multiple molecules: envelope around [min, max] with adaptive buffer
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))
                buffer = (0.5 * scale if is_conserved else 1.2 * scale) * tolerance_factor

                t_min = max(0.0, min_val - buffer) if col not in ("logP",) else min_val - buffer
                t_max = max_val + buffer
                if col in ("FractionCSP3", "QED"):
                    t_min = max(0.0, t_min)
                    t_max = min(1.0, t_max)

            entries[col] = PropertyProfileEntry(
                name=col,
                mean=mean_val,
                std=std_val,
                cv=cv_val,
                weight=0.0,  # updated below after normalization
                is_conserved=is_conserved,
                tolerance_min=t_min,
                tolerance_max=t_max
            )

        # Normalize weights so they sum to 1.0
        total_weight = sum(raw_weights.values())
        for col, entry in entries.items():
            entry.weight = raw_weights[col] / total_weight

        return cls(entries=entries, n_inputs=n_mols, tolerance_factor=tolerance_factor)

    def is_within_tolerance(self, props: Dict[str, float]) -> bool:
        """Check if a candidate molecule's properties fall within all tolerance windows.

        Parameters
        ----------
        props : Dict[str, float]
            Properties of the candidate molecule.

        Returns
        -------
        within : bool
            True if all properties fall within the defined tolerance bounds.
        """
        for name, entry in self.entries.items():
            val = props.get(name)
            if val is None:
                continue
            if not (entry.tolerance_min <= val <= entry.tolerance_max):
                return False
        return True

    def compute_similarity(self, props: Dict[str, float]) -> float:
        """Compute weighted normalized similarity score to the profile centroid.

        Parameters
        ----------
        props : Dict[str, float]
            Properties of the candidate molecule.

        Returns
        -------
        similarity : float
            Similarity score S in (0.0, 1.0], where 1.0 indicates exact identity with centroid.
        """
        total_dist = 0.0
        for name, entry in self.entries.items():
            val = props.get(name)
            if val is None:
                continue
            scale = self._DEFAULT_SCALES.get(name, 1.0)
            diff = abs(val - entry.mean) / scale
            total_dist += entry.weight * diff

        # Invert distance into bounded similarity metric
        return float(1.0 / (1.0 + total_dist))

    def print_summary(self) -> None:
        """Print a concise terminal summary table of the derived property profile."""
        header = f"Molecular Property Profile (Derived from {self.n_inputs} input molecule{'s' if self.n_inputs > 1 else ''})"
        print("=" * 86)
        print(header.center(86))
        print("=" * 86)
        print(f"{'Property':<14} {'Mean/Ref':>10} {'StdDev':>8} {'CV':>7} {'Weight':>8} {'Status':<11} {'Tolerance Window':>22}")
        print("-" * 86)
        for name, entry in self.entries.items():
            status = "Conserved" if entry.is_conserved else "Variable"
            window = f"[{entry.tolerance_min:7.2f}, {entry.tolerance_max:7.2f}]"
            print(
                f"{name:<14} {entry.mean:>10.2f} {entry.std:>8.2f} {entry.cv:>7.2f} "
                f"{entry.weight:>8.3f} {status:<11} {window:>22}"
            )
        print("=" * 86)


@dataclass
class AdaptiveCandidateSelector:
    """Filters candidate ligands via dynamic profile tolerance windows and ranks by property similarity.

    Parameters
    ----------
    profile : MolecularPropertyProfile
        The derived property profile.
    max_molecules : Optional[int]
        Maximum number of candidate molecules to retain (default: 1000).
    max_sascore : Optional[float]
        Optional upper-bound sanity cap on Synthetic Accessibility score (default: None).
    max_bertz : Optional[float]
        Optional upper-bound sanity cap on BertzCT molecular complexity (default: None).
    require_valid_rdkit : bool
        If True, discards candidate SMILES failing RDKit sanitization (default: True).
    """

    profile: MolecularPropertyProfile
    max_molecules: Optional[int] = 1000
    max_sascore: Optional[float] = None
    max_bertz: Optional[float] = None
    hard_filter: bool = False
    require_valid_rdkit: bool = True

    def select(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Filter and rank candidate molecules.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of molecules containing 'smiles' or 'SMILES'.
        verbose : bool, optional
            Whether to log selection statistics.

        Returns
        -------
        ranked_df : pd.DataFrame
            Filtered DataFrame sorted by property_similarity and pAffinity descending.
        """
        if df.empty:
            return df.copy()

        smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
        if smiles_col not in df.columns:
            return df.copy()

        n_initial = len(df)
        records = df.to_dict(orient="records")
        accepted: List[Dict[str, Any]] = []

        for rec in records:
            smi = rec.get(smiles_col)
            if not isinstance(smi, str) or len(smi) < 2:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                if self.require_valid_rdkit:
                    continue
                else:
                    accepted.append(rec)
                    continue

            # Compute profile properties
            props = self.profile.calculate_properties(mol)

            # Record tolerance status and optionally apply hard window filter
            is_within = self.profile.is_within_tolerance(props)
            props["within_tolerance"] = is_within

            if self.hard_filter and not is_within:
                continue

            # Optional upper-bound sanity caps (SA, Bertz)
            if self.max_sascore is not None:
                try:
                    sa_val = float(sascorer.calculateScore(mol))
                    props["SAScore"] = sa_val
                    if sa_val > self.max_sascore:
                        continue
                except Exception:
                    pass

            if self.max_bertz is not None:
                try:
                    bertz_val = float(BertzCT(mol))
                    props["BertzCT"] = bertz_val
                    if bertz_val > self.max_bertz:
                        continue
                except Exception:
                    pass

            # Compute property similarity to input profile
            sim = self.profile.compute_similarity(props)
            props["property_similarity"] = sim

            # Merge into record
            rec.update(props)
            accepted.append(rec)

        if not accepted:
            cols = list(df.columns)
            for c in ["MW", "logP", "TPSA", "HBA", "HBD", "RotBonds", "AromaticRings", "FractionCSP3", "QED", "property_similarity"]:
                if c not in cols:
                    cols.append(c)
            return pd.DataFrame(columns=cols)

        out_df = pd.DataFrame(accepted)

        # Sort primarily by property_similarity, secondarily by pAffinity
        sort_cols = ["property_similarity"]
        ascending = [False]
        if "pAffinity" in out_df.columns:
            sort_cols.append("pAffinity")
            ascending.append(False)

        out_df = out_df.sort_values(by=sort_cols, ascending=ascending)

        # Cap max molecules if requested
        if self.max_molecules is not None and len(out_df) > self.max_molecules:
            out_df = out_df.iloc[: self.max_molecules]

        out_df = out_df.reset_index(drop=True)

        if verbose:
            print(
                f"[AdaptiveCandidateSelector] Retained {len(out_df)} / {n_initial} candidate molecules "
                f"matching input property profile."
            )

        return out_df


@dataclass(frozen=True)
class PipelineResult:
    """Consolidated outputs from an end-to-end bioactivity query pipeline run.

    Attributes
    ----------
    query_molecules : List[NormalizedMolecule]
        List of standardized input query structures and InChIKeys.
    targets : List[TargetRecord]
        List of identified target receptors interacting with the query compounds.
    property_profile : MolecularPropertyProfile
        The derived multidimensional physicochemical property profile.
    papyrus_curated : pd.DataFrame
        Curated benchmark subset harvested exclusively from Papyrus chemogenomics,
        filtered and ranked by property similarity to the input profile.
    """

    query_molecules: List[NormalizedMolecule]
    targets: List[TargetRecord]
    property_profile: MolecularPropertyProfile
    papyrus_curated: pd.DataFrame

    @property
    def query_molecule(self) -> NormalizedMolecule:
        """First standardized query molecule (for backward compatibility)."""
        return self.query_molecules[0] if self.query_molecules else NormalizedMolecule("", "", "")

    @property
    def molecules(self) -> pd.DataFrame:
        """Primary DataFrame of harvested and ranked candidate molecules."""
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

    Standardizes query ligand(s), derives an automatic physicochemical property profile,
    discovers biological targets directly within Papyrus (no ChEMBL unless explicitly enabled),
    and extracts shared-target active molecules ranked by property similarity.

    Parameters
    ----------
    papyrus_version : str, optional
        Version tag of the Papyrus database to read (default: ``'latest'``).
    papyrus_source_path : str, optional
        Explicit path to Papyrus repository folder. If None, automatically detected.
    allow_chembl_fallback : bool, optional
        Whether to query remote ChEMBL API if a molecule is not indexed in Papyrus (default: False).
    timeout : int, optional
        HTTP request timeout in seconds for optional ChEMBL queries (default: 15).
    session : requests.Session, optional
        Custom HTTP session for connection pooling. If None, a default session is created.
    auto_download_papyrus : bool, optional
        Whether to automatically download Papyrus dataset if missing on disk (default: True).
    """

    @staticmethod
    def _resolve_papyrus_path() -> Optional[str]:
        """Automatically locate local Papyrus repository cache if present on disk."""
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
        allow_chembl_fallback: bool = False,
        timeout: int = 15,
        session: Optional[requests.Session] = None,
        auto_download_papyrus: bool = True
    ) -> None:
        self.papyrus_version: str = papyrus_version
        self.allow_chembl_fallback: bool = allow_chembl_fallback
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

        Parameters
        ----------
        smiles : str
            Input raw SMILES representation of the query molecule.

        Returns
        -------
        normalized : NormalizedMolecule
            Named tuple containing ``(canonical_smiles, inchikey, connectivity)``.
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
        """Query ChEMBL REST API only if explicitly allowed by user.

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
        target_type: str = "SINGLE PROTEIN",
        allow_chembl_fallback: Optional[bool] = None
    ) -> List[TargetRecord]:
        """Identify biological receptor targets binding a query compound.

        Queries Papyrus directly. Remote ChEMBL query is strictly disabled unless
        allow_chembl_fallback is True.

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
        allow_chembl_fallback : bool, optional
            Override instance allow_chembl_fallback setting.

        Returns
        -------
        targets : List[TargetRecord]
            Identified target receptors.
        """
        conn = connectivity or inchikey.split("-")[0]

        # 1. Search Papyrus strictly
        papyrus_targets = self.find_targets_papyrus(
            inchikey=inchikey,
            connectivity=conn,
            min_paffinity=min_paffinity,
            organism=organism
        )
        if papyrus_targets:
            return papyrus_targets

        # 2. Only query ChEMBL if explicitly allowed
        use_chembl = self.allow_chembl_fallback if allow_chembl_fallback is None else allow_chembl_fallback
        if use_chembl:
            return self.find_targets_chembl(
                inchikey=inchikey,
                min_paffinity=min_paffinity,
                organism=organism,
                target_type=target_type
            )

        return []

    def get_papyrus_curated(
        self,
        target_accessions: Sequence[str],
        min_paffinity: Optional[float] = 6.0,
        quality_filter: Optional[str] = "High",
        deduplicate: bool = True
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
        smiles: str | Sequence[str],
        min_paffinity: float = 6.0,
        quality_filter: Optional[str] = "High",
        organism: Optional[str] = "Homo sapiens",
        min_target_consensus: int = 1,
        tolerance_factor: float = 1.0,
        max_molecules: Optional[int] = 1000,
        max_sascore: Optional[float] = None,
        max_bertz: Optional[float] = None,
        hard_filter: bool = False,
        allow_chembl_fallback: Optional[bool] = None,
        auto_relax: bool = False,
        min_relax_paffinity: float = 5.0,
        min_ligands_threshold: int = 50,
        deduplicate: bool = True,
        show_progress: bool = True
    ) -> PipelineResult:
        """Execute the automatic property-profiled Papyrus bioactivity pipeline.

        Chains:
        1. Standardization of query molecule(s) via DrugEx DefaultStandardizer
        2. Automatic extraction of MolecularPropertyProfile and adaptive CV weighting
        3. Target receptor discovery in Papyrus (union across query inputs; no ChEMBL unless specified)
        4. Extraction of shared-target active ligands from Papyrus
        5. AdaptiveCandidateSelector: pure soft ranking by property similarity (or dynamic window gating if hard_filter=True)

        Parameters
        ----------
        smiles : str or sequence of str
            One or more query molecule SMILES strings.
        min_paffinity : float, optional
            Minimum affinity threshold (default: 6.0, pIC50 >= 6.0).
        quality_filter : str, optional
            Papyrus curation quality filter (``'High'``, ``'Medium'``, or None, default: ``'High'``).
        organism : str, optional
            Target source taxonomy filter (default: ``'Homo sapiens'``).
        min_target_consensus : int, optional
            Minimum number of query molecules that must bind a target for inclusion (default: 1).
        tolerance_factor : float, optional
            Tolerance window multiplier (<1.0 = stricter, >1.0 = looser, default: 1.0).
        max_molecules : int, optional
            Upper cap on candidate molecules returned (default: 1000).
        max_sascore : float, optional
            Optional upper-bound sanity cap on Synthetic Accessibility score (default: None).
        max_bertz : float, optional
            Optional upper-bound sanity cap on Bertz complexity (default: None).
        hard_filter : bool, optional
            Whether to strictly drop candidates outside tolerance windows (default: False, pure soft ranking).
        allow_chembl_fallback : bool, optional
            Whether to allow remote ChEMBL target query if not in Papyrus (default: False).
        auto_relax : bool, optional
            Whether to relax pAffinity if fewer than min_ligands_threshold candidates found (default: False).
        min_relax_paffinity : float, optional
            Floor limit for affinity relaxation (default: 5.0).
        min_ligands_threshold : int, optional
            Minimum candidates desired if auto_relax is enabled (default: 50).
        deduplicate : bool, optional
            Whether to deduplicate molecules by SMILES (default: True).
        show_progress : bool, optional
            Whether to display progress bars and profile summary table (default: True).

        Returns
        -------
        result : PipelineResult
            Consolidated dataclass containing `query_molecules`, `targets`, `property_profile`, and `papyrus_curated`.
        """
        # 1. Normalize inputs
        smiles_list: List[str] = [smiles] if isinstance(smiles, str) else list(smiles)
        if not smiles_list:
            raise ValueError("No SMILES provided to run pipeline.")

        if show_progress:
            self._print_progress(0.10, f"Step 1/4: Standardizing {len(smiles_list)} input molecule{'s' if len(smiles_list) > 1 else ''} with DrugEx...")

        normalized_molecules: List[NormalizedMolecule] = []
        for s in smiles_list:
            normalized_molecules.append(self.standardize_molecule(s))

        # 2. Derive MolecularPropertyProfile
        if show_progress:
            self._print_progress(0.25, "Step 2/4: Computing input property profile & adaptive weights...")

        profile = MolecularPropertyProfile.from_molecules(
            normalized_molecules,
            tolerance_factor=tolerance_factor
        )

        if show_progress:
            # Print derived profile summary table to terminal
            print("\n")
            profile.print_summary()

        current_paffinity = float(min_paffinity)

        while True:
            if show_progress:
                self._print_progress(0.45, f"Step 3/4: Deconvoluting targets in Papyrus (pAffinity >= {current_paffinity:.1f})...")

            # 3. Target Deconvolution in Papyrus
            target_counts: Dict[str, int] = {}
            target_map: Dict[str, TargetRecord] = {}

            for norm in normalized_molecules:
                t_list = self.find_targets(
                    inchikey=norm.inchikey,
                    connectivity=norm.connectivity,
                    min_paffinity=current_paffinity,
                    organism=organism,
                    allow_chembl_fallback=allow_chembl_fallback
                )
                for t in t_list:
                    acc = t["uniprot_acc"]
                    target_counts[acc] = target_counts.get(acc, 0) + 1
                    if acc not in target_map or t["pchembl_query"] > target_map[acc]["pchembl_query"]:
                        target_map[acc] = t

            # Consensus filter
            targets: List[TargetRecord] = [
                t for acc, t in target_map.items()
                if target_counts.get(acc, 0) >= min_target_consensus
            ]
            target_accessions = [t["uniprot_acc"] for t in targets]

            if show_progress:
                self._print_progress(0.70, f"Step 4/4: Found {len(targets)} targets. Extracting & ranking candidates from Papyrus...")

            # 4. Extract candidates from Papyrus
            raw_curated = self.get_papyrus_curated(
                target_accessions=target_accessions,
                min_paffinity=current_paffinity,
                quality_filter=quality_filter,
                deduplicate=deduplicate
            )

            # 5. Adaptive Candidate Selection (dynamic tolerance window + similarity ranking)
            selector = AdaptiveCandidateSelector(
                profile=profile,
                max_molecules=max_molecules,
                max_sascore=max_sascore,
                max_bertz=max_bertz,
                hard_filter=hard_filter
            )
            papyrus_curated = selector.select(raw_curated)

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
                    self._print_progress(0.45, msg)
                current_paffinity = next_paffinity
                continue

            break

        if show_progress:
            self._print_progress(
                1.00,
                f"Done! {len(targets)} targets | {len(papyrus_curated)} candidates ranked by property similarity"
            )

        return PipelineResult(
            query_molecules=normalized_molecules,
            targets=targets,
            property_profile=profile,
            papyrus_curated=papyrus_curated
        )
