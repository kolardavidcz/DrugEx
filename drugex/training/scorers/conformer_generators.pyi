from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.training.generators.interfaces import Generator
OE_AVAILABLE: bool = ...
CDPL_AVAILABLE: bool = ...
oemolprop: Any = ...

import gc
import os
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Callable
from typing import List
from drugex.training.scorers.interfaces import ConformerGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

_RDKIT_TAUTOMER_MAX: Any = ...
_RDKIT_TAUTOMER_ENUMERATOR: Any = ...
_MAX_TAUTOMERS: Any = ...
logger: logging.Logger

class OmegaConformerGenerator(ConformerGenerator):
    """
    Conformer Generator using Omega

    Attributes:
        max_conformers (int): max number of conformers to generate
        max_centers (int): maximum number of stereocenters to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than
            max_rotatable_bonds
        use_gpu (bool): whether to use GPU for conformer generation
        show_progress (bool): whether to show progress during conformer generation
    """

    def __init__(self, max_conformers: int = ..., max_centers: int = ..., max_heavy_atoms: int = ..., max_rotatable_bonds: int = ..., filter: oemolprop.OEFilter | int | str | None = ..., use_gpu: bool = ..., show_progress: bool = ..., protonate: bool = ..., canonicalize_tautomer: bool = ..., timeout: int = ...) -> None:
        """
        Initialize the conformer generator

        Args:
            max_conformers (int): max number of conformers to generate
            max_centers (int): maximum number of stereocenters to enumerate
            max_heavy_atoms (int): drop molecules with more heavy atoms than
                max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than
                max_rotatable_bonds
            filter (oemolprop.OEFilter | int | str | None): filter to apply
                Either a path to a OEFilter file or an OEFilter object, or
                a int representing a OEFilter_Type (can be given like:
                    oemolprop.OEFilterType_BlockBuster)
            use_gpu (bool): whether to use GPU for conformer generation
            show_progress (bool): whether to show progress during conformer generation
            timeout (int): per-molecule Omega search-time cap in seconds. Omega has
                NO default per-molecule cap, so a torsion-driving-heavy molecule can
                run unbounded and blow the GPU walltime (audit 2026-06-25). 20s bounds
                the pathological tail; applied to both the top-level search-time and
                the torsion-driving search-time via SetMaxSearchTime.
        """

        ...

    def _build_omega_options(self) -> Any:
        """Build the OEOmegaOptions with the proven parameters + the search-time cap."""

        ...

    def _omega_options_for_test(self) -> Any:
        """Return the configured OEOmegaOptions (introspection for tests)."""

        ...

    def _create_fresh_omega(self) -> Any:
        """Create a fresh Omega instance with proven parameters"""

        ...

    def _filter_mol(self, smi: Any, mol: Chem.Mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        ...

    def _get_isomers(self, mol: Chem.Mol) -> Any:
        """Generate isomers for a molecule using OMEGA"""

        ...

    def genConformers(self, smiles_list: List[str], out_dir: str) -> str:
        """
        Generate conformers using Openeye Omega

        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for.
            out_dir (str): Path to the output directory for the generated conformers.

        Returns:
            str: Path to the output conformers file (oeb.gz)
        """

        ...


class SchrodingerConformerGenerator(ConformerGenerator):
    """
    Conformer generator using Schrodinger's software.

    Attributes:
        max_conformers: Maximum number of conformers to generate.
        max_isomers: Maximum number of isomers to generate.
        max_heavy_atoms: Maximum number of heavy atoms allowed.
        max_rotatable_bonds: Maximum number of rotatable bonds allowed.
        reactions: List of reaction functions to apply.
            Should take an rdkit molecule as input and return a modified molecule.
    """

    def __init__(self, max_conformers: int = ..., max_isomers: int = ..., max_heavy_atoms: int = ..., max_rotatable_bonds: int = ..., reactions: list[Callable] | None = ...) -> None:
        ...

    def _filter_mol(self, mol: Chem.Mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        ...

    def apply_reactions(self, mol: Chem.Mol) -> Chem.Mol:
        """Apply reactions to hydrolyze beta-lactams and DBO scaffolds"""

        ...

    def genConformers(self, smiles_list: List[str], out_dir: str) -> str:
        """
        Generate conformers using Schrodinger confgenx

        Returns:
            str: Path to the output SDF file containing generated conformers.
        """

        ...


class RDKitConformerGenerator(ConformerGenerator):
    """
    Conformer Generator using RDKit ETKDGv3

    Attributes:
        max_conformers (int): max number of conformers to generate
        max_isomers (int): maximum number of stereoisomers to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than
            max_rotatable_bonds
        show_progress (bool): whether to show progress during conformer generation
    """

    def __init__(self, max_conformers: int = ..., max_isomers: int = ..., max_heavy_atoms: int = ..., max_rotatable_bonds: int = ..., num_threads: int = ..., timeout: int = ..., show_progress: bool = ..., protonate: bool = ..., canonicalize_tautomer: bool = ...) -> None:
        """
        Initialize the conformer generator

        Args:
            max_conformers (int): max number of conformers to generate
            max_isomers (int): maximum number of stereoisomers to enumerate
            max_heavy_atoms (int): drop molecules with more heavy atoms than
                max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than
                max_rotatable_bonds
            num_threads (int): Number of threads for ETKDG conformer generation.
                0 = use all available CPU cores (default).
            timeout (int): Per-isomer timeout in seconds for ETKDG embedding.
                0 = no timeout. Default 20s bounds the ~10% of molecules that fail
                to embed cleanly (these otherwise burn timeout x max_isomers seconds
                each and dominate an RL epoch; audit 2026-06-23). The fast ~90% embed
                in <1s and are unaffected.
            show_progress (bool): whether to show progress during conformer generation
        """

        ...

    def _create_fresh_etkdg(self) -> Any:
        """Create ETKDGv3 parameters for conformer generation."""

        ...

    def _filter_mol(self, smi: Any, mol: Chem.Mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        ...

    def _get_isomers(self, mol: Chem.Mol) -> Any:
        """Generate isomers for a molecule using RDKit"""

        ...

    def genConformers(self, smiles_list: List[str], out_dir: str) -> str:
        """
        Generate conformers using RDKit ETKDG

        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for.
            out_dir (str): Path to the output directory for the generated conformers.

        Returns:
            str: Path to the output conformers file (SDF)
        """

        ...

    def write_conformers(self, mols: List[Any], out_file: str) -> None:
        """
        Write conformers for RDKit molecules to an SDF file

        Args:
            mols: List of RDKit molecules
            out_file: Path to output SDF file
        """

        ...


class CDPKitConformerGenerator(ConformerGenerator):
    """
    Conformer Generator using CDPKit CDPL.ConfGen

    CDPKit conformer generation following the official CDPKit examples.
    Based on gen_confs.py from CDPKit GitHub repository.

    This implementation uses StereoisomerGenerator for explicit stereoisomer enumeration,
    matching the behavior of RDKit and OpenEye implementations.

    Attributes:
        max_conformers (int): max number of conformers to generate per stereoisomer
        max_isomers (int): maximum number of stereoisomers to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than max_rotatable_bonds
        timeout (int): timeout for conformer generation in seconds
        min_rmsd (float): minimum RMSD between conformers
        energy_window (float): energy window for conformer selection in kcal/mol
        show_progress (bool): whether to show progress during conformer generation
    """

    def __init__(self, max_conformers: int = ..., max_isomers: int = ..., max_heavy_atoms: int = ..., max_rotatable_bonds: int = ..., timeout: int = ..., min_rmsd: float = ..., energy_window: float = ..., show_progress: bool = ..., protonate: bool = ..., canonicalize_tautomer: bool = ...) -> None:
        """
        Initialize the CDPKit conformer generator

        Args:
            max_conformers (int): max number of conformers to generate per stereoisomer
            max_isomers (int): maximum number of stereoisomers to enumerate
            max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than max_rotatable_bonds
            timeout (int): timeout for conformer generation in seconds
            min_rmsd (float): minimum RMSD between conformers
            energy_window (float): energy window for conformer selection in kcal/mol
            show_progress (bool): whether to show progress during conformer generation
        """

        ...

    def _create_conf_generator(self) -> Any:
        """Create a CDPKit ConformerGenerator with optimal settings following CDPKit examples"""

        ...

    def _gen_conf_ensemble(self, mol: Chem.Mol, conf_gen: Any) -> Any:
        """
        Generate conformer ensemble following CDPKit gen_confs.py pattern

        Directly from CDPKit documentation:
        https://cdpkit.org/cdpl_python_cookbook/confgen/gen_ensemble.html

        Returns:
            tuple: (status, num_conformers)
        """

        ...

    def _smiles_to_cdpl_mol(self, smiles: str) -> Any:
        """
        Convert SMILES to CDPKit molecule

        Initializes basic molecular properties (hydrogen counts, hybridization,
        ring flags, aromaticity) required for filtering and downstream processing.

        Uses CDPKit's official calcBasicProperties pattern from the cookbook:
        https://cdpkit.org/cdpl_python_cookbook/descr/fingerprints/ecfp.html
        """

        ...

    def _filter_mol(self, smi: str, mol: Chem.Mol) -> bool:
        """
        Filter molecules based on heavy atoms and rotatable bonds

        Note: Molecule must have basic properties initialized via
        calcBasicProperties before calling this method (done in _smiles_to_cdpl_mol).

        Args:
            smi: SMILES string for error reporting
            mol: CDPKit molecule object with basic properties initialized

        Returns:
            True if molecule should be filtered out (rejected), False otherwise
        """

        ...

    def _get_isomers(self, mol: Chem.Mol) -> Any:
        """
        Generate stereoisomers for a molecule using CDPKit StereoisomerGenerator

        This explicitly enumerates stereoisomers up to the ``max_isomers`` limit,
        matching the behavior of RDKit and OpenEye implementations.

        See: https://cdpkit.org/cdpl_api_doc/python_api_doc/classCDPL_1_1Chem_1_1StereoisomerGenerator.html

        Args:
            mol: CDPKit molecule object

        Yields:
            CDPKit molecule objects (stereoisomers)
        """

        ...

    def _canonical_tautomer(self, mol: Chem.Mol) -> Any:
        """
        Return the dominant (highest-scoring) tautomer via NATIVE CDPKit.

        The enumeration is bounded: CDPKit's DefaultTautomerGenerator has no native
        cap, so we count accepted tautomers in the callback and ``return False`` to
        stop generation once ``_MAX_TAUTOMERS`` have been scored (verified: a False
        return halts enumeration). This bounds the worst case for tautomer-rich
        molecules (BLOCKER B6, audit 2026-06-25).
        """

        ...

    def _canonical_tautomer_count(self, mol: Chem.Mol) -> Any:
        """Count how many tautomers the (capped) enumeration accepts (test probe)."""

        ...

    def _protonate_physiological(self, mol: Chem.Mol) -> Any:
        """Return the pH-7.4 dominant protonation state via NATIVE CDPKit."""

        ...

    def genConformers(self, smiles_list: List[str], out_dir: str) -> str:
        """
        Generate conformers for a list of SMILES and save to SDF

        For each input SMILES:
        1. Enumerate up to max_isomers stereoisomers using StereoisomerGenerator
        2. For each stereoisomer, generate up to max_conformers conformations
        3. Total output: up to (max_isomers × max_conformers) structures per molecule

        This matches the behavior of RDKit and OpenEye implementations.

        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for
            out_dir (str): Output directory for the conformer SDF file

        Returns:
            str: Path to the generated SDF file with conformers
        """

        ...

    def write_conformers(self, mols: List[Any], out_file: str) -> None:
        """
        Write conformers for RDKit molecules to an SDF file

        Args:
            mols: List of RDKit molecules
            out_file: Path to output SDF file
        """

        ...

