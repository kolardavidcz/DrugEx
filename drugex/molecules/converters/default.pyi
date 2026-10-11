from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.converters.interfaces import MolConverter
from drugex.molecules.mol import DrExMol


class SmilesToDrEx(MolConverter):
    """Converter parsing a SMILES string into a DrExMol instance."""

    def __call__(self, smiles: str) -> DrExMol:
        """Constructs and validates DrExMol from SMILES string, raising ConversionException if invalid."""

        ...


class RDKitToDrEx(MolConverter):
    """Converter wrapping an existing RDKit Mol object into a DrExMol instance."""

    def __call__(self, rd_mol: Chem.Mol) -> DrExMol:
        """Wraps RDKit Mol into DrExMol object, raising ConversionException if molecule is empty."""

        ...


class DrExToSMILES(MolConverter):
    """Converter extracting the canonical SMILES string from a DrExMol instance."""

    def __call__(self, drugex_mol: DrExMol) -> str:
        """Returns canonical SMILES string from DrExMol.smiles property."""

        ...


class Identity(MolConverter):
    """Identity converter passing input items through unchanged."""

    def __call__(self, item: Any) -> Any:
        """Returns item unchanged."""

        ...

