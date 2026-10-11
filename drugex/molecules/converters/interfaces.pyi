from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from abc import ABC
from abc import abstractmethod


class ConversionException(Exception):
    """Raised when a molecule conversion operation fails."""

    ...


class MolConverter(ABC):
    """Abstract callable interface for converting molecules between representations."""

    @abstractmethod
    def __call__(self, representation: Any) -> Any:
        """Abstract callable transforming input representation into converted output, returning None or raising ConversionException on failure."""

        ...

