from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from drugex.parallel.interfaces import ListCollector


class ListExtend(ListCollector):
    """
    Concrete result collector for parallel processing workers. Extends an internal list buffer with iterable batch results received from concurrent worker processes.

    Notes
    -----
    Used as the default collector in Standardization and ParallelSupplierEvaluator when worker outputs are lists.
    """

    def __init__(self) -> None:
        ...

    def __call__(self, result: Sequence[Any]) -> None:
        """Extends the internal results list with the sequence of items in result."""

        ...

    def getList(self) -> List[Any]:
        """Returns the aggregated master results list."""

        ...


class ListAppend(ListExtend):
    """
    Parallel result collector appending individual worker return items into an internal master list.

    Notes
    -----
    Contrasts with ListExtend which expects iterable sequence results per worker callback.
    """

    def __call__(self, result: Any) -> None:
        """Appends single item to internal results list."""

        ...

