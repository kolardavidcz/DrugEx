from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem



def get_Pareto_fronts(scores: np.ndarray) -> List[np.ndarray]:
    """
    Identify the Pareto fronts from a given set of scores.

    Parameters
    ----------
    scores : numpy.ndarray
        An (n_points, n_scores) array of scores.

    Returns
    -------
    list of numpy.ndarray
        A list containing the indices of points belonging to each Pareto front.
    """

    ...

