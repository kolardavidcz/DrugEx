from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem



class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer: torch.optim.Optimizer, init_lr: float, d_model: int, n_warmup_steps: int = ...) -> None:
        ...

    def step(self) -> None:
        """Step with the inner optimizer"""

        ...

    def zero_grad(self) -> None:
        """Zero out the gradients with the inner optimizer"""

        ...

    def _get_lr_scale(self) -> Any:
        ...

    def _update_learning_rate(self) -> Any:
        """Learning rate scheduling per step"""

        ...

