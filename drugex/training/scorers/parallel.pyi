"""
Multiprocessing start-method selection for the ROCS scorers.

The cdpkit and rdkit ROCS scorers parallelise per-molecule scoring with a ``multiprocessing``
``Pool``. The default ``fork`` start method is dangerous inside the RL loop: the Pool is
created *after* ``forward()``/``evolve()`` have spun up torch/OpenMP worker threads, and
forking many workers from a multi-threaded process can deadlock — a child inherits a mutex
(e.g. the allocator lock) locked by a parent thread that does not exist in the child, so a
worker wedges and ``pool.map`` blocks forever. This is probabilistic and worsens with worker
count; it hung cdpkit/supermol/eps0.3/pt_pt (cell 615) at n_jobs=62, epoch 8.

``forkserver`` forks workers from a clean intermediary process that never had those threads,
so it sidesteps the deadlock; unlike ``spawn`` it does not re-import ``__main__`` (the RL
training runs from a ``python << PYEOF`` heredoc, which has no importable main module).

``DRUGEX_MP_CONTEXT`` (``fork`` | ``forkserver`` | ``spawn``) selects the method. The default
is ``fork`` so external users see byte-for-byte unchanged behaviour; the CCR2 RL scripts set
``forkserver``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import multiprocessing as mp
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class ScoreTimeout(Exception):
    """Raised when a single molecule exceeds the per-molecule scoring time limit."""

    ...


def pool_context(default: str = ...) -> mp.context.BaseContext:
    """
    Return the multiprocessing context for scorer Pools, honouring DRUGEX_MP_CONTEXT.

    Parameters
    ----------
    default : str, optional
        Default start method if unspecified or unavailable, by default "fork".

    Returns
    -------
    mp.context.BaseContext
        Multiprocessing context instance.
    """

    ...


def capped_n_jobs(requested: int) -> int:
    """
    Cap the scorer worker count via ``DRUGEX_SCORE_NJOBS``.

    Parameters
    ----------
    requested : int
        Requested worker count.

    Returns
    -------
    int
        Capped worker count.
    """

    ...


def score_timeout() -> float | None:
    """
    Per-batch scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_TIMEOUT``.

    Returns
    -------
    float or None
        Timeout in seconds or None if unbounded.
    """

    ...


def mol_timeout() -> float | None:
    """
    Per-molecule scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_MOL_TIMEOUT``.

    Returns
    -------
    float or None
        Per-molecule timeout in seconds or None if unbounded.
    """

    ...


@contextmanager
def molecule_time_limit(seconds: float | None) -> Iterator[None]:
    """
    Best-effort per-molecule wall-clock cap via ``SIGALRM``.

    Raises ``ScoreTimeout`` when exceeded. No-op when seconds is falsy or not on main thread.

    Parameters
    ----------
    seconds : float or None
        Allowed computation time in seconds.

    Yields
    ------
    None
    """

    ...

