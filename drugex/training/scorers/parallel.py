"""Multiprocessing start-method selection for the ROCS scorers.

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

import multiprocessing as mp
import os
import signal
import threading
from contextlib import contextmanager


from typing import Iterator

class ScoreTimeout(Exception):
    """Raised when a single molecule exceeds the per-molecule scoring time limit."""


def pool_context(default: str = "fork") -> mp.context.BaseContext:
    """Return the multiprocessing context for scorer Pools, honouring DRUGEX_MP_CONTEXT.

    Parameters
    ----------
    default : str, optional
        Default start method if unspecified or unavailable, by default "fork".

    Returns
    -------
    mp.context.BaseContext
        Multiprocessing context instance.
    """
    method = os.environ.get("DRUGEX_MP_CONTEXT") or default
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context(default)


def capped_n_jobs(requested: int) -> int:
    """Cap the scorer worker count via ``DRUGEX_SCORE_NJOBS``.

    Parameters
    ----------
    requested : int
        Requested worker count.

    Returns
    -------
    int
        Capped worker count.
    """
    cap = os.environ.get("DRUGEX_SCORE_NJOBS")
    if not cap:
        return requested
    try:
        c = int(cap)
    except ValueError:
        return requested
    if c <= 0:
        return requested
    return min(requested, c)


def score_timeout() -> float | None:
    """Per-batch scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_TIMEOUT``.

    Returns
    -------
    float or None
        Timeout in seconds or None if unbounded.
    """
    v = os.environ.get("DRUGEX_SCORE_TIMEOUT")
    if not v:
        return None
    try:
        t = float(v)
    except ValueError:
        return None
    return t if t > 0 else None


def mol_timeout() -> float | None:
    """Per-molecule scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_MOL_TIMEOUT``.

    Returns
    -------
    float or None
        Per-molecule timeout in seconds or None if unbounded.
    """
    v = os.environ.get("DRUGEX_SCORE_MOL_TIMEOUT")
    if not v:
        return None
    try:
        t = float(v)
    except ValueError:
        return None
    return t if t > 0 else None


@contextmanager
def molecule_time_limit(seconds: float | None) -> Iterator[None]:
    """Best-effort per-molecule wall-clock cap via ``SIGALRM``.

    Raises ``ScoreTimeout`` when exceeded. No-op when seconds is falsy or not on main thread.

    Parameters
    ----------
    seconds : float or None
        Allowed computation time in seconds.

    Yields
    ------
    None
    """
    there and in the serial path), or on a platform without ``SIGALRM``.

    This bounds Python-interruptible hangs — including CDPKit's Python tautomer callback and
    the per-conformer alignment loop — so a single pathological molecule is scored 0 (invalid
    -> 0 reward) instead of wedging the whole epoch (cell 615). A pure-C single-call hang that
    never returns to Python is caught by the Pool-level ``score_timeout`` backstop instead.
    """
    if (not seconds
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")):
        yield
        return

    def _handler(signum, frame):
        raise ScoreTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
