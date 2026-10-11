"""
Score modifiers for transforming raw objective values into normalized reward functions.
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
ScoreType = Union[float, np.ndarray, Sequence[float]]

from functools import partial
from typing import Sequence
from typing import TypeVar
from typing import Union
from drugex.training.scorers.interfaces import ScoreModifier


class Chained(ScoreModifier):
    """
    Applies multiple score modifiers sequentially in pipeline order.

    For example:
        `score = modifier3(modifier2(modifier1(raw_score)))`

    Parameters
    ----------
    modifiers : Sequence[ScoreModifier]
        List or sequence of modifiers to be evaluated in FIFO order.
    """

    def __init__(self, modifiers: Sequence[ScoreModifier]) -> None:
        """
        Initialize the chained modifier.

        Parameters
        ----------
        modifiers : Sequence[ScoreModifier]
            A sequence of modifiers to be applied in order.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Execute the chained modifiers in sequence.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw or intermediate score(s).

        Returns
        -------
        float | np.ndarray
            The sequentially transformed score(s).
        """

        ...


class Linear(ScoreModifier):
    """
    Linear score modifier that scales the score by a constant slope factor.

    `f(x) = slope * x`

    Parameters
    ----------
    slope : float, optional
        Multiplicative scaling factor, by default 1.0 (identity).
    """

    def __init__(self, slope: float = ...) -> None:
        """
        Initialize the Linear modifier with a given slope.

        Parameters
        ----------
        slope : float, optional
            The slope factor, by default 1.0.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Multiply the input score by the slope.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Scaled score(s).
        """

        ...


class Squared(ScoreModifier):
    """
    Quadratic penalty modifier with maximum at target_value.

    Decreases quadratically with increasing distance from the target:
    `f(x) = 1.0 - coefficient * (target_value - x)^2`

    Parameters
    ----------
    target_value : float
        Optimal point where the score achieves its maximum (1.0).
    coefficient : float, optional
        Steepness of quadratic curvature penalty, by default 1.0.
    """

    def __init__(self, target_value: float, coefficient: float = ...) -> None:
        """
        Initialize the Squared modifier with target and curvature coefficient.

        Parameters
        ----------
        target_value : float
            Optimal target value.
        coefficient : float, optional
            Quadratic penalty coefficient, by default 1.0.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Compute quadratic penalty from target value.

        Parameters
        ----------
        x : float | np.ndarray
            The input score(s).

        Returns
        -------
        float | np.ndarray
            The penalized score(s).
        """

        ...


class AbsoluteScore(ScoreModifier):
    """
    L1 linear penalty modifier with maximum at target_value.

    Decreases linearly with increasing distance from target:
    `f(x) = 1.0 - |target_value - x|`

    Parameters
    ----------
    target_value : float
        Optimal point where the score achieves 1.0.
    """

    def __init__(self, target_value: float) -> None:
        """
        Initialize the AbsoluteScore modifier with a target value.

        Parameters
        ----------
        target_value : float
            Target value for peak score.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Compute linear distance penalty from target value.

        Parameters
        ----------
        x : float | np.ndarray
            The input score(s).

        Returns
        -------
        float | np.ndarray
            The modified score(s).
        """

        ...


class Gaussian(ScoreModifier):
    """
    Gaussian bell-curve score modifier.

    `f(x) = exp(-0.5 * ((x - mu) / sigma)^2)`

    Parameters
    ----------
    mu : float
        Mean / center of the Gaussian peak where score reaches 1.0.
    sigma : float
        Standard deviation controlling width / tolerance of acceptable range.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        """
        Initialize the Gaussian modifier with mean and standard deviation.

        Parameters
        ----------
        mu : float
            The mean (peak location) of the Gaussian.
        sigma : float
            The standard deviation (width) of the Gaussian.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Evaluate Gaussian bell function on input score(s).

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Gaussian bell-curve score(s) in (0.0, 1.0].
        """

        ...


class MinMaxGaussian(ScoreModifier):
    """
    One-sided half-Gaussian score modifier for directional optimization.

    - When `minimize=True`: score is 1.0 for all `x <= mu`, and decreases as a half-Gaussian
      for `x > mu`.
    - When `minimize=False`: score is 1.0 for all `x >= mu`, and decreases as a half-Gaussian
      for `x < mu`.

    Parameters
    ----------
    mu : float
        Threshold value beyond which the Gaussian decay begins.
    sigma : float
        Standard deviation controlling the steepness of the penalty tail.
    minimize : bool, optional
        True for minimization (penalize larger values), False for maximization (penalize smaller values),
        by default False.
    """

    def __init__(self, mu: float, sigma: float, minimize: bool = ...) -> None:
        """
        Initialize the MinMaxGaussian modifier.

        Parameters
        ----------
        mu : float
            Threshold boundary.
        sigma : float
            Width / decay constant.
        minimize : bool, optional
            Whether to penalize larger values, by default False.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Evaluate one-sided half-Gaussian score.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            One-sided Gaussian score(s) bounded in [0.0, 1.0].
        """

        ...


class ClippedScore(ScoreModifier):
    r"""
    Piecewise-linear score modifier with hard clamping bounds.

    Maps inputs linearly between `lower_x` and `upper_x`, and clips outside
    to [`low_score`, `high_score`]:

    ```
       upper_x < lower_x                 lower_x < upper_x
    __________                                   ____________
              \                                 /
               \                               /
                \__________          _________/
    ```

    Parameters
    ----------
    upper_x : float
        Input x-value where output reaches `high_score` (if upper_x > lower_x) or
        transition point.
    lower_x : float, optional
        Input x-value where output reaches `low_score`, by default 0.0.
    high_score : float, optional
        Maximum clipped score, by default 1.0.
    low_score : float, optional
        Minimum clipped score, by default 0.0.
    """

    def __init__(self, upper_x: float, lower_x: float = ..., high_score: float = ..., low_score: float = ...) -> None:
        """
        Initialize the ClippedScore modifier.

        Parameters
        ----------
        upper_x : float
            High threshold for linear ramp.
        lower_x : float, optional
            Low threshold for linear ramp, by default 0.0.
        high_score : float, optional
            Ceiling score, by default 1.0.
        low_score : float, optional
            Floor score, by default 0.0.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Map and clamp input score(s).

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Linearly interpolated and clipped score(s).
        """

        ...


class SmoothClippedScore(ScoreModifier):
    """
    Smooth sigmoid-based approximation of ClippedScore.

    Uses a logistic function matched to the slope of `ClippedScore` at the midpoint:
    `f(x) = low_score + (high_score - low_score) / (1 + exp(-k * (x - middle_x)))`

    Parameters
    ----------
    upper_x : float
        Value where score approaches `high_score`.
    lower_x : float, optional
        Value where score approaches `low_score`, by default 0.0.
    high_score : float, optional
        Upper asymptote, by default 1.0.
    low_score : float, optional
        Lower asymptote, by default 0.0.
    """

    def __init__(self, upper_x: float, lower_x: float = ..., high_score: float = ..., low_score: float = ...) -> None:
        """
        Initialize the SmoothClippedScore modifier.

        Parameters
        ----------
        upper_x : float
            Upper boundary value.
        lower_x : float, optional
            Lower boundary value, by default 0.0.
        high_score : float, optional
            Max asymptotic score, by default 1.0.
        low_score : float, optional
            Min asymptotic score, by default 0.0.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Evaluate smooth sigmoid score.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Smooth sigmoidal score(s).
        """

        ...


class ThresholdedLinear(ScoreModifier):
    """
    Linear modifier with upper saturation cap at threshold.

    `f(x) = min(x, threshold) / threshold`

    Parameters
    ----------
    threshold : float
        Saturation threshold where score reaches 1.0.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initialize the ThresholdedLinear modifier.

        Parameters
        ----------
        threshold : float
            Threshold value for saturation.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Calculate thresholded linear ratio.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Normalized linear score(s) capped at 1.0.
        """

        ...


class SmoothHump(ScoreModifier):
    """
    Smooth bump / plateau modifier bounded by two half-Gaussians.

    The score is 1.0 for `x` within `[lower_x, upper_x]` and smoothly decays
    towards zero via Gaussian tails for `x < lower_x` and `x > upper_x`.

    Parameters
    ----------
    lower_x : float
        Lower boundary of the optimal plateau (1.0).
    upper_x : float
        Upper boundary of the optimal plateau (1.0).
    sigma : float
        Gaussian standard deviation for the decay tails.
    """

    def __init__(self, lower_x: float, upper_x: float, sigma: float) -> None:
        """
        Initialize the SmoothHump modifier.

        Parameters
        ----------
        lower_x : float
            Left threshold of optimal plateau.
        upper_x : float
            Right threshold of optimal plateau.
        sigma : float
            Decay width for the flanking half-Gaussians.
        """

        ...

    def __call__(self, x: ScoreType) -> ScoreType:  # type: ignore[override]
        """
        Evaluate smooth hump function.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Plateau score in (0.0, 1.0].
        """

        ...

