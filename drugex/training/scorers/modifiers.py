"""Score modifiers for transforming raw objective values into normalized reward functions."""

from __future__ import annotations

from functools import partial
from typing import Sequence, TypeVar, Union

import numpy as np

from drugex.training.scorers.interfaces import ScoreModifier

ScoreType = TypeVar("ScoreType", float, np.ndarray)


class Chained(ScoreModifier):
    """Applies multiple score modifiers sequentially in pipeline order.

    For example:
        `score = modifier3(modifier2(modifier1(raw_score)))`

    Parameters
    ----------
    modifiers : Sequence[ScoreModifier]
        List or sequence of modifiers to be evaluated in FIFO order.
    """

    def __init__(self, modifiers: Sequence[ScoreModifier]) -> None:
        """Initialize the chained modifier.

        Parameters
        ----------
        modifiers : Sequence[ScoreModifier]
            A sequence of modifiers to be applied in order.
        """
        self.modifiers = list(modifiers)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Execute the chained modifiers in sequence.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw or intermediate score(s).

        Returns
        -------
        float | np.ndarray
            The sequentially transformed score(s).
        """
        score: ScoreType = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score


class Linear(ScoreModifier):
    """Linear score modifier that scales the score by a constant slope factor.

    `f(x) = slope * x`

    Parameters
    ----------
    slope : float, optional
        Multiplicative scaling factor, by default 1.0 (identity).
    """

    def __init__(self, slope: float = 1.0) -> None:
        """Initialize the Linear modifier with a given slope.

        Parameters
        ----------
        slope : float, optional
            The slope factor, by default 1.0.
        """
        self.slope = float(slope)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Multiply the input score by the slope.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Scaled score(s).
        """
        return self.slope * x


class Squared(ScoreModifier):
    """Quadratic penalty modifier with maximum at target_value.

    Decreases quadratically with increasing distance from the target:
    `f(x) = 1.0 - coefficient * (target_value - x)^2`

    Parameters
    ----------
    target_value : float
        Optimal point where the score achieves its maximum (1.0).
    coefficient : float, optional
        Steepness of quadratic curvature penalty, by default 1.0.
    """

    def __init__(self, target_value: float, coefficient: float = 1.0) -> None:
        """Initialize the Squared modifier with target and curvature coefficient.

        Parameters
        ----------
        target_value : float
            Optimal target value.
        coefficient : float, optional
            Quadratic penalty coefficient, by default 1.0.
        """
        self.target_value = float(target_value)
        self.coefficient = float(coefficient)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Compute quadratic penalty from target value.

        Parameters
        ----------
        x : float | np.ndarray
            The input score(s).

        Returns
        -------
        float | np.ndarray
            The penalized score(s).
        """
        return 1.0 - self.coefficient * np.square(self.target_value - x)


class AbsoluteScore(ScoreModifier):
    """L1 linear penalty modifier with maximum at target_value.

    Decreases linearly with increasing distance from target:
    `f(x) = 1.0 - |target_value - x|`

    Parameters
    ----------
    target_value : float
        Optimal point where the score achieves 1.0.
    """

    def __init__(self, target_value: float) -> None:
        """Initialize the AbsoluteScore modifier with a target value.

        Parameters
        ----------
        target_value : float
            Target value for peak score.
        """
        self.target_value = float(target_value)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Compute linear distance penalty from target value.

        Parameters
        ----------
        x : float | np.ndarray
            The input score(s).

        Returns
        -------
        float | np.ndarray
            The modified score(s).
        """
        return 1.0 - np.abs(self.target_value - x)


class Gaussian(ScoreModifier):
    """Gaussian bell-curve score modifier.

    `f(x) = exp(-0.5 * ((x - mu) / sigma)^2)`

    Parameters
    ----------
    mu : float
        Mean / center of the Gaussian peak where score reaches 1.0.
    sigma : float
        Standard deviation controlling width / tolerance of acceptable range.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        """Initialize the Gaussian modifier with mean and standard deviation.

        Parameters
        ----------
        mu : float
            The mean (peak location) of the Gaussian.
        sigma : float
            The standard deviation (width) of the Gaussian.
        """
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate Gaussian bell function on input score(s).

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Gaussian bell-curve score(s) in (0.0, 1.0].
        """
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussian(ScoreModifier):
    """One-sided half-Gaussian score modifier for directional optimization.

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

    def __init__(self, mu: float, sigma: float, minimize: bool = False) -> None:
        """Initialize the MinMaxGaussian modifier.

        Parameters
        ----------
        mu : float
            Threshold boundary.
        sigma : float
            Width / decay constant.
        minimize : bool, optional
            Whether to penalize larger values, by default False.
        """
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.minimize = bool(minimize)
        self._full_gaussian = Gaussian(mu=self.mu, sigma=self.sigma)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate one-sided half-Gaussian score.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            One-sided Gaussian score(s) bounded in [0.0, 1.0].
        """
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussian = partial(MinMaxGaussian, minimize=True)
MaxGaussian = partial(MinMaxGaussian, minimize=False)


class ClippedScore(ScoreModifier):
    r"""Piecewise-linear score modifier with hard clamping bounds.

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

    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        """Initialize the ClippedScore modifier.

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
        assert low_score < high_score, f"low_score ({low_score}) must be < high_score ({high_score})"

        self.upper_x = float(upper_x)
        self.lower_x = float(lower_x)
        self.high_score = float(high_score)
        self.low_score = float(low_score)

        self.slope = (self.high_score - self.low_score) / (self.upper_x - self.lower_x)
        self.intercept = self.high_score - self.slope * self.upper_x

    def __call__(self, x: ScoreType) -> ScoreType:
        """Map and clamp input score(s).

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Linearly interpolated and clipped score(s).
        """
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScore(ScoreModifier):
    """Smooth sigmoid-based approximation of ClippedScore.

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

    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        """Initialize the SmoothClippedScore modifier.

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
        assert low_score < high_score, f"low_score ({low_score}) must be < high_score ({high_score})"

        self.upper_x = float(upper_x)
        self.lower_x = float(lower_x)
        self.high_score = float(high_score)
        self.low_score = float(low_score)

        # Slope of a standard logistic function at the center is 0.25 -> rescale k accordingly
        self.k = 4.0 / (self.upper_x - self.lower_x)
        self.middle_x = (self.upper_x + self.lower_x) / 2.0
        self.L = self.high_score - self.low_score

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate smooth sigmoid score.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Smooth sigmoidal score(s).
        """
        return self.low_score + self.L / (1.0 + np.exp(-self.k * (x - self.middle_x)))


class ThresholdedLinear(ScoreModifier):
    """Linear modifier with upper saturation cap at threshold.

    `f(x) = min(x, threshold) / threshold`

    Parameters
    ----------
    threshold : float
        Saturation threshold where score reaches 1.0.
    """

    def __init__(self, threshold: float) -> None:
        """Initialize the ThresholdedLinear modifier.

        Parameters
        ----------
        threshold : float
            Threshold value for saturation.
        """
        self.threshold = float(threshold)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Calculate thresholded linear ratio.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Normalized linear score(s) capped at 1.0.
        """
        return np.minimum(x, self.threshold) / self.threshold


class SmoothHump(ScoreModifier):
    """Smooth bump / plateau modifier bounded by two half-Gaussians.

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
        """Initialize the SmoothHump modifier.

        Parameters
        ----------
        lower_x : float
            Left threshold of optimal plateau.
        upper_x : float
            Right threshold of optimal plateau.
        sigma : float
            Decay width for the flanking half-Gaussians.
        """
        self.sigma = float(sigma)
        self.lower_x = float(lower_x)
        self.upper_x = float(upper_x)
        self._maximize_gaussian = MinMaxGaussian(mu=lower_x, sigma=sigma, minimize=False)
        self._minimize_gaussian = MinMaxGaussian(mu=upper_x, sigma=sigma, minimize=True)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate smooth hump function.

        Parameters
        ----------
        x : float | np.ndarray
            The input raw score(s).

        Returns
        -------
        float | np.ndarray
            Plateau score in (0.0, 1.0].
        """
        y_maximize = self._maximize_gaussian(x)
        y_minimize = self._minimize_gaussian(x)
        return np.minimum(y_maximize, y_minimize)
