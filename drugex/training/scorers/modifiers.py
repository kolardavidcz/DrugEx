"""Score modifiers for transforming raw objective values into normalized reward functions."""

from __future__ import annotations

from functools import partial
from typing import Sequence, TypeVar, Union

# [PACKAGE: numpy] WHY: Array operations, power/exp math, min/max clipping, and vector transformations. WHAT: Numerical computing library.
import numpy as np

from drugex.training.scorers.interfaces import ScoreModifier

ScoreType = TypeVar("ScoreType", float, np.ndarray)


class Chained(ScoreModifier):
    """Applies multiple score modifiers sequentially in pipeline order."""

    def __init__(self, modifiers: Sequence[ScoreModifier]) -> None:
        """Initialize the chained modifier."""
        self.modifiers = list(modifiers)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Execute the chained modifiers in sequence."""
        score: ScoreType = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score


class Linear(ScoreModifier):
    """Linear score modifier that scales the score by a constant slope factor."""

    def __init__(self, slope: float = 1.0) -> None:
        """Initialize the Linear modifier with a given slope."""
        self.slope = float(slope)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Multiply the input score by the slope."""
        return self.slope * x


class Squared(ScoreModifier):
    """Quadratic penalty modifier with maximum at target_value."""

    def __init__(self, target_value: float, coefficient: float = 1.0) -> None:
        """Initialize the Squared modifier with target and curvature coefficient."""
        self.target_value = float(target_value)
        self.coefficient = float(coefficient)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Compute quadratic penalty from target value."""
        return 1.0 - self.coefficient * np.square(self.target_value - x)


class AbsoluteScore(ScoreModifier):
    """L1 linear penalty modifier with maximum at target_value."""

    def __init__(self, target_value: float) -> None:
        """Initialize the AbsoluteScore modifier with a target value."""
        self.target_value = float(target_value)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Compute linear distance penalty from target value."""
        return 1.0 - np.abs(self.target_value - x)


class Gaussian(ScoreModifier):
    """Gaussian bell-curve score modifier."""

    def __init__(self, mu: float, sigma: float) -> None:
        """Initialize the Gaussian modifier with mean and standard deviation."""
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate Gaussian bell function on input score(s)."""
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussian(ScoreModifier):
    """One-sided half-Gaussian score modifier for directional optimization."""

    def __init__(self, mu: float, sigma: float, minimize: bool = False) -> None:
        """Initialize the MinMaxGaussian modifier."""
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.minimize = bool(minimize)
        self._full_gaussian = Gaussian(mu=self.mu, sigma=self.sigma)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate one-sided half-Gaussian score."""
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussian = partial(MinMaxGaussian, minimize=True)
MaxGaussian = partial(MinMaxGaussian, minimize=False)


class ClippedScore(ScoreModifier):
    """Piecewise-linear score modifier with hard clamping bounds."""

    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        """Initialize the ClippedScore modifier."""
        assert low_score < high_score, f"low_score ({low_score}) must be < high_score ({high_score})"

        self.upper_x = float(upper_x)
        self.lower_x = float(lower_x)
        self.high_score = float(high_score)
        self.low_score = float(low_score)

        self.slope = (self.high_score - self.low_score) / (self.upper_x - self.lower_x)
        self.intercept = self.high_score - self.slope * self.upper_x

    def __call__(self, x: ScoreType) -> ScoreType:
        """Map and clamp input score(s)."""
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScore(ScoreModifier):
    """Smooth sigmoid-based approximation of ClippedScore."""

    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        """Initialize the SmoothClippedScore modifier."""
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
        """Evaluate smooth sigmoid score."""
        return self.low_score + self.L / (1.0 + np.exp(-self.k * (x - self.middle_x)))


class ThresholdedLinear(ScoreModifier):
    """Linear modifier with upper saturation cap at threshold."""

    def __init__(self, threshold: float) -> None:
        """Initialize the ThresholdedLinear modifier."""
        self.threshold = float(threshold)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Calculate thresholded linear ratio."""
        return np.minimum(x, self.threshold) / self.threshold


class SmoothHump(ScoreModifier):
    """Smooth bump / plateau modifier bounded by two half-Gaussians."""

    def __init__(self, lower_x: float, upper_x: float, sigma: float) -> None:
        """Initialize the SmoothHump modifier."""
        self.sigma = float(sigma)
        self.lower_x = float(lower_x)
        self.upper_x = float(upper_x)
        self._maximize_gaussian = MinMaxGaussian(mu=lower_x, sigma=sigma, minimize=False)
        self._minimize_gaussian = MinMaxGaussian(mu=upper_x, sigma=sigma, minimize=True)

    def __call__(self, x: ScoreType) -> ScoreType:
        """Evaluate smooth hump function."""
        y_maximize = self._maximize_gaussian(x)
        y_minimize = self._minimize_gaussian(x)
        return np.minimum(y_maximize, y_minimize)
