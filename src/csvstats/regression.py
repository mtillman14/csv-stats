"""Simple linear regression: the ONE definition of a line of best fit.

A plot's fitted line, its R² and p-value annotation, and its confidence band
all come from :func:`linear_fit`, and so does the code a plot exports. So the
number printed on a figure and the number a table reports cannot disagree.

Light on purpose, like :mod:`csvstats.descriptive`: importing this module
loads numpy only. scipy is imported inside the two places that need the t
distribution (the p-value and the band), so a plotting package pays for it
only when a fit is actually drawn.

Conventions (each one is a decision, stated once):

- **Ordinary least squares** of ``y`` on ``x``: ``y = intercept + slope · x``.
- **Missing values**: a pair is used only when both ``x`` and ``y`` are
  finite. ``n`` counts the pairs used.
- **x is centred before fitting**, then the intercept is moved back. A
  datetime axis in milliseconds since 1970 (~1.7e12) would otherwise square
  to ~3e24 and lose every significant digit of the slope.
- **R²** is the squared Pearson correlation, ``1 - SS_res / SS_tot``.
- **p** is the two-sided t-test of ``slope = 0`` on ``n - 2`` degrees of
  freedom (what ``scipy.stats.linregress`` reports).
- **The band** is the t-based confidence interval of the MEAN response at
  each x (not a prediction interval), at :data:`CI_LEVEL`.
- **Too few values give NaN, never 0.** Below three pairs there are no
  residual degrees of freedom; with no spread in x there is no slope. The
  fit then carries a ``reason`` and every number is NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Default two-sided confidence level of the band.
CI_LEVEL = 0.95
#: Fewest pairs that leave a residual degree of freedom for p and the band.
MIN_PAIRS = 3

_NAN = float("nan")


@dataclass(frozen=True)
class LinearFit:
    """One simple linear regression of y on x."""

    n: int
    slope: float = _NAN
    intercept: float = _NAN
    r_squared: float = _NAN
    p_value: float = _NAN
    #: Standard error of the slope.
    slope_se: float = _NAN
    #: Residual standard error, ``sqrt(SS_res / (n - 2))``.
    residual_se: float = _NAN
    x_mean: float = _NAN
    #: Sum of squares of x about its mean (``Sxx``), kept for the band.
    sxx: float = _NAN
    x_min: float = _NAN
    x_max: float = _NAN
    #: Why the numbers are NaN, or None when the fit exists.
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None

    def predict(self, x: Any) -> np.ndarray:
        """The fitted line at ``x`` (NaN when the fit does not exist)."""
        xs = np.asarray(x, dtype=float)
        return self.intercept + self.slope * xs

    def band(self, x: Any, level: float = CI_LEVEL) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` of the CI of the mean response at each ``x``.

        ``ŷ ± t(1 - α/2, n - 2) · s · sqrt(1/n + (x - x̄)² / Sxx)``.
        """
        xs = np.asarray(x, dtype=float)
        centre = self.predict(xs)
        if not self.ok:
            nan = np.full_like(centre, np.nan)
            return nan, nan
        from scipy import stats  # lazy: see the module docstring

        t = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, self.n - 2))
        half = t * self.residual_se * np.sqrt(1.0 / self.n + (xs - self.x_mean) ** 2 / self.sxx)
        return centre - half, centre + half

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "slope_se": self.slope_se,
            "residual_se": self.residual_se,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "reason": self.reason,
        }


def _pairs(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"x and y differ in length: {xs.size} vs {ys.size}")
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def linear_fit(x: Any, y: Any) -> LinearFit:
    """Fit ``y = intercept + slope · x`` by ordinary least squares.

    Accepts anything numpy can turn into a 1-D float array (lists, arrays,
    pandas Series). Pairs with a missing x or y are dropped.
    """
    xs, ys = _pairs(x, y)
    n = int(xs.size)
    if n < MIN_PAIRS:
        return LinearFit(
            n=n,
            reason=f"needs at least {MIN_PAIRS} points with both x and y; has {n}",
        )
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    dx = xs - x_mean
    dy = ys - y_mean
    sxx = float(np.dot(dx, dx))
    if not sxx > 0.0:
        return LinearFit(n=n, x_mean=x_mean, x_min=float(xs.min()), x_max=float(xs.max()),
                         reason="every point has the same x, so there is no slope")
    sxy = float(np.dot(dx, dy))
    syy = float(np.dot(dy, dy))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    ss_res = max(syy - slope * sxy, 0.0)
    r_squared = 1.0 - ss_res / syy if syy > 0.0 else _NAN
    dof = n - 2
    residual_se = math.sqrt(ss_res / dof)
    slope_se = residual_se / math.sqrt(sxx)
    if slope_se > 0.0:
        from scipy import stats  # lazy: see the module docstring

        t_stat = slope / slope_se
        p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    else:
        # A perfect fit: the slope is known exactly. A zero slope through
        # constant y is no evidence of anything (p = 1); any other exact line
        # is p = 0, which is what linregress reports.
        p_value = 1.0 if slope == 0.0 else 0.0
    return LinearFit(
        n=n,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        slope_se=slope_se,
        residual_se=residual_se,
        x_mean=x_mean,
        sxx=sxx,
        x_min=float(xs.min()),
        x_max=float(xs.max()),
    )
