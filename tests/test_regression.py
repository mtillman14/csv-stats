"""csvstats.regression: the one definition of a line of best fit."""

import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

from csvstats.regression import MIN_PAIRS, linear_fit


def test_matches_scipy_linregress():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 2.0 + 0.7 * x + rng.normal(scale=0.5, size=40)
    fit = linear_fit(x, y)
    ref = stats.linregress(x, y)
    assert fit.ok
    assert fit.n == 40
    assert fit.slope == pytest.approx(ref.slope)
    assert fit.intercept == pytest.approx(ref.intercept)
    assert fit.r_squared == pytest.approx(ref.rvalue**2)
    assert fit.p_value == pytest.approx(ref.pvalue)
    assert fit.slope_se == pytest.approx(ref.stderr)


def test_hand_computed_case():
    # x = 1..4, y = 2, 4, 5, 4: Sxx = 5, Sxy = 3.5, Syy = 4.75, so
    # slope 0.7, intercept 3.75 - 0.7 * 2.5 = 2.0, R² = 1 - (4.75 - 0.7 * 3.5) / 4.75.
    fit = linear_fit([1, 2, 3, 4], [2, 4, 5, 4])
    assert fit.slope == pytest.approx(0.7)
    assert fit.intercept == pytest.approx(2.0)
    assert fit.r_squared == pytest.approx(1 - 2.3 / 4.75)


def test_missing_pairs_are_dropped():
    fit = linear_fit([1, 2, np.nan, 3, 4, 5], [1, 2, 3, np.nan, 4, 5])
    assert fit.n == 4
    assert fit.slope == pytest.approx(1.0)


def test_too_few_points_give_nan_and_a_reason():
    fit = linear_fit([1, 2], [3, 4])
    assert not fit.ok
    assert fit.n == 2
    assert str(MIN_PAIRS) in fit.reason
    assert np.isnan(fit.slope) and np.isnan(fit.p_value) and np.isnan(fit.r_squared)


def test_no_spread_in_x_gives_nan_and_a_reason():
    fit = linear_fit([2, 2, 2, 2], [1, 2, 3, 4])
    assert not fit.ok
    assert "same x" in fit.reason
    assert np.isnan(fit.slope)


def test_epoch_milliseconds_keep_their_precision():
    """A datetime axis in ms since 1970: centring keeps the slope exact."""
    base = 1.7e12
    day = 86_400_000.0
    x = base + day * np.arange(10)
    y = 3.0 + 2.0 * np.arange(10)  # +2 per day
    fit = linear_fit(x, y)
    assert fit.slope * day == pytest.approx(2.0, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.predict(base) == pytest.approx(3.0, abs=1e-6)


def test_perfect_fit_has_p_zero_and_constant_y_has_p_one():
    assert linear_fit([1, 2, 3, 4], [2, 4, 6, 8]).p_value == 0.0
    flat = linear_fit([1, 2, 3, 4], [5, 5, 5, 5])
    assert flat.p_value == 1.0
    assert np.isnan(flat.r_squared)


def test_band_is_the_t_interval_of_the_mean_response():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=25)
    y = 1.0 + 0.5 * x + rng.normal(size=25)
    fit = linear_fit(x, y)
    xs = np.array([0.0, fit.x_mean, 10.0])
    low, high = fit.band(xs)
    t = stats.t.ppf(0.975, fit.n - 2)
    half = t * fit.residual_se * np.sqrt(1 / fit.n + (xs - fit.x_mean) ** 2 / fit.sxx)
    np.testing.assert_allclose(high - fit.predict(xs), half)
    np.testing.assert_allclose(fit.predict(xs) - low, half)
    # Narrowest at the mean of x.
    assert (high - low)[1] < (high - low)[0]
    assert (high - low)[1] < (high - low)[2]


def test_band_of_a_missing_fit_is_nan():
    low, high = linear_fit([1, 2], [1, 2]).band([1.0, 2.0])
    assert np.isnan(low).all() and np.isnan(high).all()


def test_regression_imports_no_heavy_dependencies():
    """A plotting package imports this module; scipy stays lazy."""
    code = (
        "import sys, csvstats.regression\n"
        "heavy = {'pingouin', 'statsmodels', 'matplotlib', 'reportlab', 'seaborn',"
        " 'scipy'}\n"
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
