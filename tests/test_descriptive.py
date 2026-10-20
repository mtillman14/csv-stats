"""csvstats.descriptive: the one definition of each descriptive formula."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from csvstats.descriptive import (
    ci_half_width,
    ci_interval,
    sem,
    summarize,
    t_critical,
)
from csvstats.utils.summary_stats import calculate_summary_statistics

#: t(0.975, n - 1) from published tables.
T_975 = {2: 12.7062, 5: 2.7764, 30: 2.0452}


@pytest.mark.parametrize("n", sorted(T_975))
def test_t_critical_matches_published_tables(n):
    assert t_critical(n) == pytest.approx(T_975[n], abs=1e-4)


def test_t_critical_is_nan_below_two_values():
    assert np.isnan(t_critical(1))
    assert np.isnan(t_critical(0))


def test_array_and_series_shapes_are_kept():
    counts = pd.Series([2, 5, 1], index=["a", "b", "c"])
    out = t_critical(counts)
    assert list(out.index) == ["a", "b", "c"]
    assert out["a"] == pytest.approx(T_975[2], abs=1e-4)
    assert np.isnan(out["c"])
    widths = ci_half_width(np.array([1.0, 2.0]), np.array([5, 30]))
    assert widths == pytest.approx([T_975[5] / np.sqrt(5), 2 * T_975[30] / np.sqrt(30)], rel=1e-4)


@pytest.mark.parametrize("n", sorted(T_975))
def test_ci_is_t_based_not_z(n):
    values = np.arange(1.0, n + 1.0) ** 1.5
    sd = values.std(ddof=1)
    low, high = ci_interval(values)
    half = T_975[n] * sd / np.sqrt(n)
    assert (high - low) / 2 == pytest.approx(half, rel=1e-4)
    assert (high - low) / 2 != pytest.approx(1.96 * sd / np.sqrt(n), rel=1e-3)
    assert ci_half_width(sd, n) == pytest.approx(half, rel=1e-4)


def test_ci_on_the_median_keeps_the_mean_half_width():
    values = [1.0, 2.0, 3.0, 10.0]
    low, high = ci_interval(values, centre="median")
    assert (low + high) / 2 == pytest.approx(2.5)
    assert (high - low) / 2 == pytest.approx(float(ci_half_width(np.std(values, ddof=1), 4)))


def test_missing_values_are_skipped_and_counted_separately():
    s = summarize([1.0, 2.0, 3.0, None, np.nan])
    assert s["n"] == 3 and s["n_missing"] == 2
    assert s["mean"] == pytest.approx(2.0)
    assert s["sd"] == pytest.approx(1.0)
    assert s["sem"] == pytest.approx(1.0 / np.sqrt(3))
    assert s["ci_high"] - s["mean"] == pytest.approx(float(t_critical(3)) / np.sqrt(3))


def test_summarize_reference_values():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    s = summarize(values)
    assert s["mean"] == pytest.approx(5.0)
    assert s["sd"] == pytest.approx(np.std(values, ddof=1))
    assert s["median"] == pytest.approx(4.5)
    assert (s["q1"], s["q3"]) == pytest.approx((4.0, 5.5))
    assert s["iqr"] == pytest.approx(1.5)
    assert s["range"] == pytest.approx(7.0)
    assert s["mode"] == 4.0 and s["mode_count"] == 3
    assert s["cv"] == pytest.approx(s["sd"] / 5.0)
    # Bias-corrected G1 / G2 (scipy bias=False), as SPSS reports them.
    assert s["skewness"] == pytest.approx(0.8184875, abs=1e-6)
    assert s["kurtosis"] == pytest.approx(0.940625, abs=1e-6)


def test_too_few_values_give_nan_not_zero():
    one = summarize([3.0])
    assert one["n"] == 1 and one["mean"] == 3.0
    for key in ("sd", "sem", "ci_low", "ci_high", "variance", "skewness", "kurtosis"):
        assert np.isnan(one[key]), key
    assert np.isnan(sem(1.0, 1))
    empty = summarize([])
    assert empty["n"] == 0 and np.isnan(empty["mean"])


def test_summary_statistics_count_excludes_missing_values():
    """Regression: count was len(group), which counted NaN, so n and the CI
    disagreed with the mean and SD."""
    data = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 3, "v": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0]})
    out = calculate_summary_statistics(data, "g", "v")
    assert out["grouped"]["count"] == {"a": 3, "b": 3}
    assert out["grouped"]["n_missing"] == {"a": 1, "b": 0}
    assert out["grouped"]["ci_95_high"]["a"] == pytest.approx(summarize([1.0, 2.0, 3.0])["ci_high"])
    assert out["overall"]["count"] == 6
    assert out["overall"]["n_missing"] == 1


def test_descriptive_imports_no_heavy_dependencies():
    """A plotting package depends on this module; it must stay light.

    scipy included: it is imported lazily by the functions that use it, so a
    plot that draws no interval never pays its import (2 s on Windows).
    """
    code = (
        "import sys, csvstats.descriptive\n"
        "heavy = {'pingouin', 'statsmodels', 'matplotlib', 'reportlab', 'seaborn',"
        " 'scipy'}\n"
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
