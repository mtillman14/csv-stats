"""Descriptive statistics: the ONE definition of each formula.

Everything that reports a mean, SD, SEM, confidence interval, quartile or
shape statistic gets it from here. That includes csv-stats' own test results
(``utils.summary_stats``) and scistack's plots (error bars) and tables
(StatStudio descriptives). So an error bar on a figure and the number in a
table cannot disagree.

Light on purpose: this module imports numpy, pandas and scipy only (never
pingouin, statsmodels, matplotlib or reportlab), so a plotting package can
depend on it cheaply. ``tests/test_descriptive.py`` guards that.

Conventions (each one is a decision, stated once):

- **Missing values** (NaN) are skipped. ``n`` is the count of non-missing
  values; missing values are counted separately (``n_missing``), never
  folded into ``n``.
- **SD** is the sample SD, divisor ``n - 1`` (:data:`SD_DDOF`).
- **Quartiles** interpolate linearly (:data:`QUANTILE_METHOD`, numpy's and
  pandas' default).
- **The confidence interval of the mean** is t-based:
  ``mean ± t(1 - α/2, n - 1) · SD / √n``. It is not ``1.96 · SEM`` (a z
  interval, too narrow at small n) and not a bootstrap (it would change
  between runs).
- **Too few values give NaN, never 0:** below two values there is no SD, SEM
  or interval. How to *draw* "no interval" is the caller's choice.
- **Skewness and kurtosis** are the bias-corrected sample estimators (G1,
  and G2 as *excess* kurtosis), as SPSS, SAS and Excel report them.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

#: Divisor offset of the sample SD (and variance): ``n - SD_DDOF``.
SD_DDOF = 1
#: Quantile interpolation for quartiles, IQR and percentiles.
QUANTILE_METHOD = "linear"
#: Default two-sided confidence level.
CI_LEVEL = 0.95

Centre = Literal["mean", "median"]


def _like(result: np.ndarray, template: Any) -> Any:
    """Return ``result`` shaped like ``template`` (scalar, array or Series)."""
    if isinstance(template, pd.Series):
        return pd.Series(result, index=template.index)
    if np.ndim(template) == 0:
        return float(result)
    return result


def t_critical(n: Any, level: float = CI_LEVEL) -> Any:
    """Two-sided t critical value ``t(1 - α/2, n - 1)`` per count.

    NaN where ``n < 2``. Accepts a scalar, numpy array or pandas Series
    (index kept) and returns the same kind.
    """
    counts = np.asarray(n, dtype=float)
    upper = 1.0 - (1.0 - level) / 2.0
    with np.errstate(invalid="ignore"):
        t = np.where(counts >= 2, stats.t.ppf(upper, np.maximum(counts - 1, 1)), np.nan)
    return _like(t, n)


def sem(sd: Any, n: Any) -> Any:
    """Standard error of the mean, ``SD / √n``. NaN where ``n < 2``."""
    counts = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts >= 2, np.asarray(sd, dtype=float) / np.sqrt(counts), np.nan)
    return _like(out, sd if isinstance(sd, pd.Series) else n)


def ci_half_width(sd: Any, n: Any, level: float = CI_LEVEL) -> Any:
    """Half-width of the t-based CI of the mean: ``t · SD / √n``.

    Works on precomputed SD and n (scalars, arrays or Series), so grouped and
    per-position callers use it without a second formula. NaN where
    ``n < 2``.
    """
    counts = np.asarray(n, dtype=float)
    t = np.asarray(t_critical(counts, level), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = t * np.asarray(sd, dtype=float) / np.sqrt(np.where(counts > 0, counts, np.nan))
    return _like(out, sd if isinstance(sd, pd.Series) else n)


def _clean(values: Any) -> pd.Series:
    return pd.Series(values, dtype=float).dropna()


def ci_interval(
    values: Any, level: float = CI_LEVEL, centre: Centre = "mean"
) -> tuple[float, float]:
    """``(low, high)`` of the t-based CI of the mean, placed on ``centre``.

    ``centre="median"`` keeps the mean's half-width but sits it on the
    median (what a median bar with a CI error band draws). The signature fits
    seaborn's callable ``errorbar=``. ``(nan, nan)`` below two values.
    """
    clean = _clean(values)
    n = len(clean)
    middle = clean.median() if centre == "median" else clean.mean()
    if n < 2:
        return (float("nan"), float("nan"))
    half = float(ci_half_width(clean.std(ddof=SD_DDOF), n, level))
    return (float(middle - half), float(middle + half))


def summarize(values: Any, level: float = CI_LEVEL) -> dict[str, float]:
    """Every descriptive statistic of one sample, under the conventions above.

    Keys: ``n``, ``n_missing``, ``mean``, ``sd``, ``variance``, ``sem``,
    ``ci_low``, ``ci_high``, ``median``, ``q1``, ``q3``, ``iqr``, ``min``,
    ``max``, ``range``, ``cv``, ``skewness``, ``kurtosis``, ``mode``,
    ``mode_count``.

    ``mode`` is the most frequent value (the smallest one on a tie), and
    ``mode_count`` is how often it occurs. For continuous data where every
    value is unique, ``mode_count`` is 1, so read it before trusting ``mode``.
    ``cv`` (SD / mean) is NaN when the mean is 0. Skewness needs 3 values and
    kurtosis 4; with fewer they are NaN.
    """
    raw = pd.Series(values, dtype=float)
    clean = raw.dropna()
    n = int(len(clean))
    nan = float("nan")
    out: dict[str, float] = {"n": n, "n_missing": int(raw.isna().sum())}
    if n == 0:
        for key in (
            "mean", "sd", "variance", "sem", "ci_low", "ci_high", "median", "q1",
            "q3", "iqr", "min", "max", "range", "cv", "skewness", "kurtosis",
            "mode", "mode_count",
        ):
            out[key] = nan
        return out
    mean = float(clean.mean())
    sd = float(clean.std(ddof=SD_DDOF)) if n >= 2 else nan
    q1, median, q3 = (float(v) for v in clean.quantile([0.25, 0.5, 0.75], interpolation=QUANTILE_METHOD))
    counts = clean.value_counts()
    top = int(counts.max())
    low, high = ci_interval(clean, level) if n >= 2 else (nan, nan)
    out.update(
        mean=mean,
        sd=sd,
        variance=sd * sd if n >= 2 else nan,
        sem=float(sem(sd, n)),
        ci_low=low,
        ci_high=high,
        median=median,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        min=float(clean.min()),
        max=float(clean.max()),
        range=float(clean.max() - clean.min()),
        cv=sd / mean if n >= 2 and mean != 0 else nan,
        skewness=float(stats.skew(clean, bias=False)) if n >= 3 else nan,
        kurtosis=float(stats.kurtosis(clean, fisher=True, bias=False)) if n >= 4 else nan,
        mode=float(min(counts[counts == top].index)),
        mode_count=top,
    )
    return out
