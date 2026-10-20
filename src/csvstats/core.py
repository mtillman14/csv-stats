"""Hypothesis tests that return the common core record (requirements C1–C9).

Every public test here:

- takes one keyword design signature (C4):
  ``fn(data, dv, *, between=(), within=(), subject=None, levels=None,
  alpha=0.05, ci_level=0.95, tails="two-sided", ...)``;
- is pure (C1): it never writes a file, never prints, and never mutates the
  caller's DataFrame; identical input gives identical output;
- returns full-precision JSON primitives (C2): NaN becomes ``None``;
- returns the common core (C3): ``test``, ``design``, ``n``, ``effects`` (a
  list, one entry per term), ``assumptions``, ``warnings``, ``engine``,
  ``details``;
- states the comparison direction (C7): every estimate and signed effect size
  is ``a - b``, with ``a``/``b`` named and ordered by ``levels`` (default:
  natural sort of the level names);
- reports dropped rows and subjects (C6) in ``n.excluded_detail``.

Each two-sample effect also carries ``effect_sizes``: every variant computed
for it, named exactly (``d_s``, ``g_s``, ``d_z``, ``d_av``, ``d_rm``,
``glass_delta``, ``rank_biserial``). The caller chooses which to show. A
variant whose CI has no implementation yet has ``"ci": None``.

Descriptive formulas come from :mod:`csvstats.descriptive`, the one owner.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import __version__
from .descriptive import SD_DDOF, ci_half_width, t_critical

log = logging.getLogger("csvstats")

SCHEMA_VERSION = 1
TAILS = ("two-sided", "greater", "less")
CORRECTIONS = ("none", "bonferroni", "holm", "fdr_bh", "fdr_by")
#: Below this many values in a cell, a ``small_cell_n`` warning is added.
SMALL_CELL_N = 3


class NotSupported(NotImplementedError):
    """The design is valid, but this test does not handle it yet."""


# ---------------------------------------------------------------------------
# JSON-safe numbers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """A float, or None for NaN / inf / missing (JSON has no NaN)."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pair(low: Any, high: Any) -> list[float | None] | None:
    lo, hi = _num(low), _num(high)
    return None if lo is None and hi is None else [lo, hi]


def _text(value: Any) -> str:
    return str(value)


# ---------------------------------------------------------------------------
# p-value correction (C9)
# ---------------------------------------------------------------------------


def correct_pvalues(pvalues: Sequence[float | None], method: str = "holm") -> list[float | None]:
    """Adjusted p-values, in input order. ``None`` entries stay ``None``.

    Methods: ``none``, ``bonferroni``, ``holm`` (step-down), ``fdr_bh``
    (Benjamini–Hochberg), ``fdr_by`` (Benjamini–Yekutieli). The family is the
    non-None entries.
    """
    if method not in CORRECTIONS:
        raise ValueError(f"Unknown correction {method!r}; choose one of {list(CORRECTIONS)}.")
    present = [(i, float(p)) for i, p in enumerate(pvalues) if _num(p) is not None]
    out: list[float | None] = [None] * len(pvalues)
    m = len(present)
    if m == 0:
        return out
    if method == "none":
        for i, p in present:
            out[i] = p
        return out
    if method == "bonferroni":
        for i, p in present:
            out[i] = min(1.0, p * m)
        return out
    order = sorted(present, key=lambda item: item[1])
    adjusted = [0.0] * m
    if method == "holm":
        running = 0.0
        for rank, (_, p) in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p))
            adjusted[rank] = running
    else:
        scale = sum(1.0 / k for k in range(1, m + 1)) if method == "fdr_by" else 1.0
        running = 1.0
        for rank in range(m - 1, -1, -1):
            p = order[rank][1]
            running = min(running, min(1.0, p * m * scale / (rank + 1)))
            adjusted[rank] = running
    for rank, (i, _) in enumerate(order):
        out[i] = adjusted[rank]
    return out


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def _natural_key(value: Any):
    parts = re.split(r"(\d+)", str(value))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _level_order(series: pd.Series, given: Iterable[Any] | None) -> list[Any]:
    present = list(pd.unique(series.dropna()))
    if given is None:
        return sorted(present, key=_natural_key)
    given = list(given)
    unknown = [lv for lv in given if lv not in present]
    if unknown:
        raise ValueError(f"levels {unknown!r} are not in column {series.name!r}")
    return given


def _engine_versions(extra: Iterable[str] = ()) -> dict[str, str]:
    import scipy

    versions = {"csvstats": __version__, "scipy": scipy.__version__}
    for name in extra:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - only named when used
            pass
    return versions


class _Prepared:
    """Copied, validated, missing-dropped data for one test."""

    def __init__(
        self,
        data: pd.DataFrame,
        dv: str,
        factors: list[str],
        subject: str | None,
        levels: dict[str, list[Any]] | None,
        extra: Sequence[str] = (),
    ) -> None:
        """``extra``: further columns a test reads (an LMM's covariates,
        random-effect groups and slopes), carried along; a row missing one
        is dropped like a row missing the dv."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")
        needed = [dv, *factors, *([subject] if subject else [])]
        # dict.fromkeys: a column named twice (a covariate that is also a
        # random slope) must be selected once, or frame[c] is a DataFrame
        needed += [c for c in dict.fromkeys(extra) if c not in needed]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            raise ValueError(f"column(s) {missing!r} are not in the data")
        if dv in factors or (subject and subject in factors) or (subject == dv):
            raise ValueError(f"dv {dv!r}, factors {factors!r} and subject {subject!r} must differ")
        frame = data[needed].copy()
        frame[dv] = pd.to_numeric(frame[dv], errors="coerce")
        self.excluded_detail: list[dict[str, Any]] = []
        self.warnings: list[dict[str, str]] = []
        bad = frame[dv].isna() | frame[factors].isna().any(axis=1)
        if subject:
            bad |= frame[subject].isna()
        carried = [c for c in dict.fromkeys(extra) if c not in (dv, subject, *factors)]
        if carried:
            bad |= frame[carried].isna().any(axis=1)
        if bad.any():
            self.excluded_detail.append(
                {"rows": int(bad.sum()), "reason": f"missing {dv} or a factor/subject value"
                                                  + (f" or one of {carried}" if carried else "")}
            )
            self.warnings.append(
                {"code": "rows_dropped_missing", "message": f"{int(bad.sum())} row(s) with a missing value dropped"}
            )
            frame = frame[~bad]
        self.frame = frame
        self.dv = dv
        self.factors = factors
        self.subject = subject
        self.levels = {f: _level_order(frame[f], (levels or {}).get(f)) for f in factors}
        if levels:
            for f in factors:
                frame = frame[frame[f].isin(self.levels[f])]
            self.frame = frame

    def cells(self) -> list[dict[str, Any]]:
        if not self.factors:
            return [{"cell": {}, "n": int(len(self.frame))}]
        out = []
        for key, part in self.frame.groupby(self.factors, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            out.append({"cell": {f: _text(v) for f, v in zip(self.factors, key)}, "n": int(len(part))})
            if len(part) < SMALL_CELL_N:
                self.warnings.append(
                    {"code": "small_cell_n", "message": f"cell {dict(zip(self.factors, map(str, key)))} has n={len(part)}"}
                )
        return out

    def complete_subjects(self, within: list[str]) -> None:
        """Keep subjects with exactly one value in every within cell (C6)."""
        subject = self.subject
        assert subject is not None
        cells = self.frame.groupby([subject, *within]).size()
        duplicated = cells[cells > 1]
        if len(duplicated):
            first = duplicated.index[0]
            raise ValueError(
                f"subject {first[0]!r} has {int(duplicated.iloc[0])} rows in cell "
                f"{dict(zip(within, first[1:]))}; a within-subject test needs one row "
                f"per subject per cell (average them first)"
            )
        n_cells = int(np.prod([len(self.levels[w]) for w in within]))
        per_subject = self.frame.groupby(subject).size()
        incomplete = per_subject[per_subject < n_cells].index.tolist()
        if incomplete:
            for s in incomplete:
                self.excluded_detail.append({"subject": _text(s), "reason": "missing a within-subject level"})
            self.warnings.append(
                {
                    "code": "subjects_dropped_incomplete",
                    "message": f"{len(incomplete)} subject(s) without every within level dropped: "
                    + ", ".join(map(str, incomplete[:10])),
                }
            )
            self.frame = self.frame[~self.frame[subject].isin(incomplete)]

    def n_record(self) -> dict[str, Any]:
        excluded = sum(d.get("rows", 1) for d in self.excluded_detail)
        return {
            "total": int(len(self.frame)),
            "subjects": int(self.frame[self.subject].nunique()) if self.subject else None,
            "per_cell": self.cells(),
            "excluded": int(excluded),
            "excluded_detail": self.excluded_detail,
        }


def _check_tails(tails: str) -> None:
    if tails not in TAILS:
        raise ValueError(f"tails must be one of {list(TAILS)}, got {tails!r}")


def _design(dv, between, within, subject) -> dict[str, Any]:
    return {
        "dv": dv,
        "between": list(between),
        "within": list(within),
        "subject": subject,
        "covariates": [],
    }


def _record(
    test: str,
    label: str,
    prep: _Prepared,
    *,
    between,
    within,
    alpha,
    ci_level,
    tails,
    effects,
    assumptions,
    details,
    extra_engines: Iterable[str] = (),
) -> dict[str, Any]:
    n = prep.n_record()
    return {
        "schema_version": SCHEMA_VERSION,
        "test": test,
        "test_label": label,
        "design": _design(prep.dv, between, within, prep.subject),
        "levels": {f: [_text(v) for v in prep.levels[f]] for f in prep.factors},
        "alpha": alpha,
        "ci_level": ci_level,
        "tails": tails,
        "n": n,
        "effects": effects,
        "assumptions": assumptions,
        "post_hoc": None,
        "warnings": prep.warnings,
        "engine": _engine_versions(extra_engines),
        "seed": None,
        "details": details,
    }


def _one_factor(between, within, what: str) -> tuple[str, bool]:
    between, within = list(between), list(within)
    if len(between) + len(within) != 1:
        raise NotSupported(f"{what} compares the levels of exactly one factor; got between={between}, within={within}")
    return (between or within)[0], bool(within)


def _two_levels(prep: _Prepared, factor: str, what: str) -> tuple[Any, Any]:
    levels = prep.levels[factor]
    if len(levels) != 2:
        raise ValueError(
            f"{what} needs exactly 2 levels of {factor!r}, found {len(levels)}: "
            f"{[str(v) for v in levels]} (pass levels= to choose two)"
        )
    return levels[0], levels[1]


def _shapiro(values: np.ndarray, scope: str, alpha: float) -> dict[str, Any]:
    if len(values) < 3:
        return {"name": "normality", "test": "shapiro_wilk", "scope": scope, "statistic": None, "p": None, "violated": None}
    w, p = stats.shapiro(values)
    return {"name": "normality", "test": "shapiro_wilk", "scope": scope, "statistic": _num(w), "p": _num(p), "violated": bool(p < alpha)}


def _levene(groups: list[np.ndarray], alpha: float) -> dict[str, Any]:
    if any(len(g) < 2 for g in groups):
        return {"name": "homogeneity_of_variance", "test": "levene", "scope": "groups", "statistic": None, "p": None, "violated": None}
    w, p = stats.levene(*groups, center="median")
    return {"name": "homogeneity_of_variance", "test": "levene", "scope": "groups", "statistic": _num(w), "p": _num(p), "violated": bool(p < alpha)}


def assumption_checks(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """The checks a caller uses to RECOMMEND a test, before running one.

    Normality (Shapiro–Wilk) of the residuals the parametric model would have:
    ``y − cell mean`` for between-only designs, and additionally centred on
    each subject's mean when there is a within factor (every observation is
    used; nobody is dropped). Homogeneity of variance (Levene, median-centred)
    across the between cells, when there are any.

    Returns ``{"assumptions": [...], "n": ...}``. Pre-testing assumptions to
    choose a test is debated; callers should show this as evidence for a
    recommendation the user can override, never as a silent switch.
    """
    between, within = list(between), list(within)
    factors = [*between, *within]
    prep = _Prepared(data, dv, factors, subject if within else None, levels)
    frame = prep.frame
    y = frame[dv].astype(float)
    resid = y - (frame.groupby(factors)[dv].transform("mean") if factors else y.mean())
    if within and subject:
        resid = resid - frame.groupby(subject)[dv].transform("mean") + y.mean()
    checks = [_shapiro(resid.to_numpy(float), "residuals", alpha)]
    if between:
        groups = [g[dv].to_numpy(float) for _, g in frame.groupby(between)]
        checks.append(_levene(groups, alpha))
    return {"assumptions": checks, "n": prep.n_record()}


def _flag_violations(assumptions: list[dict], warnings: list[dict]) -> None:
    for a in assumptions:
        if a.get("violated"):
            warnings.append({"code": f"assumption_violated:{a['name']}", "message": f"{a['test']} p={a['p']:.3g} ({a['scope']})"})


# ---------------------------------------------------------------------------
# Effect-size CIs via the noncentral t
# ---------------------------------------------------------------------------


def _nct_ci(t_obs: float, df: float, scale: float, ci_level: float) -> list[float | None] | None:
    """CI of ``ncp / scale`` given an observed t (Steiger & Fouladi 1997)."""
    if not (math.isfinite(t_obs) and df > 0 and scale > 0):
        return None
    upper_q = 1.0 - (1.0 - ci_level) / 2.0
    lower_q = 1.0 - upper_q

    def solve(target: float) -> float | None:
        f = lambda ncp: stats.nct.cdf(t_obs, df, ncp) - target  # noqa: E731
        width = max(10.0, abs(t_obs) * 4)
        lo, hi = t_obs - width, t_obs + width
        try:
            return float(optimize.brentq(f, lo, hi, xtol=1e-10))
        except (ValueError, RuntimeError):
            return None

    lo, hi = solve(upper_q), solve(lower_q)
    if lo is None or hi is None:
        return None
    return [lo / scale, hi / scale]


def _hedges_j(df: float) -> float:
    """Exact small-sample correction J(df) = Γ(df/2) / (√(df/2) Γ((df-1)/2))."""
    if df <= 1:
        return float("nan")
    return float(math.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / math.sqrt(df / 2))


def _effect_size(name: str, value: Any, ci: Any = None, **extra: Any) -> dict[str, Any]:
    out = {"name": name, "value": _num(value), "ci": ci}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# t-tests
# ---------------------------------------------------------------------------


def ttest(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    equal_var: bool = False,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    tails: str = "two-sided",
) -> dict[str, Any]:
    """Two-level comparison: Welch / Student independent t, or paired t.

    One factor with exactly two levels. ``between`` gives the independent
    test (Welch unless ``equal_var``); ``within`` + ``subject`` gives the
    paired test on within-subject differences ``a - b``.
    """
    _check_tails(tails)
    factor, paired = _one_factor(between, within, "ttest")
    if paired and not subject:
        raise ValueError("a within-subject t-test needs subject=")
    prep = _Prepared(data, dv, [factor], subject, levels)
    a, b = _two_levels(prep, factor, "ttest")
    prep.frame = prep.frame[prep.frame[factor].isin([a, b])]
    prep.levels[factor] = [a, b]
    alternative = tails
    if paired:
        return _paired_t(prep, factor, a, b, alpha, ci_level, tails, alternative)
    return _independent_t(prep, factor, a, b, equal_var, alpha, ci_level, tails, alternative)


def _independent_t(prep, factor, a, b, equal_var, alpha, ci_level, tails, alternative):
    dv = prep.dv
    xa = prep.frame.loc[prep.frame[factor] == a, dv].to_numpy(float)
    xb = prep.frame.loc[prep.frame[factor] == b, dv].to_numpy(float)
    na, nb = len(xa), len(xb)
    if na < 2 or nb < 2:
        raise ValueError(f"each level needs at least 2 values; {a}: {na}, {b}: {nb}")
    ma, mb = xa.mean(), xb.mean()
    va, vb = xa.var(ddof=SD_DDOF), xb.var(ddof=SD_DDOF)
    diff = ma - mb
    result = stats.ttest_ind(xa, xb, equal_var=equal_var, alternative=alternative)
    if equal_var:
        df = na + nb - 2
        sp2 = ((na - 1) * va + (nb - 1) * vb) / df
        se = math.sqrt(sp2 * (1 / na + 1 / nb))
        df_method, test, label = "pooled", "ttest_ind_student", "Student's independent-samples t-test"
    else:
        se = math.sqrt(va / na + vb / nb)
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        df_method, test, label = "welch", "ttest_ind_welch", "Welch's independent-samples t-test"
    # t_critical takes a count n and uses df = n - 1, so n = df + 1 (Welch df is fractional).
    half = float(t_critical(df + 1, ci_level)) * se
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d_s = diff / pooled_sd if pooled_sd > 0 else float("nan")
    scale = math.sqrt(na * nb / (na + nb))
    d_ci = _nct_ci(d_s * scale, na + nb - 2, scale, ci_level) if math.isfinite(d_s) else None
    j = _hedges_j(na + nb - 2)
    g_s = d_s * j
    g_ci = [d_ci[0] * j, d_ci[1] * j] if d_ci else None
    sd_b = math.sqrt(vb)
    sizes = {
        "d_s": _effect_size("d_s", d_s, d_ci, denominator="pooled SD", ci_method="noncentral t"),
        "g_s": _effect_size("g_s", g_s, g_ci, denominator="pooled SD", correction="exact J", ci_method="noncentral t"),
        "glass_delta": _effect_size("glass_delta", diff / sd_b if sd_b > 0 else None, None, denominator=f"SD of {b}"),
    }
    effect = {
        "term": factor,
        "a": _text(a),
        "b": _text(b),
        "statistic_name": "t",
        "statistic": _num(result.statistic),
        "df": [_num(df)],
        "df_method": df_method,
        "p": _num(result.pvalue),
        "sphericity_correction": None,
        "estimate": {
            "name": "mean_difference",
            "value": _num(diff),
            "ci": _pair(diff - half, diff + half),
            "se": _num(se),
            "method": df_method,
        },
        "effect_size": sizes["g_s"],
        "effect_sizes": sizes,
    }
    assumptions = [_shapiro(xa, str(a), alpha), _shapiro(xb, str(b), alpha), _levene([xa, xb], alpha)]
    _flag_violations(assumptions, prep.warnings)
    details = {"means": {_text(a): _num(ma), _text(b): _num(mb)}, "sds": {_text(a): _num(math.sqrt(va)), _text(b): _num(sd_b)}}
    log.debug("ttest %s %s: %s - %s = %.6g, t=%.6g, df=%.6g", test, factor, a, b, diff, result.statistic, df)
    return _record(test, label, prep, between=[factor], within=[], alpha=alpha, ci_level=ci_level,
                   tails=tails, effects=[effect], assumptions=assumptions, details=details)


def _paired_wide(prep: _Prepared, factor: str, a, b) -> pd.DataFrame:
    prep.complete_subjects([factor])
    wide = prep.frame.pivot(index=prep.subject, columns=factor, values=prep.dv)
    return wide[[a, b]]


def _paired_t(prep, factor, a, b, alpha, ci_level, tails, alternative):
    wide = _paired_wide(prep, factor, a, b)
    xa, xb = wide[a].to_numpy(float), wide[b].to_numpy(float)
    d = xa - xb
    n = len(d)
    if n < 2:
        raise ValueError(f"a paired t-test needs at least 2 complete subjects, found {n}")
    result = stats.ttest_rel(xa, xb, alternative=alternative)
    mean_d, sd_d = d.mean(), d.std(ddof=SD_DDOF)
    se = sd_d / math.sqrt(n)
    half = float(ci_half_width(sd_d, n, ci_level))
    sd_a, sd_b = xa.std(ddof=SD_DDOF), xb.std(ddof=SD_DDOF)
    r = float(np.corrcoef(xa, xb)[0, 1]) if sd_a > 0 and sd_b > 0 else float("nan")
    d_z = mean_d / sd_d if sd_d > 0 else float("nan")
    dz_ci = _nct_ci(d_z * math.sqrt(n), n - 1, math.sqrt(n), ci_level) if math.isfinite(d_z) else None
    d_av = mean_d / ((sd_a + sd_b) / 2) if sd_a + sd_b > 0 else float("nan")
    rm_den = math.sqrt(max(sd_a**2 + sd_b**2 - 2 * r * sd_a * sd_b, 0.0)) if math.isfinite(r) else float("nan")
    d_rm = mean_d / rm_den * math.sqrt(2 * (1 - r)) if rm_den and math.isfinite(rm_den) and rm_den > 0 else float("nan")
    sizes = {
        "d_z": _effect_size("d_z", d_z, dz_ci, denominator="SD of the differences", ci_method="noncentral t"),
        "d_av": _effect_size("d_av", d_av, None, denominator="mean of the two SDs"),
        "d_rm": _effect_size("d_rm", d_rm, None, denominator="SD corrected for the correlation", r=_num(r)),
    }
    effect = {
        "term": factor,
        "a": _text(a),
        "b": _text(b),
        "statistic_name": "t",
        "statistic": _num(result.statistic),
        "df": [float(n - 1)],
        "df_method": "paired",
        "p": _num(result.pvalue),
        "sphericity_correction": None,
        "estimate": {
            "name": "mean_difference",
            "value": _num(mean_d),
            "ci": _pair(mean_d - half, mean_d + half),
            "se": _num(se),
            "sd_of_differences": _num(sd_d),
            "method": "paired",
        },
        "effect_size": sizes["d_z"],
        "effect_sizes": sizes,
    }
    assumptions = [_shapiro(d, "differences", alpha)]
    _flag_violations(assumptions, prep.warnings)
    details = {"means": {_text(a): _num(xa.mean()), _text(b): _num(xb.mean())}, "correlation": _num(r), "subjects": int(n)}
    return _record("ttest_paired", "Paired-samples t-test", prep, between=[], within=[factor],
                   alpha=alpha, ci_level=ci_level, tails=tails, effects=[effect],
                   assumptions=assumptions, details=details)


# ---------------------------------------------------------------------------
# Rank tests
# ---------------------------------------------------------------------------


def _hl_ci(sorted_values: np.ndarray, k: int) -> list[float | None] | None:
    m = len(sorted_values)
    if k < 0 or k >= m - k:
        return None
    return [_num(sorted_values[k]), _num(sorted_values[m - k - 1])]


def rank_test(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    tails: str = "two-sided",
) -> dict[str, Any]:
    """Two-level rank test: Mann–Whitney U (between) or Wilcoxon signed-rank (within).

    The estimate is the Hodges–Lehmann shift ``a - b`` with its
    distribution-free CI (normal approximation to the order-statistic rank);
    the effect size is the rank-biserial correlation (positive when ``a``
    tends to be larger).
    """
    _check_tails(tails)
    factor, paired = _one_factor(between, within, "rank_test")
    if paired and not subject:
        raise ValueError("a within-subject rank test needs subject=")
    prep = _Prepared(data, dv, [factor], subject, levels)
    a, b = _two_levels(prep, factor, "rank_test")
    prep.frame = prep.frame[prep.frame[factor].isin([a, b])]
    prep.levels[factor] = [a, b]
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    if paired:
        wide = _paired_wide(prep, factor, a, b)
        d = wide[a].to_numpy(float) - wide[b].to_numpy(float)
        n = len(d)
        if n < 2:
            raise ValueError(f"a Wilcoxon test needs at least 2 complete subjects, found {n}")
        nonzero = d[d != 0]
        if len(nonzero) < len(d):
            prep.warnings.append({"code": "zero_differences_dropped", "message": f"{len(d) - len(nonzero)} zero difference(s) dropped (Wilcoxon)"})
        ranks = stats.rankdata(np.abs(nonzero))
        if len(np.unique(np.abs(nonzero))) < len(nonzero):
            prep.warnings.append({"code": "ties_present", "message": "tied absolute differences; p uses a normal approximation"})
        t_plus, t_minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
        r_rb = (t_plus - t_minus) / (t_plus + t_minus) if t_plus + t_minus > 0 else float("nan")
        result = stats.wilcoxon(d, alternative=tails) if len(nonzero) else None
        walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
        m = len(walsh)
        k = int(math.floor(m / 2 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
        estimate = float(np.median(walsh))
        test, label, name, term_within = "wilcoxon", "Wilcoxon signed-rank test", "W", [factor]
        statistic = result.statistic if result is not None else None
        pvalue = result.pvalue if result is not None else None
        details = {"t_plus": _num(t_plus), "t_minus": _num(t_minus), "subjects": int(n)}
        between_out = []
    else:
        xa = prep.frame.loc[prep.frame[factor] == a, prep.dv].to_numpy(float)
        xb = prep.frame.loc[prep.frame[factor] == b, prep.dv].to_numpy(float)
        na, nb = len(xa), len(xb)
        if na < 1 or nb < 1:
            raise ValueError(f"each level needs at least 1 value; {a}: {na}, {b}: {nb}")
        combined = np.concatenate([xa, xb])
        if len(np.unique(combined)) < len(combined):
            prep.warnings.append({"code": "ties_present", "message": "tied values; p uses a tie-corrected approximation"})
        result = stats.mannwhitneyu(xa, xb, alternative=tails)
        u_a = float(result.statistic)
        r_rb = 2 * u_a / (na * nb) - 1
        diffs = np.sort((xa[:, None] - xb[None, :]).ravel())
        m = len(diffs)
        k = int(math.floor(m / 2 - z * math.sqrt(na * nb * (na + nb + 1) / 12)))
        estimate = float(np.median(diffs))
        test, label, name, term_within = "mann_whitney", "Mann–Whitney U test", "U", []
        statistic, pvalue = u_a, result.pvalue
        details = {"u_a": _num(u_a), "n_a": na, "n_b": nb}
        between_out = [factor]
        walsh = diffs
    sizes = {"rank_biserial": _effect_size("rank_biserial", r_rb, None)}
    effect = {
        "term": factor,
        "a": _text(a),
        "b": _text(b),
        "statistic_name": name,
        "statistic": _num(statistic),
        "df": [],
        "df_method": "n/a",
        "p": _num(pvalue),
        "sphericity_correction": None,
        "estimate": {
            "name": "hodges_lehmann",
            "value": _num(estimate),
            "ci": _hl_ci(walsh, k),
            "method": "distribution-free, normal approximation",
        },
        "effect_size": sizes["rank_biserial"],
        "effect_sizes": sizes,
    }
    return _record(test, label, prep, between=between_out, within=term_within, alpha=alpha,
                   ci_level=ci_level, tails=tails, effects=[effect], assumptions=[], details=details)


# ---------------------------------------------------------------------------
# ANOVA family
# ---------------------------------------------------------------------------


def _term(source: str) -> str:
    """pingouin's ``a * b`` term spelling as ``a:b``."""
    return ":".join(part.strip() for part in str(source).split("*"))


def _col(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def _p_unc(row: pd.Series) -> Any:
    """Uncorrected p: `p_unc` in pingouin 0.6, `p-unc` before."""
    return _col(row, "p_unc", "p-unc")


def anova(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Factorial ANOVA: between-only (any number of factors, type II SS),
    within-only (1–2 factors), or mixed (1 between + 1 within).

    Within terms with more than two levels are Greenhouse–Geisser corrected
    (``sphericity_correction`` and the corrected df say so). Effect sizes per
    term: ``eta2_p`` and ``eta2_g`` (Olejnik & Algina 2003, all factors
    treated as manipulated); ``omega2_p`` for between-only designs.
    """
    import pingouin as pg

    between, within = list(between), list(within)
    if not between and not within:
        raise ValueError("anova needs at least one factor")
    if within and not subject:
        raise ValueError("a within-subject ANOVA needs subject=")
    if within and len(within) > 2:
        raise NotSupported(f"within-subject ANOVA with {len(within)} within factors is not yet in csv-stats")
    if within and between and (len(between) > 1 or len(within) > 1):
        raise NotSupported(
            f"mixed ANOVA with between={between}, within={within} is not yet in csv-stats (1 between + 1 within only)"
        )
    prep = _Prepared(data, dv, [*between, *within], subject, levels)
    for factor in prep.factors:
        if len(prep.levels[factor]) < 2:
            raise ValueError(f"factor {factor!r} has fewer than 2 levels")
    if within:
        prep.complete_subjects(within)
    frame = prep.frame
    effects: list[dict[str, Any]] = []
    assumptions: list[dict[str, Any]] = []
    if not within:
        table = pg.anova(data=frame, dv=dv, between=between if len(between) > 1 else between[0], ss_type=2, detailed=True)
        n_total = len(frame)
        # pingouin names the residual row "Within" for one factor, "Residual" for several.
        is_resid = table["Source"].isin(["Residual", "Within"])
        df2 = float(table[is_resid].iloc[0]["DF"])
        for _, row in table[~is_resid].iterrows():
            f, df1 = float(row["F"]), float(row["DF"])
            np2 = (f * df1) / (f * df1 + df2)
            omega = (df1 * (f - 1)) / (df1 * (f - 1) + n_total) if math.isfinite(f) else float("nan")
            sizes = {
                "eta2_p": _effect_size("eta2_p", np2),
                "eta2_g": _effect_size("eta2_g", np2, None, note="equals eta2_p in a between-only design"),
                "omega2_p": _effect_size("omega2_p", omega),
            }
            effects.append({
                "term": _term(row["Source"]), "statistic_name": "F", "statistic": _num(f),
                "df": [df1, df2], "df_method": "residual", "p": _num(_p_unc(row)),
                "sphericity_correction": None, "estimate": None,
                "effect_size": sizes["eta2_p"], "effect_sizes": sizes,
            })
        residuals = frame[dv] - frame.groupby(between)[dv].transform("mean")
        assumptions.append(_shapiro(residuals.to_numpy(float), "residuals", alpha))
        groups = [g[dv].to_numpy(float) for _, g in frame.groupby(between)]
        assumptions.append(_levene(groups, alpha))
        test, label = "anova_between", "Between-subjects ANOVA (type II SS)"
    elif not between:
        table = pg.rm_anova(data=frame, dv=dv, within=within if len(within) > 1 else within[0], subject=subject, detailed=True, correction=True)
        rows = table[table["Source"] != "Error"] if "Error" in set(table["Source"]) else table
        error_df = None
        if "Error" in set(table["Source"]):
            error_df = float(table[table["Source"] == "Error"].iloc[0]["DF"])
        for _, row in rows.iterrows():
            f = float(row["F"])
            df1 = float(_col(row, "DF", "ddof1"))
            df2 = float(_col(row, "ddof2") if _col(row, "ddof2") is not None else error_df)
            eps = _num(_col(row, "eps"))
            effects.append(_within_effect(row, f, df1, df2, eps, prep, _term(row["Source"]), within))
        test, label = "anova_rm", "Repeated-measures ANOVA"
    else:
        table = pg.mixed_anova(data=frame, dv=dv, within=within[0], between=between[0], subject=subject, correction=True)
        ss_err = {}
        for _, row in table.iterrows():
            f, df1, df2 = float(row["F"]), float(row["DF1"]), float(row["DF2"])
            ss_err[row["Source"]] = float(row["SS"]) * df2 / (f * df1) if f > 0 else float("nan")
        err_between = ss_err.get(between[0], float("nan"))
        err_within = ss_err.get(within[0], float("nan"))
        for _, row in table.iterrows():
            f, df1, df2 = float(row["F"]), float(row["DF1"]), float(row["DF2"])
            ss = float(row["SS"])
            ng2 = ss / (ss + err_between + err_within)
            eps = _num(_col(row, "eps")) if row["Source"] != between[0] else None
            term = "Interaction" if row["Source"] == "Interaction" else row["Source"]
            name = f"{between[0]}:{within[0]}" if term == "Interaction" else term
            effects.append(_within_effect(row, f, df1, df2, eps, prep, name, within, eta2_g=ng2))
        test, label = "anova_mixed", "Mixed ANOVA (1 between, 1 within)"
    _flag_violations(assumptions, prep.warnings)
    return _record(test, label, prep, between=between, within=within, alpha=alpha, ci_level=ci_level,
                   tails="two-sided", effects=effects, assumptions=assumptions,
                   details={"ss_type": 2 if not within else None}, extra_engines=["pingouin"])


def _within_effect(row, f, df1, df2, eps, prep, term, within, eta2_g=None) -> dict[str, Any]:
    np2 = (f * df1) / (f * df1 + df2)
    ng2 = eta2_g if eta2_g is not None else _num(_col(row, "ng2"))
    p = _num(_p_unc(row))
    correction = None
    df = [df1, df2]
    # Sphericity only matters for a within factor with 3+ levels in the term.
    many_levels = any(len(prep.levels[name]) > 2 for name in term.split(":") if name in within)
    if eps is not None and many_levels:
        gg = _col(row, "p_GG_corr", "p-GG-corr")
        if gg is not None:
            p = _num(gg)
            df = [df1 * eps, df2 * eps]
            correction = {"method": "greenhouse_geisser", "epsilon": eps, "p_uncorrected": _num(_p_unc(row))}
    sizes = {"eta2_p": _effect_size("eta2_p", np2), "eta2_g": _effect_size("eta2_g", ng2)}
    return {
        "term": term, "statistic_name": "F", "statistic": _num(f), "df": [_num(v) for v in df],
        "df_method": "residual", "p": p, "sphericity_correction": correction, "estimate": None,
        "effect_size": sizes["eta2_g"], "effect_sizes": sizes,
    }


def kruskal(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Kruskal–Wallis H over one between factor; effect size epsilon²."""
    factor, paired = _one_factor(between, (), "kruskal")
    prep = _Prepared(data, dv, [factor], None, levels)
    groups = [prep.frame.loc[prep.frame[factor] == lv, dv].to_numpy(float) for lv in prep.levels[factor]]
    if len(groups) < 2:
        raise ValueError(f"kruskal needs at least 2 levels of {factor!r}")
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = h * (n + 1) / (n**2 - 1) if n > 1 else float("nan")
    sizes = {"epsilon2": _effect_size("epsilon2", eps2)}
    effect = {"term": factor, "statistic_name": "H", "statistic": _num(h), "df": [float(len(groups) - 1)],
              "df_method": "n/a", "p": _num(p), "sphericity_correction": None, "estimate": None,
              "effect_size": sizes["epsilon2"], "effect_sizes": sizes}
    return _record("kruskal", "Kruskal–Wallis test", prep, between=[factor], within=[], alpha=alpha,
                   ci_level=ci_level, tails="two-sided", effects=[effect], assumptions=[], details={})


def friedman(
    data: pd.DataFrame,
    dv: str,
    *,
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Friedman chi² over one within factor; effect size Kendall's W."""
    factor, _ = _one_factor((), within, "friedman")
    if not subject:
        raise ValueError("friedman needs subject=")
    prep = _Prepared(data, dv, [factor], subject, levels)
    prep.complete_subjects([factor])
    wide = prep.frame.pivot(index=subject, columns=factor, values=dv)[prep.levels[factor]]
    n, k = wide.shape
    if k < 3:
        raise ValueError(f"friedman needs at least 3 levels of {factor!r} (use rank_test for 2)")
    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy(float) for c in wide.columns])
    w = chi2 / (n * (k - 1)) if n > 0 else float("nan")
    sizes = {"kendall_w": _effect_size("kendall_w", w)}
    effect = {"term": factor, "statistic_name": "chi2", "statistic": _num(chi2), "df": [float(k - 1)],
              "df_method": "n/a", "p": _num(p), "sphericity_correction": None, "estimate": None,
              "effect_size": sizes["kendall_w"], "effect_sizes": sizes}
    return _record("friedman", "Friedman test", prep, between=[], within=[factor], alpha=alpha,
                   ci_level=ci_level, tails="two-sided", effects=[effect], assumptions=[],
                   details={"subjects": int(n)})


# ---------------------------------------------------------------------------
# SPM1d: statistical parametric mapping on 1-D curves (wraps spm1d)
# ---------------------------------------------------------------------------


def _curve_matrix(values, label: str) -> np.ndarray:
    arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    lengths = sorted({len(a) for a in arrays})
    if len(lengths) != 1:
        raise ValueError(
            f"{label}: curves of unequal length {lengths}; SPM needs every curve time-normalised "
            f"to the same number of nodes"
        )
    return np.vstack(arrays)


def spm_ttest(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    tails: str = "two-sided",
) -> dict[str, Any]:
    """SPM{t} over 1-D curves: two-sample (``between``) or paired (``within``).

    Each value of ``dv`` is one curve (a sequence), every curve with the same
    number of nodes. Returns the core record: one effect whose ``clusters``
    are the supra-threshold regions (``start`` / ``end`` nodes, ``p``),
    ``threshold`` the critical t*, ``curve`` the SPM{t}, and
    ``mean_difference`` the mean a − b curve. ``p`` is the smallest cluster
    p-value, or None when no cluster crosses the threshold.

    Needs ``spm1d`` (``pip install spm1d``); without it, raises
    :class:`NotSupported`.
    """
    try:
        import spm1d
    except ImportError as exc:
        raise NotSupported("SPM needs the spm1d package (pip install spm1d)") from exc
    _check_tails(tails)
    factor, paired = _one_factor(between, within, "spm_ttest")
    if paired and not subject:
        raise ValueError("a within-subject SPM test needs subject=")
    if not isinstance(data, pd.DataFrame) or dv not in data.columns:
        raise ValueError(f"column {dv!r} is not in the data")
    frame = data.copy()
    missing = frame[dv].isna() if frame[dv].dtype != object else frame[dv].map(lambda v: v is None)
    frame = frame[~missing]
    prep_levels = _level_order(frame[factor], (levels or {}).get(factor))
    if len(prep_levels) != 2:
        raise ValueError(f"spm_ttest needs exactly 2 levels of {factor!r}, found {[str(v) for v in prep_levels]}")
    a, b = prep_levels
    excluded: list[dict[str, Any]] = []
    if paired:
        counts = frame.groupby(subject)[factor].nunique()
        complete = counts[counts == 2].index
        dropped = [str(s) for s in counts.index if s not in set(complete)]
        excluded += [{"subject": s, "reason": "missing a within-subject level"} for s in dropped]
        frame = frame[frame[subject].isin(complete)].sort_values(subject)
        ya = _curve_matrix(frame.loc[frame[factor] == a, dv], f"{factor}={a}")
        yb = _curve_matrix(frame.loc[frame[factor] == b, dv], f"{factor}={b}")
        if ya.shape[1] != yb.shape[1]:
            raise ValueError(f"{factor}: the two levels' curves have {ya.shape[1]} and {yb.shape[1]} nodes")
        spm = spm1d.stats.ttest_paired(ya, yb)
        test, label = "spm_ttest_paired", "SPM paired t-test (spm1d)"
    else:
        ya = _curve_matrix(frame.loc[frame[factor] == a, dv], f"{factor}={a}")
        yb = _curve_matrix(frame.loc[frame[factor] == b, dv], f"{factor}={b}")
        if ya.shape[1] != yb.shape[1]:
            raise ValueError(f"{factor}: the two levels' curves have {ya.shape[1]} and {yb.shape[1]} nodes")
        spm = spm1d.stats.ttest2(ya, yb)
        test, label = "spm_ttest2", "SPM two-sample t-test (spm1d)"
    inference = spm.inference(alpha=alpha, two_tailed=(tails == "two-sided"), interp=True)
    clusters = []
    for cluster in getattr(inference, "clusters", None) or []:
        endpoints = getattr(cluster, "endpoints", (None, None))
        clusters.append({
            "start": _num(endpoints[0]), "end": _num(endpoints[1]),
            "p": _num(getattr(cluster, "P", None)), "extent": _num(getattr(cluster, "extent", None)),
        })
    curve = np.asarray(getattr(inference, "z", spm.z), dtype=float)
    p_values = [c["p"] for c in clusters if c["p"] is not None]
    effect = {
        "term": factor, "a": _text(a), "b": _text(b),
        "statistic_name": "SPM{t}", "statistic": _num(np.nanmax(np.abs(curve))) if curve.size else None,
        "df": [_num(d) for d in np.atleast_1d(getattr(spm, "df", []))],
        "df_method": "spm1d", "p": min(p_values) if p_values else None, "sphericity_correction": None,
        "estimate": {"name": "mean_difference_curve", "value": None, "ci": None,
                     "curve": [_num(v) for v in (ya.mean(axis=0) - yb.mean(axis=0))]},
        "effect_size": None, "effect_sizes": {},
        "clusters": clusters, "threshold": _num(getattr(inference, "zstar", None)),
        "curve": [_num(v) for v in curve], "nodes": int(ya.shape[1]),
    }
    n = {"total": int(ya.shape[0] + yb.shape[0]), "subjects": int(ya.shape[0]) if paired else None,
         "per_cell": [{"cell": {factor: _text(a)}, "n": int(ya.shape[0])}, {"cell": {factor: _text(b)}, "n": int(yb.shape[0])}],
         "excluded": len(excluded), "excluded_detail": excluded}
    import spm1d as _spm1d

    return {
        "schema_version": SCHEMA_VERSION, "test": test, "test_label": label,
        "design": _design(dv, [] if paired else [factor], [factor] if paired else [], subject if paired else None),
        "levels": {factor: [_text(a), _text(b)]}, "alpha": alpha, "ci_level": ci_level, "tails": tails,
        "n": n, "effects": [effect], "assumptions": [], "post_hoc": None,
        "warnings": ([{"code": "subjects_dropped_incomplete", "message": f"{len(excluded)} subject(s) dropped"}]
                     if excluded else []),
        "engine": {**_engine_versions(), "spm1d": getattr(_spm1d, "__version__", "unknown")},
        "seed": None, "details": {"h0reject": bool(getattr(inference, "h0reject", False))},
    }
