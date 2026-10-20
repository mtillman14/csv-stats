"""Linear mixed models (requirements §3.5): use every observation.

A repeated-measures ANOVA or a paired test needs every subject at every
within level, so ``core._Prepared.complete_subjects`` drops (and reports) a
subject missing any one. A linear mixed model with a random intercept per
subject uses every observation instead, under the usual missing-at-random
assumption. That is why it exists here: a longitudinal study whose later
sessions are still being recorded loses no one.

Two tests, both returning the common core record (C3):

- :func:`lmm`: one Wald test per fixed-effect TERM (main effects and
  interactions, sum-to-zero coding), for the omnibus.
- :func:`lmm_pairwise`: the difference between two levels of one factor, for
  the pairwise comparisons. Every subject with at least one of the two levels
  contributes; unpaired observations inform the estimate through the random
  intercept.

**Degrees of freedom: Wald only, for now** (§3.5). statsmodels ``MixedLM``
gives Wald z / chi² tests, not t / F with Satterthwaite df. Every effect says
``df_method: "wald"``, the statistic is ``chi2`` (omnibus) or ``z``
(pairwise), and no t or F is ever implied. Satterthwaite (lmerTest's default)
is a planned addition, to be validated against R.

Fitting problems (non-convergence, a singular fit) become ``warnings``, never
exceptions (§3.5).
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .core import _Prepared, _num, _record, _text, _check_tails, log

#: Safe column names inside the model frame, so a factor called "speed (m/s)"
#: never meets the formula parser.
_Y, _G = "y__", "g__"


def _model_frame(prep: _Prepared, factors: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The prepared rows with safe names; factors as ordered Categoricals so
    patsy uses ``prep.levels`` order."""
    names = {f: f"f{i}__" for i, f in enumerate(factors)}
    frame = pd.DataFrame({_Y: prep.frame[prep.dv].astype(float).to_numpy(),
                          _G: prep.frame[prep.subject].astype(str).to_numpy()})
    for f, safe in names.items():
        frame[safe] = pd.Categorical(prep.frame[f].to_numpy(), categories=prep.levels[f])
    return frame, names


def _check_identifiable(prep: _Prepared) -> None:
    per_subject = prep.frame.groupby(prep.subject).size()
    if len(per_subject) < 2:
        raise ValueError("a linear mixed model needs at least 2 subjects")
    if int(per_subject.max()) < 2:
        raise ValueError(
            "every subject has one row, so a random intercept per subject is not identifiable; "
            "use an independent-samples test"
        )


def _fit(formula: str, frame: pd.DataFrame, reml: bool, warnings: list[dict[str, str]]):
    import statsmodels.formula.api as smf

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        model = smf.mixedlm(formula, frame, groups=frame[_G])
        result = model.fit(reml=reml)
    for w in caught:
        text = str(w.message)
        code = "convergence_failure" if "converge" in text.lower() else "lmm_fit"
        warnings.append({"code": code, "message": text})
        log.info("lmm fit warning (%s): %s", code, text)
    if not getattr(result, "converged", True):
        warnings.append({"code": "convergence_failure", "message": "the mixed model did not converge"})
    group_var = float(np.asarray(result.cov_re)[0, 0]) if np.size(result.cov_re) else 0.0
    # Relative to the residual variance: an optimiser stops NEAR the boundary,
    # not at exactly zero, so an absolute threshold misses real singular fits.
    if group_var <= 1e-4 * max(float(result.scale), 1e-12):
        warnings.append({
            "code": "singular_fit",
            "message": "the subject variance is estimated at zero (singular fit): "
                       "the model reduces to ordinary regression",
        })
    return result


def _variance_details(result) -> dict[str, Any]:
    group_var = float(np.asarray(result.cov_re)[0, 0]) if np.size(result.cov_re) else 0.0
    resid = float(result.scale)
    total = group_var + resid
    return {
        "subject_intercept_variance": _num(group_var),
        "residual_variance": _num(resid),
        "icc": _num(group_var / total) if total > 0 else None,
    }


def _coefficients(result, ci_level: float) -> list[dict[str, Any]]:
    fe = result.fe_params
    k = len(fe)
    cov = np.asarray(result.cov_params())[:k, :k]
    z_crit = stats.norm.ppf(0.5 + ci_level / 2)
    out = []
    for i, name in enumerate(fe.index):
        se = float(np.sqrt(cov[i, i]))
        est = float(fe.iloc[i])
        z = est / se if se > 0 else float("nan")
        out.append({
            "name": str(name), "estimate": _num(est), "se": _num(se), "z": _num(z),
            "p": _num(2 * stats.norm.sf(abs(z))), "ci": [_num(est - z_crit * se), _num(est + z_crit * se)],
        })
    return out


def _incomplete_note(prep: _Prepared, within: list[str], warnings: list[dict[str, str]]) -> None:
    """Say, without dropping anyone, which subjects lack some within cells."""
    if not within:
        return
    n_cells = int(np.prod([len(prep.levels[w]) for w in within]))
    cells = prep.frame.groupby(prep.subject)[within].apply(lambda g: len(g.drop_duplicates()))
    incomplete = [str(s) for s, c in cells.items() if c < n_cells]
    if incomplete:
        warnings.append({
            "code": "incomplete_within_used",
            "message": f"{len(incomplete)} subject(s) lack some within cells; every observation "
                       f"they have is used (mixed model): {', '.join(incomplete)}",
        })


def lmm(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    reml: bool = True,
) -> dict[str, Any]:
    """Linear mixed model ``dv ~ factors (full factorial) + (1 | subject)``.

    One effect per fixed term: a Wald chi² test of all its coefficients
    (sum-to-zero coding, so a main effect is tested at the average of the
    other factors). Coefficients, variance components, ICC, log-likelihood
    and convergence are in ``details``. No subject is dropped for missing a
    within level; the ones that do are named in an ``incomplete_within_used``
    warning.
    """
    between, within = list(between), list(within)
    factors = [*between, *within]
    if not factors:
        raise ValueError("lmm needs at least one factor")
    if not subject:
        raise ValueError("a linear mixed model needs subject=")
    prep = _Prepared(data, dv, factors, subject, levels)
    for f in factors:
        if len(prep.levels[f]) < 2:
            raise ValueError(f"factor {f!r} has fewer than 2 levels")
    _check_identifiable(prep)
    _incomplete_note(prep, within, prep.warnings)
    frame, names = _model_frame(prep, factors)
    formula = f"{_Y} ~ " + " * ".join(f"C({names[f]}, Sum)" for f in factors)
    result = _fit(formula, frame, reml, prep.warnings)

    fe = result.fe_params
    k = len(fe)
    cov = np.asarray(result.cov_params())[:k, :k]
    beta = fe.to_numpy(float)
    # Map each formula term to its fixed-effect columns via patsy's design info.
    design_info = result.model.data.design_info
    back = {safe: f for f, safe in names.items()}
    effects = []
    for term, columns in design_info.term_name_slices.items():
        if term == "Intercept":
            continue
        idx = np.arange(k)[columns]
        L = np.zeros((len(idx), k))
        L[np.arange(len(idx)), idx] = 1.0
        lb = L @ beta
        middle = L @ cov @ L.T
        try:
            chi2 = float(lb @ np.linalg.solve(middle, lb))
        except np.linalg.LinAlgError:
            chi2 = float("nan")
        df = int(len(idx))
        label = ":".join(back[part.split(",")[0].replace("C(", "").strip()] for part in term.split(":"))
        effects.append({
            "term": label, "statistic_name": "chi2", "statistic": _num(chi2), "df": [float(df)],
            "df_method": "wald", "p": _num(stats.chi2.sf(chi2, df)) if np.isfinite(chi2) else None,
            "sphericity_correction": None, "estimate": None, "effect_size": None, "effect_sizes": {},
        })
    details = {
        "formula": f"{dv} ~ {' * '.join(factors)} + (1 | {subject})",
        "reml": reml, "converged": bool(getattr(result, "converged", True)),
        "log_likelihood": _num(result.llf), "coefficients": _coefficients(result, ci_level),
        "coding": "sum-to-zero", **_variance_details(result),
    }
    return _record("lmm", "Linear mixed model (Wald tests)", prep, between=between, within=within,
                   alpha=alpha, ci_level=ci_level, tails="two-sided", effects=effects, assumptions=[],
                   details=details, extra_engines=["statsmodels"])


def lmm_pairwise(
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
    reml: bool = True,
) -> dict[str, Any]:
    """The difference ``a − b`` between two levels of ONE factor from
    ``dv ~ factor + (1 | subject)``: a Wald z test, every observation used.

    The pairwise counterpart of :func:`lmm`. A subject with only one of the
    two levels is kept: it informs the estimate through the random
    intercept, where a paired t-test would have to drop it.
    """
    from .core import _one_factor, _two_levels

    _check_tails(tails)
    factor, _ = _one_factor(between, within, "lmm_pairwise")
    if not subject:
        raise ValueError("a linear mixed model needs subject=")
    prep = _Prepared(data, dv, [factor], subject, levels)
    a, b = _two_levels(prep, factor, "lmm_pairwise")
    _check_identifiable(prep)
    _incomplete_note(prep, list(within), prep.warnings)
    frame, names = _model_frame(prep, [factor])
    safe = names[factor]
    # Reference level b, so the one coefficient IS a − b (C7).
    frame[safe] = pd.Categorical(frame[safe].astype(object), categories=[b, a])
    result = _fit(f"{_Y} ~ C({safe})", frame, reml, prep.warnings)
    coef = result.fe_params
    k = len(coef)
    cov = np.asarray(result.cov_params())[:k, :k]
    est = float(coef.iloc[1])
    se = float(np.sqrt(cov[1, 1]))
    z = est / se if se > 0 else float("nan")
    if tails == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif tails == "greater":
        p = stats.norm.sf(z)
    else:
        p = stats.norm.cdf(z)
    z_crit = stats.norm.ppf(0.5 + ci_level / 2)
    effect = {
        "term": factor, "a": _text(a), "b": _text(b), "statistic_name": "z", "statistic": _num(z),
        "df": [], "df_method": "wald", "p": _num(p),
        "estimate": {"name": "mean_difference", "value": _num(est),
                     "ci": [_num(est - z_crit * se), _num(est + z_crit * se)],
                     "se": _num(se), "sd_of_differences": None, "method": "lmm"},
        "effect_size": None, "effect_sizes": {},
    }
    details = {
        "formula": f"{dv} ~ {factor} + (1 | {subject})", "reml": reml,
        "converged": bool(getattr(result, "converged", True)), "log_likelihood": _num(result.llf),
        **_variance_details(result),
    }
    return _record("lmm_pairwise", "Linear mixed model contrast (Wald z)", prep, between=list(between),
                   within=list(within), alpha=alpha, ci_level=ci_level, tails=tails, effects=[effect],
                   assumptions=[], details=details, extra_engines=["statsmodels"])
