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

**The fit** is csv-stats' own REML (``_RandomIntercept``), matched to lme4 on
the reference datasets in tests/reference (test_lmm_reference.py): the fixed
effects, their SEs (sigma^2 (X' V^-1 X)^-1, as lme4), the variance
components, the REML log-likelihood and the singular-fit flag.

**Degrees of freedom: Wald only, for now** (§3.5). Every effect says
``df_method: "wald"``, the statistic is ``chi2`` (omnibus) or ``z``
(pairwise), and no t or F is ever implied. Satterthwaite (lmerTest's default)
is being added on this fit (.claude/plan-lmm-satterthwaite.md).

Fitting problems (non-convergence, a singular fit) become ``warnings``, never
exceptions (§3.5).
"""

from __future__ import annotations

import re
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


#: lme4's isSingular tolerance: theta = sigma_subject / sigma_residual below
#: this is "on the boundary".
SINGULAR_TOL = 1e-4


class _RandomIntercept:
    """REML (or ML) for ``y = X beta + u[subject] + e``, done here rather than
    by statsmodels (stage 2 of .claude/plan-lmm-satterthwaite.md).

    Checked against lme4 on 75 reference datasets, statsmodels' MixedLM fell
    short in two ways:
    - Its ``cov_params()`` comes from the joint likelihood over fixed effects
      AND variance parameters. On unbalanced data the two interact, and its
      SEs differed from lme4's by 0.1-4%. lme4 (and lmerTest's Satterthwaite,
      built on it) uses sigma^2 (X' V^-1 X)^-1 at the estimated variances.
    - Its optimizer stopped short of the boundary: a positive subject variance
      with a WORSE likelihood where lme4 found exactly 0, so singular fits
      went unflagged.

    With one random intercept, the profiled REML criterion depends on ONE
    parameter, theta = sigma_subject / sigma_residual >= 0 (lme4's own
    parameterisation). So it is minimised directly: a grid, then bounded
    Brent, with theta = 0 always compared explicitly. Every quantity is a
    closed form over per-subject sums:
    V_i / sigma^2 = I + theta^2 J, whose inverse is I - c_i J with
    c_i = theta^2 / (1 + n_i theta^2).

    ``deviance(theta)`` is lme4's REML criterion exactly:
    sum log(1 + n_i theta^2) + log|X' W X| + (n - p)(1 + log(2 pi q / (n - p))),
    where q is the weighted residual sum of squares.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, codes: np.ndarray, reml: bool = True) -> None:
        self.reml = reml
        self.n, self.p = X.shape
        # Centre y for numerical stability (data around 1e4 make y'y huge and
        # q a difference of large numbers); the intercept is restored below.
        self.y_shift = float(np.mean(y))
        yc = y - self.y_shift
        g = int(codes.max()) + 1
        self.ng = np.bincount(codes, minlength=g).astype(float)
        self.S = np.zeros((g, self.p))
        np.add.at(self.S, codes, X)
        self.T = np.bincount(codes, weights=yc, minlength=g)
        self.XtX = X.T @ X
        self.Xty = X.T @ yc
        self.yty = float(yc @ yc)

    def parts(self, theta: float) -> tuple[np.ndarray, np.ndarray, float]:
        """``(A = X'WX, beta_centred, q = r'Wr)`` at ``theta``."""
        c = theta ** 2 / (1.0 + self.ng * theta ** 2)
        A = self.XtX - (self.S * c[:, None]).T @ self.S
        b = self.Xty - self.S.T @ (c * self.T)
        beta = np.linalg.solve(A, b)
        q = self.yty - float((c * self.T ** 2).sum()) - float(b @ beta)
        return A, beta, q

    def deviance(self, theta: float) -> float:
        A, _, q = self.parts(theta)
        dof = self.n - self.p if self.reml else self.n
        out = float(np.log1p(self.ng * theta ** 2).sum()) + dof * (1.0 + np.log(2.0 * np.pi * q / dof))
        if self.reml:
            out += float(np.linalg.slogdet(A)[1])
        return out

    def fit(self) -> "_Fit":
        from scipy.optimize import minimize_scalar

        grid = np.concatenate([[0.0], np.logspace(-4, 3, 57)])
        values = np.array([self.deviance(t) for t in grid])
        i = int(np.argmin(values))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)] if i < len(grid) - 1 else grid[-1] * 100
        res = minimize_scalar(self.deviance, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        theta, best = float(res.x), float(res.fun)
        if self.deviance(0.0) <= best:  # the boundary is a candidate in its own right
            theta, best = 0.0, self.deviance(0.0)
        A, beta, q = self.parts(theta)
        sigma2 = q / (self.n - self.p if self.reml else self.n)
        beta = beta.copy()
        beta[0] += self.y_shift  # column 0 is the intercept (all ones)
        return _Fit(theta=theta, beta=beta, sigma2=sigma2, var_subject=theta ** 2 * sigma2,
                    cov_beta=sigma2 * np.linalg.inv(A), loglik=-best / 2.0, converged=bool(res.success),
                    singular=theta < SINGULAR_TOL, model=self)

    # --- Satterthwaite (stage 3) -------------------------------------------
    # As lmerTest: the variance parameters are varpar = (theta, sigma); their
    # asymptotic covariance is 2 * inverse Hessian of the UNprofiled REML
    # deviance at the estimate; a contrast's df is 2 v^2 / (g' A g) with
    # v = l' C(varpar) l and g its gradient in varpar.

    def deviance_at(self, varpar: np.ndarray) -> float:
        """The REML (or ML) deviance at (theta, sigma), not profiled in sigma:
        sum log(1 + n_i theta^2) + log|A| + (n - p) log(2 pi sigma^2) + q / sigma^2.
        At sigma^2 = q / (n - p) it equals :meth:`deviance`."""
        theta, sigma = float(varpar[0]), float(varpar[1])
        A, _, q = self.parts(theta)
        dof = self.n - self.p if self.reml else self.n
        out = float(np.log1p(self.ng * theta ** 2).sum()) + dof * np.log(2.0 * np.pi * sigma ** 2) + q / sigma ** 2
        if self.reml:
            out += float(np.linalg.slogdet(A)[1])
        return out

    def vcov_at(self, varpar: np.ndarray) -> np.ndarray:
        """Cov(beta-hat) = sigma^2 (X' W(theta) X)^-1 at (theta, sigma)."""
        A, _, _ = self.parts(float(varpar[0]))
        return float(varpar[1]) ** 2 * np.linalg.inv(A)


def _steps(x: np.ndarray) -> np.ndarray:
    """Finite-difference steps for varpar = (theta, sigma).

    - theta is dimensionless and can sit exactly at 0. Its step has a floor
      (0.1 * 1e-3). At 0 a step to -h is fine: everything depends on theta^2,
      so -h is a mirror image.
    - sigma carries the data's units and is never 0, so its step is purely
      relative. A shared 0.1 floor made the step 14% of sigma for data around
      1e-3, and the df there were off by 1.5e-4 (reference run 2026-10-19).
    """
    h = 1e-3 * np.abs(np.asarray(x, dtype=float))
    h[0] = 1e-3 * max(abs(float(x[0])), 0.1)
    return h


def _gradient(f, x: np.ndarray) -> np.ndarray:
    """Central differences with one Richardson extrapolation (error O(h^4))."""
    h = _steps(x)
    out = np.zeros(len(x))
    for i in range(len(x)):
        def d(step: float) -> float:
            e = np.zeros(len(x)); e[i] = step
            return (f(x + e) - f(x - e)) / (2 * step)
        out[i] = (4 * d(h[i] / 2) - d(h[i])) / 3
    return out


def _hessian(f, x: np.ndarray) -> np.ndarray:
    """Central-difference Hessian with one Richardson extrapolation."""
    m = len(x)
    h = _steps(x)

    def at(scale: float) -> np.ndarray:
        H = np.zeros((m, m))
        f0 = f(x)
        for i in range(m):
            ei = np.zeros(m); ei[i] = h[i] * scale
            H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / (h[i] * scale) ** 2
            for j in range(i + 1, m):
                ej = np.zeros(m); ej[j] = h[j] * scale
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                    4 * h[i] * h[j] * scale ** 2)
        return H

    return (4 * at(0.5) - at(1.0)) / 3


def _varpar_cov(fit) -> np.ndarray | None:
    """Asymptotic covariance of (theta, sigma): 2 * inverse Hessian of the
    REML deviance (lmerTest's vcov_varpar). None when the Hessian is not
    positive definite (it can't give a df)."""
    if getattr(fit, "_varpar_cov_cache", None) is not None:
        return fit._varpar_cov_cache
    varpar = np.array([fit.theta, np.sqrt(fit.sigma2)])
    H = _hessian(fit.model.deviance_at, varpar)
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        log.info("lmm Satterthwaite: varpar Hessian not positive definite: %s", H.tolist())
        return None
    fit._varpar_cov_cache = 2.0 * np.linalg.inv(H)
    return fit._varpar_cov_cache


def satterthwaite_df(fit, l: np.ndarray) -> tuple[float | None, str | None]:
    """``(df, None)`` for the contrast ``l' beta``, or ``(None, why)``.

    df = 2 v^2 / (g' A g): v = l' C l, g = d v / d varpar, A = Cov(varpar).
    Checked against lmerTest on the reference datasets (test_lmm_reference.py).
    """
    A = _varpar_cov(fit)
    if A is None:
        return None, "the variance parameters' Hessian is not positive definite"
    varpar = np.array([fit.theta, np.sqrt(fit.sigma2)])
    v = float(l @ fit.cov_beta @ l)
    g = _gradient(lambda vp: float(l @ fit.model.vcov_at(vp) @ l), varpar)
    denom = float(g @ A @ g)
    if not (np.isfinite(v) and np.isfinite(denom)) or denom <= 0 or v <= 0:
        return None, f"non-finite or non-positive terms (v={v:.3g}, g'Ag={denom:.3g})"
    return 2.0 * v * v / denom, None


class _Fit:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _fit(formula: str, frame: pd.DataFrame, reml: bool, warnings: list[dict[str, str]]):
    """Design matrix (patsy, the coding in ``formula``) + the REML fit.

    Returns the fit with ``columns`` and ``design_info`` attached.
    """
    import patsy

    y, X = patsy.dmatrices(formula, frame, return_type="dataframe")
    if X.columns[0] != "Intercept":
        raise ValueError("internal: the design must start with the intercept")
    codes = pd.factorize(frame.loc[X.index, _G])[0]
    result = _RandomIntercept(X.to_numpy(float), y.to_numpy(float).ravel(), codes, reml=reml).fit()
    result.columns = list(X.columns)
    result.design_info = X.design_info
    log.debug("lmm fit: theta=%.6g sigma2=%.6g loglik=%.6f singular=%s",
              result.theta, result.sigma2, result.loglik, result.singular)
    if not result.converged:
        warnings.append({"code": "convergence_failure", "message": "the mixed model did not converge"})
    if result.singular:
        warnings.append({
            "code": "singular_fit",
            "message": "the subject variance is estimated at zero (singular fit, as lme4's isSingular): "
                       "the model reduces to ordinary regression",
        })
    return result


def _variance_details(result) -> dict[str, Any]:
    group_var, resid = float(result.var_subject), float(result.sigma2)
    total = group_var + resid
    return {
        "subject_intercept_variance": _num(group_var),
        "residual_variance": _num(resid),
        "icc": _num(group_var / total) if total > 0 else None,
        "theta": _num(result.theta),
    }


#: patsy's name for one Sum-coded column: ``C(f0__, Sum)[S.<level>]``.
_SUM_COLUMN = re.compile(r"^C\((f\d+__), Sum\)\[S\.(.*)\]$")


def _column_names(column: str, names: dict[str, str], levels: dict[str, list[Any]]) -> tuple[str, str]:
    """``(r_name, label)`` for one fixed-effect column.

    ``r_name`` is lme4's name under ``contr.sum``: the factor followed by the
    1-based index of the level the column codes (``group1:time2``), so a fit
    can be compared with R's coefficient for coefficient. ``label`` is for
    people: ``group[ctl] : time[t2]``.
    """
    if column == "Intercept":
        return "(Intercept)", "(Intercept)"
    back = {safe: f for f, safe in names.items()}
    r_parts, labels = [], []
    for part in column.split(":"):
        match = _SUM_COLUMN.match(part)
        if match is None:
            return column, column
        factor = back[match.group(1)]
        level = match.group(2)
        index = [str(v) for v in levels[factor]].index(level) + 1
        r_parts.append(f"{factor}{index}")
        labels.append(f"{factor}[{level}]")
    return ":".join(r_parts), " : ".join(labels)


def _coefficients(result, ci_level: float, names: dict[str, str], levels: dict[str, list[Any]],
                  warnings: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Every fixed-effect coefficient with a Satterthwaite t test (stage 3),
    or the Wald z test with ``df_method: "wald"`` and a
    ``satterthwaite_unavailable`` warning when the df can't be computed."""
    cov = result.cov_beta
    out = []
    unavailable = []
    for i, column in enumerate(result.columns):
        se = float(np.sqrt(cov[i, i]))
        est = float(result.beta[i])
        stat = est / se if se > 0 else float("nan")
        r_name, label = _column_names(str(column), names, levels)
        l = np.zeros(len(result.columns)); l[i] = 1.0
        df, why = satterthwaite_df(result, l)
        entry = {"name": r_name, "label": label, "estimate": _num(est), "se": _num(se),
                 "z": _num(stat), "p_wald": _num(2 * stats.norm.sf(abs(stat)))}
        if df is None:
            unavailable.append(f"{r_name}: {why}")
            crit = stats.norm.ppf(0.5 + ci_level / 2)
            entry.update(df=None, df_method="wald", t=None, p=_num(2 * stats.norm.sf(abs(stat))))
        else:
            crit = stats.t.ppf(0.5 + ci_level / 2, df)
            entry.update(df=_num(df), df_method="satterthwaite", t=_num(stat), p=_num(2 * stats.t.sf(abs(stat), df)))
        entry["ci"] = [_num(est - crit * se), _num(est + crit * se)]
        out.append(entry)
    if unavailable:
        warnings.append({"code": "satterthwaite_unavailable",
                         "message": "Wald z used for: " + "; ".join(unavailable)})
        log.info("lmm Satterthwaite unavailable: %s", unavailable)
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

    k = len(result.columns)
    cov = result.cov_beta
    beta = result.beta
    # Map each formula term to its fixed-effect columns via patsy's design info.
    design_info = result.design_info
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
        "reml": reml, "converged": bool(result.converged),
        "log_likelihood": _num(result.loglik), "coefficients": _coefficients(result, ci_level, names, prep.levels, prep.warnings),
        "coding": "sum-to-zero", **_variance_details(result),
    }
    return _record("lmm", "Linear mixed model (Wald tests)", prep, between=between, within=within,
                   alpha=alpha, ci_level=ci_level, tails="two-sided", effects=effects, assumptions=[],
                   details=details, extra_engines=["patsy"])


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
    est = float(result.beta[1])
    se = float(np.sqrt(result.cov_beta[1, 1]))
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
        "converged": bool(result.converged), "log_likelihood": _num(result.loglik),
        **_variance_details(result),
    }
    return _record("lmm_pairwise", "Linear mixed model contrast (Wald z)", prep, between=list(between),
                   within=list(within), alpha=alpha, ci_level=ci_level, tails=tails, effects=[effect],
                   assumptions=[], details=details, extra_engines=["patsy"])
