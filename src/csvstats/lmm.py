"""Linear mixed models (requirements §3.5): use every observation.

A repeated-measures ANOVA or a paired test needs every subject at every
within level, so ``core._Prepared.complete_subjects`` drops (and reports) a
subject missing any one. A linear mixed model with a random intercept per
subject uses every observation instead, under the usual missing-at-random
assumption. That is why it exists here: a longitudinal study whose later
sessions are still being recorded loses no one.

Two tests, both returning the common core record (C3):

- :func:`lmm`: one test per fixed-effect TERM (main effects and
  interactions, sum-to-zero coding), for the omnibus.
- :func:`lmm_pairwise`: the difference between two levels of one factor, for
  the pairwise comparisons. Every subject with at least one of the two levels
  contributes; unpaired observations inform the estimate through the random
  intercept.

**The random effects** default to ``(1 | subject)``. ``random=`` takes any
lme4-style structure as a list of ``{"group", "terms", "correlated"}``:
- random slopes, correlated or uncorrelated (``||``)
- several terms per group
- crossed groupings, e.g. ``(1 | subject) + (1 | item)``
- nested and interaction groupings, e.g. ``(1 | site) + (1 | site:subject)``

``lmm`` also takes numeric ``covariates=``, crossed with the factors like
``group * x``.

**The fit** is csv-stats' own REML, ``csvstats.lmm_fit.MixedModel``
(penalized least squares, Bates et al. 2015). It is matched to lme4 on two
fixture sets:
- the curated reference datasets (tests/test_lmm_reference.py)
- the sampled scenario set, 9 random-effect structures
  (tests/test_lmm_scenarios.py)

What is checked against lme4:
- the criterion at fixed theta
- the fixed effects and their SEs (sigma^2 (X' V^-1 X)^-1, as lme4)
- the variance components and theta
- the REML log-likelihood and the singular-fit flag

**Degrees of freedom: Satterthwaite** (lmerTest's default), over
varpar = (theta..., sigma):
- ``lmm``'s coefficients get t(df).
- ``lmm``'s terms get F(q, ddf), Fai–Cornelius, on type-III hypotheses
  (Goodnight 1980), with or without covariates. A factor crossed with a
  covariate is tested at covariate = 0 (``type3_at_covariate_zero``).
- ``lmm_pairwise``: every pairwise contrast from the FULL model (emmeans-style),
  t(df).

Every effect states its ``df_method``. If a df cannot be computed, the Wald
test is the result, labelled, with a ``satterthwaite_unavailable`` warning
(.claude/plan-lmm-satterthwaite.md, .claude/plan-lmm-general-fit.md).

Fitting problems (non-convergence, a singular fit) become ``warnings``, never
exceptions (§3.5).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .core import _Prepared, _num, _record, _text, _check_tails, log
from .lmm_fit import (
    INTERCEPT,
    SINGULAR_TOL,  # noqa: F401  (re-exported: lme4's isSingular tolerance)
    MixedModel,
    RandomTerm,
    build_blocks,
    parse_random,
    random_columns,
    random_formula,
)

#: Safe column names inside the model frame, so a factor called "speed (m/s)"
#: never meets the formula parser.
_Y, _G = "y__", "g__"


def _model_frame(prep: _Prepared, factors: list[str],
                 covariates: Sequence[str] = ()) -> tuple[pd.DataFrame, dict[str, str]]:
    """The prepared rows with safe names; factors as ordered Categoricals so
    patsy uses ``prep.levels`` order; covariates as floats. ``names`` maps
    every factor (``f<i>__``) and covariate (``c<i>__``) to its safe name."""
    names = {f: f"f{i}__" for i, f in enumerate(factors)}
    names.update({c: f"c{i}__" for i, c in enumerate(covariates)})
    frame = pd.DataFrame({_Y: prep.frame[prep.dv].astype(float).to_numpy(),
                          _G: prep.frame[prep.subject].astype(str).to_numpy()})
    for f in factors:
        frame[names[f]] = pd.Categorical(prep.frame[f].to_numpy(), categories=prep.levels[f])
    for c in covariates:
        frame[names[c]] = prep.frame[c].astype(float).to_numpy()
    return frame, names


def _fixed_formula(factors: list[str], covariates: Sequence[str], names: dict[str, str]) -> str:
    """Full factorial of the factors (sum coding) and covariates; intercept
    only when there are neither."""
    pieces = [f"C({names[f]}, Sum)" for f in factors] + [names[c] for c in covariates]
    return f"{_Y} ~ " + (" * ".join(pieces) if pieces else "1")


def _check_identifiable(prep: _Prepared) -> None:
    per_subject = prep.frame.groupby(prep.subject).size()
    if len(per_subject) < 2:
        raise ValueError("a linear mixed model needs at least 2 subjects")
    if int(per_subject.max()) < 2:
        raise ValueError(
            "every subject has one row, so a random intercept per subject is not identifiable; "
            "use an independent-samples test"
        )


# --- Satterthwaite -------------------------------------------------------------
# As lmerTest: the variance parameters are varpar = (theta..., sigma); their
# asymptotic covariance is 2 * inverse Hessian of the UNprofiled REML
# deviance at the estimate; a contrast's df is 2 v^2 / (g' A g) with
# v = l' C(varpar) l and g its gradient in varpar.

def _varpar(fit) -> np.ndarray:
    return np.r_[np.atleast_1d(np.asarray(fit.theta, dtype=float)), np.sqrt(fit.sigma2)]


def _steps(x: np.ndarray) -> np.ndarray:
    """Finite-difference steps for varpar = (theta..., sigma).

    - theta is dimensionless and can sit exactly at 0. Its step has a floor
      (0.1 * 1e-3). At 0 a step to -h is fine: a diagonal theta enters only
      through Lambda Lambda', so -h is a mirror image.
    - sigma carries the data's units and is never 0, so its step is purely
      relative. A shared 0.1 floor made the step 14% of sigma for data around
      1e-3, and the df there were off by 1.5e-4 (reference run 2026-10-19).
    """
    x = np.asarray(x, dtype=float)
    h = 1e-3 * np.abs(x)
    h[:-1] = 1e-3 * np.maximum(np.abs(x[:-1]), 0.1)
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


#: Hessian eigenvalues at or below this fraction of the largest are treated
#: as zero (dropped from the pseudo-inverse).
_EIGEN_RTOL = 1e-10


def _varpar_cov(fit) -> np.ndarray | None:
    """Asymptotic covariance of (theta..., sigma): 2 * the inverse Hessian
    of the REML deviance (lmerTest's vcov_varpar).

    At a boundary the Hessian can be singular or slightly indefinite. As
    lmerTest documents, only its positive eigenvalues are inverted (a
    pseudo-inverse), and the others are logged. None only when no
    eigenvalue is positive (no df can be formed)."""
    if getattr(fit, "_varpar_cov_cache", None) is not None:
        return fit._varpar_cov_cache
    H = _hessian(fit.model.deviance_at, _varpar(fit))
    H = (H + H.T) / 2.0
    evals, V = np.linalg.eigh(H)
    pos = evals > _EIGEN_RTOL * max(float(np.abs(evals).max()), 0.0)
    if not pos.any():
        log.info("lmm Satterthwaite: varpar Hessian has no positive eigenvalue: %s", evals.tolist())
        return None
    if not pos.all():
        log.info("lmm Satterthwaite: varpar Hessian eigenvalues %s; %d non-positive dropped (pseudo-inverse)",
                 np.round(evals, 8).tolist(), int((~pos).sum()))
    fit._varpar_cov_cache = 2.0 * (V[:, pos] / evals[pos]) @ V[:, pos].T
    return fit._varpar_cov_cache


def satterthwaite_df(fit, l: np.ndarray) -> tuple[float | None, str | None]:
    """``(df, None)`` for the contrast ``l' beta``, or ``(None, why)``.

    df = 2 v^2 / (g' A g): v = l' C l, g = d v / d varpar, A = Cov(varpar).
    Checked against lmerTest on the reference datasets (test_lmm_reference.py)
    and the scenario set (test_lmm_scenarios.py).
    """
    A = _varpar_cov(fit)
    if A is None:
        return None, "the variance parameters' Hessian has no positive eigenvalue"
    v = float(l @ fit.cov_beta @ l)
    g = _gradient(lambda vp: float(l @ fit.model.vcov_at(vp) @ l), _varpar(fit))
    denom = float(g @ A @ g)
    if not (np.isfinite(v) and np.isfinite(denom)) or denom <= 0 or v <= 0:
        return None, f"non-finite or non-positive terms (v={v:.3g}, g'Ag={denom:.3g})"
    return 2.0 * v * v / denom, None


# --- the fit -----------------------------------------------------------------

def _is_subject_intercept(terms: Sequence[RandomTerm], subject: str) -> bool:
    return len(terms) == 1 and terms[0].group == (subject,) and terms[0].terms == (INTERCEPT,)


def _fit(formula: str, frame: pd.DataFrame, reml: bool, prep: _Prepared, terms: Sequence[RandomTerm]):
    """Design matrix (patsy, the coding in ``formula``), the random-effect
    blocks (from ``prep.frame``, row for row) and the REML fit.

    Returns the fit with ``columns``, ``X``, ``design_info`` and ``terms``
    attached. Warnings go to ``prep.warnings``.
    """
    import patsy

    warnings = prep.warnings
    y, X = patsy.dmatrices(formula, frame, return_type="dataframe")
    if X.columns[0] != "Intercept":
        raise ValueError("internal: the design must start with the intercept")
    Xn = X.to_numpy(float)
    rank = int(np.linalg.matrix_rank(Xn))
    if rank < Xn.shape[1]:
        raise ValueError(
            f"the fixed-effect design is rank deficient ({rank} of {Xn.shape[1]} columns): a combination of "
            f"factor levels has no rows, or a covariate is collinear with the other columns")
    rows = prep.frame.iloc[X.index.to_numpy()].reset_index(drop=True)
    blocks = build_blocks(rows, terms, prep.levels)
    model = MixedModel(Xn, y.to_numpy(float).ravel(), blocks, reml=reml)
    result = model.fit()
    result.columns = list(X.columns)
    result.X = Xn
    result.design_info = X.design_info
    result.terms = list(terms)
    log.debug("lmm fit: %s; theta=%s sigma2=%.6g loglik=%.6f singular=%s", random_formula(terms),
              np.round(result.theta, 6).tolist(), result.sigma2, result.loglik, result.singular)
    if not result.converged:
        warnings.append({"code": "convergence_failure", "message": "the mixed model did not converge"})
    if result.singular:
        if _is_subject_intercept(terms, prep.subject):
            message = ("the subject variance is estimated at zero (singular fit, as lme4's isSingular): "
                       "the model reduces to ordinary regression")
        else:
            at = [n for n, t, lo in zip(model.theta_names, result.theta, model.theta_lower)
                  if lo == 0 and t < SINGULAR_TOL]
            message = ("a random-effect standard deviation is estimated at zero, or a correlation at +-1 "
                       f"(singular fit, as lme4's isSingular): {', '.join(at)}")
        warnings.append({"code": "singular_fit", "message": message})
    return result


def _variance_details(result, subject: str) -> dict[str, Any]:
    """The variance components (R's VarCorr shape), theta, and for a lone
    subject intercept the subject variance and ICC as well."""
    model = result.model
    sigma2 = float(result.sigma2)
    out: dict[str, Any] = {
        "random": random_formula(result.terms),
        "residual_variance": _num(sigma2),
        "variance_components": [{k: (_num(v) if isinstance(v, float) else v) for k, v in row.items()}
                                for row in model.variance_components(result.theta, sigma2)],
        "theta": [_num(t) for t in np.atleast_1d(result.theta)],
        "theta_names": list(model.theta_names),
    }
    if _is_subject_intercept(result.terms, subject):
        group_var = float(result.theta[0]) ** 2 * sigma2
        total = group_var + sigma2
        out["subject_intercept_variance"] = _num(group_var)
        out["icc"] = _num(group_var / total) if total > 0 else None
    return out


def _setup(data: pd.DataFrame, dv: str, between: Sequence[str], within: Sequence[str], subject: str | None,
           levels: dict[str, list[Any]] | None, covariates: Sequence[str], random: Any, reml: bool):
    """Shared by :func:`lmm` and :func:`lmm_pairwise`: validate, prepare,
    fit. Returns ``(prep, frame, names, result, factors)``."""
    between, within, covariates = list(between), list(within), list(covariates)
    factors = [*between, *within]
    if not subject:
        raise ValueError("a linear mixed model needs subject=")
    clash = [c for c in covariates if c in factors or c == subject or c == dv]
    if clash:
        raise ValueError(f"covariate(s) {clash} are also the dv, the subject or a factor")
    terms = parse_random(random, subject)
    extra = list(dict.fromkeys([*covariates, *[c for c in random_columns(terms) if c not in factors]]))
    prep = _Prepared(data, dv, factors, subject, levels, extra=extra)
    for f in factors:
        if len(prep.levels[f]) < 2:
            raise ValueError(f"factor {f!r} has fewer than 2 levels")
    for c in covariates:
        if not pd.api.types.is_numeric_dtype(prep.frame[c]):
            raise ValueError(f"covariate {c!r} must be numeric")
    _check_identifiable(prep)
    _incomplete_note(prep, within, prep.warnings)
    frame, names = _model_frame(prep, factors, covariates)
    result = _fit(_fixed_formula(factors, covariates, names), frame, reml, prep, terms)
    return prep, frame, names, result, factors


def _formula_text(dv: str, factors: list[str], covariates: Sequence[str], terms: Sequence[RandomTerm]) -> str:
    fixed = " * ".join([*factors, *covariates]) or "1"
    return f"{dv} ~ {fixed} + {random_formula(terms)}"


#: patsy's name for one Sum-coded column: ``C(f0__, Sum)[S.<level>]``.
_SUM_COLUMN = re.compile(r"^C\((f\d+__), Sum\)\[S\.(.*)\]$")


def _column_names(column: str, names: dict[str, str], levels: dict[str, list[Any]]) -> tuple[str, str]:
    """``(r_name, label)`` for one fixed-effect column.

    ``r_name`` is lme4's name under ``contr.sum``: the factor followed by the
    1-based index of the level the column codes (``group1:time2``), and a
    covariate by its own name (``group1:x``), so a fit can be compared with
    R's coefficient for coefficient. ``label`` is for people:
    ``group[ctl] : time[t2]``.
    """
    if column == "Intercept":
        return "(Intercept)", "(Intercept)"
    back = {safe: f for f, safe in names.items()}
    r_parts, labels = [], []
    for part in column.split(":"):
        match = _SUM_COLUMN.match(part)
        if match is None:
            if part in back:  # a covariate
                r_parts.append(back[part])
                labels.append(back[part])
                continue
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


# --- type-III hypotheses -------------------------------------------------------
# Sources: Goodnight (1980), "Tests of hypotheses in fixed effects linear
# models", Communications in Statistics A 9(2), 167-180; SAS/STAT User's
# Guide, "The Four Types of Estimable Functions"; for mixed models,
# Kuznetsova, Brockhoff & Christensen (2017), JSS 82(13). Written from those
# definitions, not from any implementation's source: csv-stats is MIT
# (docs/claude/lmm-r-oracle-testing.md in the scistack repo, "Licence").
#
# Why the construction matters: the F statistic depends only on the ROW SPACE
# of a term's hypothesis matrix L, but the Fai-Cornelius ddf change with any
# non-orthogonal change of its rows (scaling, mixing; not their order).
# Selecting a term's sum-coded coefficients gives lmerTest's F but, on
# unbalanced data, a ddf up to 1% off (2026-10-19). The type-III basis below
# is the one lmerTest reports; the R fixtures check it.

def term_columns(result, names: Mapping[str, str]) -> dict[str, np.ndarray]:
    """``{term label: its columns of the fixed-effect design}``, model order,
    the intercept left out. Labels use the data's names (``group:x``), from
    patsy's design info over the safe names of :func:`_model_frame`."""
    back = {safe: f for f, safe in names.items()}
    k = len(result.columns)
    out = {}
    for term, columns in result.design_info.term_name_slices.items():
        if term == "Intercept":
            continue
        label = ":".join(back[part.split(",")[0].replace("C(", "").strip()] for part in term.split(":"))
        out[label] = np.arange(k)[columns]
    return out


#: Entries below this x the largest |entry| are round-off from the
#: pseudo-inverses; zeroed so an exactly-zero hypothesis row is recognised.
_CLEAN = 1e-10


def _clean(A: np.ndarray) -> np.ndarray:
    A = np.array(A, dtype=float)
    top = float(np.abs(A).max()) if A.size else 0.0
    A[np.abs(A) <= _CLEAN * top] = 0.0
    return A


@dataclass(frozen=True)
class _ModelTerm:
    """One fixed-effect term by its variables: ``group:x`` has the factor
    ``group`` and the covariate ``x``."""

    label: str
    factors: tuple[str, ...]
    covariates: tuple[str, ...]

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(self.factors + self.covariates)

    def contains(self, other: "_ModelTerm") -> bool:
        """Goodnight's containment: every variable of ``other`` is in this
        term, and this term has more. Covariates count as variables."""
        return other.variables < self.variables


def _model_terms(labels: Sequence[str], covariates: Sequence[str]) -> list[_ModelTerm]:
    cov = set(covariates)
    out = []
    for label in labels:
        parts = label.split(":")
        out.append(_ModelTerm(label, tuple(p for p in parts if p not in cov), tuple(p for p in parts if p in cov)))
    return out


def _cell_design(terms: list[_ModelTerm], n_levels: Mapping[str, int], codes: Mapping[str, np.ndarray],
                 values: Mapping[str, np.ndarray], n: int, all_levels: bool):
    """An intercept and, per term, one column per combination of its
    factors' levels, times the product of its covariates.

    ``all_levels=False``: each factor's first level is left out (treatment
    coding; full rank for a full factorial). ``all_levels=True``: every
    level kept (the complete, rank-deficient design). Returns the matrix,
    each column's key ``(term, level indices)`` and each term's columns."""
    cols, keys, where = [np.ones(n)], [(INTERCEPT, ())], {}
    for t in terms:
        first = 0 if all_levels else 1
        base = np.ones(n)
        for c in t.covariates:
            base = base * values[c]
        start = len(cols)
        for combo in itertools.product(*[range(first, n_levels[f]) for f in t.factors]):
            col = base.copy()
            for f, lv in zip(t.factors, combo):
                col = col * (codes[f] == lv)
            cols.append(col)
            keys.append((t.label, combo))
        where[t.label] = np.arange(start, len(cols))
    return np.column_stack(cols), keys, where


def _sequential_hypotheses(X: np.ndarray, term_cols: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One term: type III is type I (Goodnight 1980), the rows of L' in the
    LDL' decomposition of X'X, i.e. the Cholesky factor's rows scaled to a
    unit diagonal."""
    R = np.linalg.cholesky(X.T @ X).T
    U = _clean(R / np.diag(R)[:, None])
    return {t: U[cols] for t, cols in term_cols.items()}


def type3_hypotheses(frame: pd.DataFrame, names: Mapping[str, str], factors: Sequence[str],
                     covariates: Sequence[str], X: np.ndarray,
                     term_cols: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``{term label: L}``: each term's type-III hypothesis on the
    coefficients of ``X`` (the fit's sum-coded design). ``term_cols``: term
    label -> its columns of X, model order, intercept excluded.

    Type III (Goodnight 1980): term T's hypothesis involves the parameters
    of T and of the terms containing T, and is orthogonal to those of every
    term containing T, in a treatment-coded parametrization:
    1. X_t, treatment-coded (full rank), and X_c, the complete design (every
       level of every factor); one column per (term, level combination),
       covariates multiplied in.
    2. X_c's columns expressed in X_t's parameters: H = (X_t^+ X_c)'.
    3. Highest-order terms first, each term's parameter columns of H are
       replaced by their least-squares residuals on the columns of every
       term that contains it.
    4. The rows of H at X_t's own keys give T's hypothesis on X_t's
       parameters, and X_t^+ X carries it to X's. All-zero rows are dropped.

    With a covariate, ``group`` is contained in ``group:x``, so the group
    hypothesis is the group difference at x = 0 (:func:`covariate_zero_terms`).
    """
    labels = list(term_cols)
    if len(labels) <= 1:
        return _sequential_hypotheses(X, term_cols)
    terms = _model_terms(labels, covariates)
    n = len(frame)
    codes = {f: frame[names[f]].cat.codes.to_numpy() for f in factors}
    n_levels = {f: len(frame[names[f]].cat.categories) for f in factors}
    values = {c: frame[names[c]].to_numpy(float) for c in covariates}
    Xt, t_keys, t_where = _cell_design(terms, n_levels, codes, values, n, all_levels=False)
    Xc, c_keys, _ = _cell_design(terms, n_levels, codes, values, n, all_levels=True)
    Xt_pinv = np.linalg.pinv(Xt)
    H = _clean((Xt_pinv @ Xc).T)
    for t in sorted(terms, key=lambda t: -len(t.variables)):  # stable: model order within an order
        above = [u for u in terms if u.contains(t)]
        if not above:
            continue
        C = H[:, np.concatenate([t_where[u.label] for u in above])]
        Y = H[:, t_where[t.label]]
        coef, *_ = np.linalg.lstsq(C, Y, rcond=None)
        H[:, t_where[t.label]] = _clean(Y - C @ coef)
        log.debug("lmm type III: %s orthogonal to %s", t.label, [u.label for u in above])
    H = H[[c_keys.index(k) for k in t_keys]]
    to_X = _clean(Xt_pinv @ X)
    out = {}
    for t in terms:
        Lt = H[:, t_where[t.label]].T
        Lt = Lt[np.abs(Lt).sum(axis=1) > 1e-8]
        out[t.label] = Lt @ to_X
        log.debug("lmm type III: %s basis %s", t.label, out[t.label].shape)
    return out


def covariate_zero_terms(labels: Sequence[str], covariates: Sequence[str]) -> dict[str, list[str]]:
    """``{term: [covariates]}`` for every term whose type-III hypothesis sits
    at those covariates' ZERO: the term is contained in a term with a
    covariate it lacks (``group`` inside ``group:x``). On an uncentred
    covariate (age 40-80) that is an extrapolation; centring the covariate
    moves it to the mean."""
    terms = _model_terms(labels, covariates)
    out = {}
    for t in terms:
        cs = sorted({c for u in terms if u.contains(t) for c in u.covariates if c not in t.covariates})
        if cs:
            out[t.label] = cs
    return out


def fai_cornelius_ddf(nu: list[float], tol: float = 1e-8) -> float:
    """Denominator df of an F test from its eigen-contrasts' df (Fai &
    Cornelius 1996), exactly as lmerTest's ``get_Fstat_ddf``: one df is
    itself; all equal gives their mean; any <= 2 gives 2; otherwise
    E = sum nu / (nu - 2) and ddf = 2E / (E - q)."""
    if len(nu) == 1:
        return float(nu[0])
    arr = np.asarray(nu, dtype=float)
    if np.all(np.abs(np.diff(arr)) < tol):
        return float(arr.mean())
    if np.any(arr <= 2):
        return 2.0
    E = float(np.sum(arr / (arr - 2)))
    return 2.0 * E / (E - len(arr))


def _term_test(result, L: np.ndarray, beta: np.ndarray, cov: np.ndarray, label: str,
               unavailable: list[str]) -> dict[str, Any]:
    """The type-III F test of one term (``L beta = 0``) with Satterthwaite
    denominator df: lmerTest's ``contestMD`` (stage 4).

    Eigen-decompose L C L'. Each eigenvector m gives a single-df contrast
    l_m = P_m' L with variance d_m, whose df is :func:`satterthwaite_df`.
    F = sum (l_m' beta)^2 / d_m / q, ddf from :func:`fai_cornelius_ddf`.
    The Wald chi2 of the same hypothesis is kept under ``"wald"``; if any
    contrast's df is unavailable, the Wald test IS the result, labelled.
    """
    lb = L @ beta
    VL = L @ cov @ L.T
    try:
        chi2 = float(lb @ np.linalg.solve(VL, lb))
    except np.linalg.LinAlgError:
        chi2 = float("nan")
    q_full = L.shape[0]
    wald = {"chi2": _num(chi2), "df": float(q_full),
            "p": _num(stats.chi2.sf(chi2, q_full)) if np.isfinite(chi2) else None}
    d, P = np.linalg.eigh(VL)
    order = np.argsort(d)[::-1]
    d, P = d[order], P[:, order]
    tol = max(np.finfo(float).eps ** 0.5 * d[0], 0.0)  # lmerTest: sqrt(eps) * the largest
    q = int(np.sum(d > tol))
    PtL = (P.T @ L)[:q]
    nus, why = [], None
    for m in range(q):
        nu, why = satterthwaite_df(result, PtL[m])
        if nu is None:
            break
        nus.append(nu)
    base = {"term": label, "sphericity_correction": None, "estimate": None,
            "effect_size": None, "effect_sizes": {}, "wald": wald}
    if why is not None or q == 0:
        unavailable.append(f"{label}: {why or 'the term has rank 0'}")
        return {**base, "statistic_name": "chi2", "statistic": wald["chi2"], "df": [float(q_full)],
                "df_method": "wald", "p": wald["p"]}
    F = float(np.sum((PtL @ beta) ** 2 / d[:q]) / q)
    ddf = fai_cornelius_ddf(nus)
    return {**base, "statistic_name": "F", "statistic": _num(F), "df": [float(q), _num(ddf)],
            "df_method": "satterthwaite", "p": _num(stats.f.sf(F, q, ddf)),
            "eigen_df": [_num(v) for v in nus]}


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
    covariates: Sequence[str] = (),
    random: Sequence[Mapping[str, Any] | RandomTerm] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    reml: bool = True,
) -> dict[str, Any]:
    """Linear mixed model ``dv ~ factors * covariates (full factorial) + random``;
    ``random`` defaults to ``(1 | subject)`` (see the module docstring for
    its shape).

    One effect per fixed term: an F test of all its coefficients with
    Satterthwaite denominator df, on the term's type-III hypothesis
    (:func:`type3_hypotheses`). Sum-to-zero coding, so a main effect is
    tested at the average of the other factors. A term contained in one with
    a covariate (``group`` in ``group:x``) is tested at that covariate = 0,
    as in lmerTest and SAS; a ``type3_at_covariate_zero`` warning names
    these terms.

    The Wald chi2 of the same hypothesis is kept under each effect's
    ``"wald"``. It becomes the result (``df_method: "wald"``, plus a
    ``satterthwaite_unavailable`` warning) when a df cannot be computed.

    ``details`` has the coefficients (with t tests), the variance components,
    theta, the log-likelihood and convergence. No subject is dropped for
    missing a within level; the ones that do are named in an
    ``incomplete_within_used`` warning. With neither factors nor covariates
    the model is intercept-only and there are no term effects.
    """
    prep, frame, names, result, factors = _setup(data, dv, between, within, subject, levels, covariates,
                                                 random, reml)
    covariates = list(covariates)
    cov = result.cov_beta
    beta = result.beta
    effects = []
    unavailable: list[str] = []
    term_cols = term_columns(result, names)
    # Type-III bases for every design, so the Fai–Cornelius ddf match
    # lmerTest's (the F is the same for any basis; the ddf is not).
    contrasts = type3_hypotheses(frame, names, factors, covariates, result.X, term_cols) if term_cols else {}
    at_zero = covariate_zero_terms(list(term_cols), covariates)
    if at_zero:
        what = "; ".join(f"{t} at {', '.join(cs)} = 0" for t, cs in at_zero.items())
        prep.warnings.append({"code": "type3_at_covariate_zero",
                              "message": f"Type-III tests (as lmerTest): {what}. Centre the covariate to test "
                                         f"these effects at its mean instead."})
        log.info("lmm type III at covariate zero: %s", at_zero)
    for label in term_cols:
        effects.append(_term_test(result, contrasts[label], beta, cov, label, unavailable))
    if unavailable:
        prep.warnings.append({"code": "satterthwaite_unavailable",
                              "message": "Wald chi2 used for: " + "; ".join(unavailable)})
        log.info("lmm term Satterthwaite unavailable: %s", unavailable)
    details = {
        "formula": _formula_text(dv, factors, covariates, result.terms),
        "reml": reml, "converged": bool(result.converged), "optimizer": result.optimizer_path,
        "log_likelihood": _num(result.loglik),
        "coefficients": _coefficients(result, ci_level, names, prep.levels, prep.warnings),
        "coding": "sum-to-zero", "term_tests": "type3", **_variance_details(result, prep.subject),
    }
    return _record("lmm", "Linear mixed model (Satterthwaite)", prep, between=list(between), within=list(within),
                   alpha=alpha, ci_level=ci_level, tails="two-sided", effects=effects, assumptions=[],
                   details=details, extra_engines=["patsy"])


def _p_t(stat: float, df: float | None, tails: str) -> float:
    """p of a t (df) or, with df None, a z statistic, for ``tails``."""
    dist = stats.norm if df is None else stats.t(df)
    if tails == "two-sided":
        return float(2 * dist.sf(abs(stat)))
    return float(dist.sf(stat) if tails == "greater" else dist.cdf(stat))


def lmm_pairwise(
    data: pd.DataFrame,
    dv: str,
    *,
    between: Sequence[str] = (),
    within: Sequence[str] = (),
    subject: str | None = None,
    levels: dict[str, list[Any]] | None = None,
    factor: str | None = None,
    random: Sequence[Mapping[str, Any] | RandomTerm] | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    tails: str = "two-sided",
    reml: bool = True,
) -> dict[str, Any]:
    """Every pairwise difference ``a − b`` between the levels of ``factor``,
    within each combination of the OTHER factors, from ONE full model
    ``dv ~ factors (full factorial) + random`` (stage 5; ``random`` as in
    :func:`lmm`, default ``(1 | subject)``).

    This is what R's emmeans does with every other factor as a ``by``
    variable: the estimate is the difference of the two cells' model
    predictions (``l = x(a, context) − x(b, context)``). Its SE comes from
    the full model's variance estimates, which use all the data, and its df
    are Satterthwaite (:func:`satterthwaite_df`). Checked against emmeans
    (``lmer.df = "satterthwaite"``) on the reference datasets.

    A subject missing some cells is kept: it informs every estimate through
    the random effects, where a paired t-test would drop it.

    One effect per (context, pair), each with ``a``, ``b`` and ``context``
    (``{other factor: level}``, empty with one factor). ``factor`` may be
    omitted when the model has one factor.
    """
    import patsy

    _check_tails(tails)
    factors = [*between, *within]
    if not factors:
        raise ValueError("lmm_pairwise needs at least one factor")
    if factor is None:
        if len(factors) != 1:
            raise ValueError(f"lmm_pairwise needs factor= when the model has several factors: {factors}")
        factor = factors[0]
    if factor not in factors:
        raise ValueError(f"factor {factor!r} is not one of {factors}")
    prep, frame, names, result, factors = _setup(data, dv, between, within, subject, levels, (), random, reml)

    others = [f for f in factors if f != factor]
    lv = prep.levels[factor]
    contexts = [dict()]
    for o in others:
        contexts = [{**c, o: v} for c in contexts for v in prep.levels[o]]
    pairs = [(lv[i], lv[j]) for i in range(len(lv)) for j in range(i + 1, len(lv))]

    def row(level: Any, context: dict) -> np.ndarray:
        cell = {factor: level, **context}
        grid = pd.DataFrame({names[f]: pd.Categorical([cell[f]], categories=prep.levels[f]) for f in factors})
        return np.asarray(patsy.build_design_matrices([result.design_info], grid)[0], dtype=float)[0]

    crit_z = stats.norm.ppf(0.5 + ci_level / 2)
    effects, unavailable = [], []
    for context in contexts:
        x = {level: row(level, context) for level in lv}
        for a, b in pairs:
            l = x[a] - x[b]
            est = float(l @ result.beta)
            se = float(np.sqrt(l @ result.cov_beta @ l))
            stat = est / se if se > 0 else float("nan")
            df, why = satterthwaite_df(result, l)
            context_text = {k: _text(v) for k, v in context.items()}
            if df is None:
                unavailable.append(f"{_text(a)} - {_text(b)} {context_text or ''}: {why}")
                crit = crit_z
            else:
                crit = stats.t.ppf(0.5 + ci_level / 2, df)
            effects.append({
                "term": factor, "a": _text(a), "b": _text(b), "context": context_text,
                "statistic_name": "z" if df is None else "t", "statistic": _num(stat),
                "df": [] if df is None else [_num(df)],
                "df_method": "wald" if df is None else "satterthwaite",
                "p": _num(_p_t(stat, df, tails)),
                "estimate": {"name": "mean_difference", "value": _num(est),
                             "ci": [_num(est - crit * se), _num(est + crit * se)],
                             "se": _num(se), "sd_of_differences": None, "method": "lmm_emm"},
                "effect_size": None, "effect_sizes": {},
                "wald": {"z": _num(stat), "p": _num(_p_t(stat, None, tails))},
            })
    if unavailable:
        prep.warnings.append({"code": "satterthwaite_unavailable",
                              "message": "Wald z used for: " + "; ".join(unavailable)})
        log.info("lmm_pairwise Satterthwaite unavailable: %s", unavailable)
    details = {
        "formula": _formula_text(dv, factors, (), result.terms), "reml": reml,
        "contrasts": f"pairwise {factor} within each combination of {others or 'nothing'} (full model)",
        "converged": bool(result.converged), "log_likelihood": _num(result.loglik),
        **_variance_details(result, prep.subject),
    }
    return _record("lmm_pairwise", "Linear mixed model contrasts (Satterthwaite)", prep, between=list(between),
                   within=list(within), alpha=alpha, ci_level=ci_level, tails=tails, effects=effects,
                   assumptions=[], details=details, extra_engines=["patsy"])
