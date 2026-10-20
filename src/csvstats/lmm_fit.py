"""The linear mixed model fit: REML (or ML) for any lme4-style random effects.

    y = X beta + Z b + e,   b = Lambda(theta) u,   u ~ N(0, sigma^2 I),   e ~ N(0, sigma^2 I)

One owner for the criterion, the optimizer and the variance components, for
every random-effect structure: one random intercept, random slopes
(correlated, or uncorrelated with ``||``), several terms per group, crossed
and nested groupings, and interaction groupings (``subject:time``).

Implemented from the published method: Bates, Maechler, Bolker & Walker
(2015), "Fitting Linear Mixed-Effects Models Using lme4", JSS 67(1), section 3
(penalized least squares, profiled deviance). NOT from lme4's source: lme4 is
GPL, csv-stats is MIT. R is used only as a black-box oracle
(tests/reference/make_lmm_scenarios.R, tests/test_lmm_scenarios.py).

The conventions that make theta comparable with R's follow lme4's documented
behaviour:

- ``||`` splits a term into one block per column.
- Blocks are reordered by decreasing number of levels, only when a later
  block has more levels than an earlier one.
- theta is the lower triangle of each block's template, column-major, with
  0 lower bounds on the diagonal.
- Names are ``group.col`` (diagonal) and ``group.row.col`` (off-diagonal).
- A fit is singular when any diagonal theta is below 1e-4.

Plan: .claude/plan-lmm-general-fit.md in the scistack repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import log

#: lme4's isSingular tolerance: a diagonal theta (a relative standard
#: deviation, sigma_b / sigma) below this is "on the boundary".
SINGULAR_TOL = 1e-4

INTERCEPT = "(Intercept)"


class _Fit:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


# --- the random-effect specification ---------------------------------------

@dataclass(frozen=True)
class RandomTerm:
    """One ``(terms | group)``: ``group`` names one or more columns (several
    = their interaction, ``subject:time``); ``terms`` are ``"(Intercept)"``,
    numeric columns, or model factors (contr.sum coded, as R). ``correlated``
    False is lme4's ``||``."""

    group: tuple[str, ...]
    terms: tuple[str, ...] = (INTERCEPT,)
    correlated: bool = True

    @property
    def label(self) -> str:
        return ":".join(self.group)


def parse_random(random: Iterable[Any] | None, subject: str) -> list[RandomTerm]:
    """``random=`` as given to :func:`csvstats.lmm.lmm`: None means
    ``(1 | subject)``; otherwise a list of :class:`RandomTerm` or of
    ``{"group": [...] | str, "terms": [...], "correlated": bool}`` (the shape
    the R fixtures record)."""
    if random is None:
        return [RandomTerm((subject,))]
    out = []
    for spec in random:
        if isinstance(spec, RandomTerm):
            out.append(spec)
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"a random term is a dict with group/terms/correlated, got {spec!r}")
        group = spec.get("group")
        group = (group,) if isinstance(group, str) else tuple(group or ())
        terms = spec.get("terms", (INTERCEPT,))
        terms = (terms,) if isinstance(terms, str) else tuple(terms)
        if not group:
            raise ValueError(f"random term {spec!r} has no group")
        if not terms:
            raise ValueError(f"random term {spec!r} has no terms")
        out.append(RandomTerm(group, terms, bool(spec.get("correlated", True))))
    if not out:
        raise ValueError("random= is empty: a mixed model needs at least one random term")
    return out


def random_formula(terms: Sequence[RandomTerm]) -> str:
    """The lme4 spelling: ``(1 + x | subject) + (1 | item)``."""
    parts = []
    for t in terms:
        lhs = " + ".join("1" if c == INTERCEPT else c for c in t.terms)
        parts.append(f"({lhs} {'|' if t.correlated else '||'} {t.label})")
    return " + ".join(parts)


def random_columns(terms: Sequence[RandomTerm]) -> list[str]:
    """Every data column the random part reads (groups and slope columns)."""
    cols: list[str] = []
    for t in terms:
        for c in (*t.group, *t.terms):
            if c != INTERCEPT and c not in cols:
                cols.append(c)
    return cols


# --- blocks -----------------------------------------------------------------

@dataclass
class Block:
    """One random-effect block after ``||`` expansion: ``n_levels`` levels of
    ``label``, each with ``p`` effects (the columns of ``Xk``)."""

    label: str
    columns: list[str]
    Xk: np.ndarray            # n x p
    codes: np.ndarray         # n, level index 0..n_levels-1
    n_levels: int
    # Filled by MixedModel: where this block sits in u and in theta.
    q0: int = 0
    t0: int = 0
    tri: list[tuple[int, int]] = field(default_factory=list)
    W: np.ndarray | None = None      # see whitening()
    Winv: np.ndarray | None = None

    @property
    def p(self) -> int:
        return self.Xk.shape[1]

    def whitening(self) -> np.ndarray:
        """W with ``Xk = Xw @ W``, Xw the block's columns centred (when the
        block has an intercept) and scaled to unit SD (RMS without one).

        The optimizer works on theta for Xw: an uncentred covariate (x on
        [40, 80]) makes raw theta badly scaled (slope ~0.01 vs intercept ~1)
        and the intercept and slope strongly confounded, and L-BFGS-B
        stalled in that valley (RIS scenarios, 2026-10-20)."""
        p = self.p
        W = np.eye(p)
        i0 = self.columns.index(INTERCEPT) if INTERCEPT in self.columns else None
        for j in range(p):
            if j == i0:
                continue
            v = self.Xk[:, j]
            if i0 is not None:
                m, s = float(v.mean()), float(v.std())
                if s > 0:
                    W[i0, j], W[j, j] = m, s
            else:
                s = float(np.sqrt(np.mean(v ** 2)))
                if s > 0:
                    W[j, j] = s
        return W

    @property
    def q(self) -> int:
        return self.n_levels * self.p


def _term_columns(frame: pd.DataFrame, term: str, levels: Mapping[str, Sequence[Any]]) -> list[tuple[str, np.ndarray]]:
    n = len(frame)
    if term == INTERCEPT:
        return [(INTERCEPT, np.ones(n))]
    if term not in frame.columns:
        raise ValueError(f"random-effect column {term!r} is not in the data")
    if term in levels:
        # contr.sum, as R under options(contrasts = c("contr.sum", ...)):
        # column i is +1 at level i, -1 at the last level.
        lv = [str(v) for v in levels[term]]
        values = frame[term].astype(str).to_numpy()
        last = values == lv[-1]
        return [(f"{term}{i + 1}", (values == level).astype(float) - last.astype(float))
                for i, level in enumerate(lv[:-1])]
    col = frame[term]
    if not pd.api.types.is_numeric_dtype(col):
        raise ValueError(f"random slope {term!r} is neither numeric nor a model factor (between/within)")
    return [(term, col.to_numpy(float))]


def build_blocks(frame: pd.DataFrame, terms: Sequence[RandomTerm],
                 levels: Mapping[str, Sequence[Any]] | None = None) -> list[Block]:
    """The blocks of ``terms`` over ``frame`` (one row per observation),
    in the FORMULA order (:class:`MixedModel` applies lme4's reordering).
    ``levels``: the model factors and their level order, for factor slopes."""
    levels = levels or {}
    blocks = []
    for t in terms:
        missing = [g for g in t.group if g not in frame.columns]
        if missing:
            raise ValueError(f"random-effect group column(s) {missing} are not in the data")
        if len(t.group) == 1:
            key = frame[t.group[0]].astype(str)
        else:
            key = frame[list(t.group)].astype(str).agg(":".join, axis=1)
        codes, uniques = pd.factorize(key, sort=True)
        cols = [c for term in t.terms for c in _term_columns(frame, term, levels)]
        if not t.correlated:
            factor_terms = [term for term in t.terms if term in levels]
            if factor_terms:
                raise ValueError(
                    f"uncorrelated (||) random effects of a factor ({factor_terms}) are not supported: lme4 expands "
                    f"them into a CORRELATED dummy-coded block; write the terms explicitly instead")
            groups = [[c] for c in cols]
        else:
            groups = [cols]
        for g in groups:
            blocks.append(Block(label=t.label, columns=[name for name, _ in g],
                                Xk=np.column_stack([v for _, v in g]), codes=codes.astype(int),
                                n_levels=len(uniques)))
    return blocks


def lme4_order(blocks: Sequence[Block]) -> list[Block]:
    """Decreasing number of levels, only when a later block has more levels
    than an earlier one (lme4's documented term ordering); ties then come
    out in reverse."""
    nl = np.array([b.n_levels for b in blocks])
    if len(nl) > 1 and np.any(np.diff(nl) > 0):
        order = np.argsort(nl, kind="stable")[::-1]
        return [blocks[i] for i in order]
    return list(blocks)


# --- the model ----------------------------------------------------------------

class MixedModel:
    """REML (or ML) for ``y = X beta + Z b + e`` with the given blocks.

    ``deviance(theta)`` is the profiled criterion (lme4's REML criterion /
    ML deviance) at lme4's theta; ``deviance_at(varpar)`` the unprofiled one
    at ``varpar = (theta..., sigma)``; ``vcov_at(varpar)`` = Cov(beta-hat)
    there. Both feed the Satterthwaite df (csvstats.lmm).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, blocks: Sequence[Block], reml: bool = True) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.reml = reml
        self.n, self.p = X.shape
        self.blocks = lme4_order(blocks)
        # Centre y for numerical stability when X has an intercept (data
        # around 1e4 make y'y huge and r^2 a difference of large numbers);
        # the intercept is restored in fit().
        self.intercept_col = next((j for j in range(self.p) if np.allclose(X[:, j], 1.0)), None)
        self.y_shift = float(np.mean(y)) if self.intercept_col is not None else 0.0
        yc = y - self.y_shift

        q0 = t0 = 0
        names, lower = [], []
        Zs = []
        for b in self.blocks:
            b.q0, b.t0 = q0, t0
            b.tri = [(i, j) for j in range(b.p) for i in range(j, b.p)]  # column-major lower triangle
            b.W = b.whitening()
            b.Winv = np.linalg.inv(b.W)
            for i, j in b.tri:
                if i == j:
                    names.append(f"{b.label}.{b.columns[i]}")
                    lower.append(0.0)
                else:
                    names.append(f"{b.label}.{b.columns[i]}.{b.columns[j]}")
                    lower.append(-np.inf)
            Zk = np.zeros((self.n, b.q))
            rows = np.arange(self.n)
            for j in range(b.p):
                Zk[rows, b.codes * b.p + j] = b.Xk[:, j]
            Zs.append(Zk)
            q0 += b.q
            t0 += len(b.tri)
        self.q = q0
        self.theta_names = names
        self.theta_lower = np.array(lower)
        self.n_theta = len(names)
        Z = np.hstack(Zs) if Zs else np.zeros((self.n, 0))
        self.ZtZ = Z.T @ Z
        self.ZtX = Z.T @ X
        self.Zty = Z.T @ yc
        self.XtX = X.T @ X
        self.Xty = X.T @ yc
        self.yty = float(yc @ yc)
        self.evaluations = 0

    # --- theta -> Lambda ---------------------------------------------------
    def template(self, block: Block, theta: np.ndarray) -> np.ndarray:
        T = np.zeros((block.p, block.p))
        for k, (i, j) in enumerate(block.tri):
            T[i, j] = theta[block.t0 + k]
        return T

    def _recast(self, theta: np.ndarray, which: str) -> np.ndarray:
        """theta in the whitened columns -> lme4's theta (``"raw"``), or back
        (``"whitened"``). The criterion depends on theta only through
        Lambda Lambda', and z_raw b = z_w (W b), so Sigma_w = W Sigma_raw W'
        and T_raw is the lower factor of W^-1 T_w (T_w)' W^-T, recomputed
        by QR so that a singular Sigma (a zero variance, a correlation of
        +-1) maps exactly too."""
        out = np.empty(self.n_theta)
        for b in self.blocks:
            M = (b.Winv if which == "raw" else b.W) @ self.template(b, theta)
            R = np.linalg.qr(M.T, mode="r")
            T = (np.where(np.diag(R) < 0, -1.0, 1.0)[:, None] * R).T
            for k, (i, j) in enumerate(b.tri):
                out[b.t0 + k] = T[i, j]
        return out

    def _times_lambda(self, M: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``M @ Lambda`` for Lambda = blockdiag(I_levels kron T_block):
        per block, each level's p columns times T (O(rows q p), never q^3)."""
        out = np.empty_like(M)
        r = M.shape[0]
        for b in self.blocks:
            T = self.template(b, theta)
            sub = M[:, b.q0:b.q0 + b.q].reshape(r, b.n_levels, b.p)
            out[:, b.q0:b.q0 + b.q] = (sub @ T).reshape(r, b.q)
        return out

    def parts(self, theta: np.ndarray) -> dict[str, Any]:
        """Penalized least squares at ``theta``: the Cholesky factor L of
        Lambda'Z'Z Lambda + I, A = R_X'R_X (= X'V^-1 X sigma^2), beta (for the
        centred y), the spherical random effects u and the penalized residual
        sum of squares r2."""
        from scipy.linalg import cho_solve, solve_triangular

        theta = np.asarray(theta, dtype=float)
        self.evaluations += 1
        ZtZL = self._times_lambda(self.ZtZ, theta)            # Z'Z Lambda
        M = self._times_lambda(ZtZL.T, theta).T                # Lambda'Z'Z Lambda
        M[np.diag_indices_from(M)] += 1.0
        L = np.linalg.cholesky(M)
        LZtX = self._times_lambda(self.ZtX.T, theta).T         # Lambda'Z'X
        LZty = self._times_lambda(self.Zty[None, :], theta)[0]  # Lambda'Z'y
        cu = solve_triangular(L, LZty, lower=True)
        RZX = solve_triangular(L, LZtX, lower=True)
        A = self.XtX - RZX.T @ RZX
        rhs = self.Xty - RZX.T @ cu
        RX = np.linalg.cholesky(A)
        beta = cho_solve((RX, True), rhs)
        r2 = self.yty - float(cu @ cu) - float(rhs @ beta)
        u = solve_triangular(L.T, cu - RZX @ beta, lower=False)
        return {"L": L, "A": A, "RX": RX, "beta": beta, "u": u, "r2": max(r2, 0.0),
                "logdet_L2": 2.0 * float(np.log(np.diag(L)).sum()),
                "logdet_A": 2.0 * float(np.log(np.diag(RX)).sum())}

    @property
    def dof(self) -> int:
        return self.n - self.p if self.reml else self.n

    def deviance(self, theta: np.ndarray) -> float:
        """The profiled criterion at ``theta`` (lme4's REML criterion, or the
        ML deviance): log|L|^2 [+ log|R_X|^2] + dof (1 + log(2 pi r2 / dof))."""
        try:
            pt = self.parts(np.atleast_1d(theta))
        except np.linalg.LinAlgError:
            return float("inf")
        if pt["r2"] <= 0:
            return float("inf")
        out = pt["logdet_L2"] + self.dof * (1.0 + np.log(2.0 * np.pi * pt["r2"] / self.dof))
        if self.reml:
            out += pt["logdet_A"]
        return float(out)

    def deviance_at(self, varpar: np.ndarray) -> float:
        """The criterion at (theta..., sigma), NOT profiled in sigma:
        log|L|^2 [+ log|R_X|^2] + dof log(2 pi sigma^2) + r2 / sigma^2. At
        sigma^2 = r2 / dof it equals :meth:`deviance`."""
        varpar = np.asarray(varpar, dtype=float)
        theta, sigma = varpar[:-1], float(varpar[-1])
        pt = self.parts(theta)
        out = pt["logdet_L2"] + self.dof * np.log(2.0 * np.pi * sigma ** 2) + pt["r2"] / sigma ** 2
        if self.reml:
            out += pt["logdet_A"]
        return float(out)

    def vcov_at(self, varpar: np.ndarray) -> np.ndarray:
        """Cov(beta-hat) = sigma^2 (R_X'R_X)^-1 at (theta..., sigma)."""
        varpar = np.asarray(varpar, dtype=float)
        A = self.parts(varpar[:-1])["A"]
        return float(varpar[-1]) ** 2 * np.linalg.inv(A)

    # --- optimisation --------------------------------------------------------
    def _optimize_1d(self) -> tuple[np.ndarray, float, bool, str]:
        """One theta (a lone random intercept, or one scalar block): a grid,
        then bounded Brent, with theta = 0 always compared explicitly."""
        from scipy.optimize import minimize_scalar

        f = lambda t: self.deviance(np.array([t]))  # noqa: E731
        grid = np.concatenate([[0.0], np.logspace(-4, 3, 57)])
        values = np.array([f(t) for t in grid])
        i = int(np.argmin(values))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)] if i < len(grid) - 1 else grid[-1] * 100
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        theta, best = float(res.x), float(res.fun)
        path = "grid+brent"
        if f(0.0) <= best:  # the boundary is a candidate in its own right
            theta, best, path = 0.0, f(0.0), "grid+brent, boundary"
        return np.array([theta]), best, bool(res.success), path

    def _minimize(self, f, x0: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """L-BFGS-B (bounded, finite-difference gradient), then a bounded
        Nelder-Mead polish; the better of the two."""
        from scipy.optimize import Bounds, minimize

        bounds = Bounds(lower, np.full(len(x0), np.inf))
        x0 = np.maximum(x0, lower)
        r1 = minimize(f, x0, jac=lambda x: self._gradient_sub(f, x, lower), method="L-BFGS-B", bounds=bounds,
                      options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 2000})
        r2 = minimize(f, r1.x, method="Nelder-Mead", bounds=bounds,
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxfev": 400 * max(len(x0), 1), "adaptive": True})
        best = r2 if r2.fun <= r1.fun else r1
        return np.maximum(np.asarray(best.x, dtype=float), lower), float(best.fun), bool(r1.success or r2.success)

    def _gradient_sub(self, f, x: np.ndarray, lower: np.ndarray) -> np.ndarray:
        """Central differences; one-sided where a step would cross a bound."""
        g = np.zeros_like(x)
        for i in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[i]))
            e = np.zeros_like(x); e[i] = h
            if x[i] - h < lower[i]:
                g[i] = (f(x + e) - f(x)) / h
            else:
                g[i] = (f(x + e) - f(x - e)) / (2 * h)
        return g

    def _optimize(self) -> tuple[np.ndarray, float, bool, str]:
        """Several thetas, optimized in the whitened columns (Block.whitening):
        the interior optimum from two starts, the identity there and lme4's
        start (1 on the diagonal, 0 off it, in raw theta); then the
        boundary face theta_j = 0 of EVERY diagonal theta (probing only the
        thetas that ended below 0.05 can miss a face the interior merely
        crept towards; suspected in RIF_014, 2026-10-20), and theta = 0. The best criterion
        wins; the result is lme4's (raw) theta."""
        lower = self.theta_lower
        f = lambda tw: self.deviance(self._recast(tw, "raw"))  # noqa: E731
        identity = np.where(lower == 0, 1.0, 0.0)
        starts = {"whitened start": identity, "lme4 start": self._recast(identity, "whitened")}
        theta = best = ok = path = None
        for name, x0 in starts.items():
            t, val, ok_start = self._minimize(f, x0, lower)
            log.debug("lmm fit: from the %s -> %.10g", name, val)
            if best is None or val < best:
                theta, best, ok, path = t, val, ok_start, f"lbfgsb+neldermead ({name})"
        diag = [k for k in range(self.n_theta) if lower[k] == 0]
        for j in diag:
            free = [k for k in range(self.n_theta) if k != j]

            def f_face(z, j=j, free=free):
                full = np.zeros(self.n_theta); full[free] = z
                return f(full)

            z, val, ok_face = self._minimize(f_face, theta[free], lower[free])
            log.debug("lmm fit: boundary face %s = 0 (whitened) -> %.10g (interior %.10g)",
                      self.theta_names[j], val, best)
            if val < best:
                theta = np.zeros(self.n_theta); theta[free] = z
                best, ok, path = val, ok_face, f"{path}, whitened face {self.theta_names[j]}=0"
        zero = self.deviance(np.zeros(self.n_theta))
        if zero < best:
            log.info("lmm fit: theta = 0 beat the interior optimum (%s)", path)
            return np.zeros(self.n_theta), zero, ok, f"{path}, all-zero"
        if "face" in path:
            log.info("lmm fit: a boundary beat the interior optimum (%s)", path)
        return self._recast(theta, "raw"), best, ok, path

    def fit(self) -> _Fit:
        if self.n_theta == 1:
            theta, best, converged, path = self._optimize_1d()
        else:
            theta, best, converged, path = self._optimize()
        pt = self.parts(theta)
        sigma2 = pt["r2"] / self.dof
        beta = pt["beta"].copy()
        if self.intercept_col is not None:
            beta[self.intercept_col] += self.y_shift
        diag = self.theta_lower == 0
        singular = bool(np.any(theta[diag] < SINGULAR_TOL))
        log.debug("lmm fit: %s; theta=%s sigma2=%.6g criterion=%.10g singular=%s evaluations=%d",
                  path, np.round(theta, 6).tolist(), sigma2, best, singular, self.evaluations)
        return _Fit(theta=theta, beta=beta, sigma2=sigma2, cov_beta=sigma2 * np.linalg.inv(pt["A"]),
                    loglik=-best / 2.0, criterion=best, converged=converged, singular=singular,
                    optimizer_path=path, model=self)

    # --- reporting -------------------------------------------------------------
    def variance_components(self, theta: np.ndarray, sigma2: float) -> list[dict[str, Any]]:
        """R's ``as.data.frame(VarCorr(fit))`` shape: per block, one row per
        variance (``sdcor`` = SD), one per covariance (``sdcor`` =
        correlation); then the residual."""
        rows = []
        for b in self.blocks:
            T = self.template(b, theta)
            S = sigma2 * (T @ T.T)
            for i, c in enumerate(b.columns):
                rows.append({"group": b.label, "var1": c, "var2": None, "vcov": float(S[i, i]),
                             "sdcor": float(np.sqrt(max(S[i, i], 0.0)))})
            for i in range(b.p):
                for j in range(i + 1, b.p):
                    denom = np.sqrt(S[i, i] * S[j, j])
                    rows.append({"group": b.label, "var1": b.columns[i], "var2": b.columns[j],
                                 "vcov": float(S[i, j]), "sdcor": float(S[i, j] / denom) if denom > 0 else None})
        rows.append({"group": "Residual", "var1": None, "var2": None, "vcov": float(sigma2),
                     "sdcor": float(np.sqrt(sigma2))})
        return rows
