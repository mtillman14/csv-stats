"""csvstats.lmm's type-III hypotheses, checked without R.

Stage 2 of .claude/plan-lmm-covariate-type3.md (scistack repo). R's
lmerTest is the oracle in test_lmm_scenarios.py; these pin the properties
that follow from the type-III definition (Goodnight 1980) itself:
- one term: the unit-diagonal LDL' rows of X'X
- each term's hypothesis spans its own sum-coded coefficients
- containment counts covariates (group is inside group:x), so a factor is
  tested at covariate = 0, and the user is told
- shifting a covariate changes a contained factor's test, never the
  highest-order term's
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csvstats.lmm import _setup, covariate_zero_terms, lmm, term_columns, type3_hypotheses


def _data(n_subjects: int = 14, seed: int = 5, offset: float = 0.0) -> pd.DataFrame:
    """group (between, 2) x time (within, 3) x reps 2, x per observation on
    [offset, offset + 1], with a group x time effect, an x slope that differs
    by group, and ~15% of rows dropped so the design is unbalanced."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        g = "ctl" if s % 2 else "trt"
        u = rng.normal(0, 1)
        for k, t in enumerate(["t1", "t2", "t3"]):
            for _ in range(2):
                x = rng.uniform()
                y = 10 + u + 0.5 * k + (0.4 * k if g == "trt" else 0) + (1.0 + 0.5 * (g == "trt")) * x
                rows.append({"subject": f"S{s:02d}", "group": g, "time": t, "x": x + offset,
                             "y": y + rng.normal()})
    frame = pd.DataFrame(rows)
    return frame.drop(frame.sample(frac=0.15, random_state=seed).index).reset_index(drop=True)


def _hypotheses(data: pd.DataFrame, between, within, covariates):
    prep, frame, names, result, factors = _setup(data, "y", between, within, "subject", None, covariates, None, True)
    cols = term_columns(result, names)
    return type3_hypotheses(frame, names, factors, covariates, result.X, cols), cols, result


def test_one_term_is_the_unit_diagonal_ldl_rows():
    """With one term, type III is type I: X'X = U' D U with U unit upper
    triangular, and the term's rows are U's."""
    X = np.column_stack([np.ones(20), np.random.default_rng(0).normal(size=(20, 2))])
    (U,) = type3_hypotheses(pd.DataFrame(), {}, [], [], X, {"t": np.arange(3)}).values()
    assert np.allclose(np.diag(U), 1.0) and np.allclose(np.tril(U, -1), 0.0)
    Ui = np.linalg.inv(U)
    D = Ui.T @ X.T @ X @ Ui
    assert np.allclose(D, np.diag(np.diag(D)), atol=1e-10)


@pytest.mark.parametrize("design", [
    (["group"], ["time"], ["x"]),
    ([], ["time"], ["x"]),
    (["group"], ["time"], []),
    (["group"], [], ["x"]),
])
def test_each_hypothesis_spans_its_own_sum_coded_coefficients(design):
    """Full factorial, sum coding, raw covariates: a term's type-III
    hypothesis has the same row space as selecting its coefficients (so the
    F is the selection's; only the basis, and with it the ddf, differs)."""
    hyp, cols, result = _hypotheses(_data(), *design)
    k = len(result.columns)
    for term, L in hyp.items():
        S = np.eye(k)[cols[term]]
        assert L.shape[0] == S.shape[0], term
        assert np.linalg.matrix_rank(np.vstack([L, S]), tol=1e-8) == S.shape[0], term


def test_covariates_count_for_containment():
    terms = ["group", "time", "x", "group:time", "group:x", "time:x", "group:time:x"]
    assert covariate_zero_terms(terms, ["x"]) == {"group": ["x"], "time": ["x"], "group:time": ["x"]}
    assert covariate_zero_terms(["x", "z", "x:z"], ["x", "z"]) == {"x": ["z"], "z": ["x"]}
    assert covariate_zero_terms(["group", "time", "group:time"], []) == {}


def test_a_shifted_covariate_moves_the_contained_factor_not_the_top_term():
    """x -> x + 10 reparametrizes the model. group:time:x (highest order)
    keeps its F and ddf; group, tested at x = 0, does not keep its F."""
    a = lmm(_data(), "y", between=["group"], within=["time"], subject="subject", covariates=["x"])
    b = lmm(_data(offset=10.0), "y", between=["group"], within=["time"], subject="subject", covariates=["x"])
    ea, eb = ({e["term"]: e for e in r["effects"]} for r in (a, b))
    top = "group:time:x"
    assert eb[top]["statistic"] == pytest.approx(ea[top]["statistic"], rel=1e-6)
    assert eb[top]["df"][1] == pytest.approx(ea[top]["df"][1], rel=1e-5)
    assert eb["group"]["statistic"] != pytest.approx(ea["group"]["statistic"], rel=1e-3)


def test_one_df_terms_are_the_coefficient_t_tests():
    """A 2-level factor with x: every term has one df, so F = t^2 and the
    ddf is the coefficient's df: group at x = 0, x averaged over groups."""
    rec = lmm(_data(), "y", between=["group"], subject="subject", covariates=["x"])
    coefs = {c["name"]: c for c in rec["details"]["coefficients"]}
    for term, name in [("group", "group1"), ("x", "x"), ("group:x", "group1:x")]:
        e = next(e for e in rec["effects"] if e["term"] == term)
        assert e["statistic"] == pytest.approx(coefs[name]["t"] ** 2, rel=1e-8), term
        assert e["df"][1] == pytest.approx(coefs[name]["df"], rel=1e-8), term


def test_the_user_is_told_which_terms_sit_at_covariate_zero():
    rec = lmm(_data(), "y", between=["group"], within=["time"], subject="subject", covariates=["x"])
    (w,) = [w for w in rec["warnings"] if w["code"] == "type3_at_covariate_zero"]
    assert "group at x = 0" in w["message"] and "group:time at x = 0" in w["message"]
    assert rec["details"]["term_tests"] == "type3"
    plain = lmm(_data(), "y", subject="subject", covariates=["x"])
    assert "type3_at_covariate_zero" not in {w["code"] for w in plain["warnings"]}
    factors = lmm(_data(), "y", between=["group"], within=["time"], subject="subject")
    assert "type3_at_covariate_zero" not in {w["code"] for w in factors["warnings"]}
