"""csvstats.lmm_fit.MixedModel, checked without R.

Stage 4 of .claude/plan-lmm-general-fit.md (scistack repo). Each check
compares the penalized-least-squares criterion with something derived
independently:
- the direct dense REML/ML formula, built from V = sigma^2 (Z Lambda Lambda' Z' + I)
- the old one-random-intercept closed form
- an equivalent formulation of the same model (|| against a correlated
  block with zero covariance)

Plus the theta layout lme4 uses, and the errors for structures csv-stats
refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csvstats.lmm import lmm
from csvstats.lmm_fit import MixedModel, RandomTerm, build_blocks, parse_random, random_formula


def _frame(n_subjects: int = 12, n_items: int = 5, reps: int = 2, seed: int = 0) -> pd.DataFrame:
    """Subjects x items x reps, a 2-level factor, a numeric x, and y with a
    random intercept and x slope per subject plus an item intercept."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        u0, u1 = rng.normal(0, 1.0), rng.normal(0, 0.5)
        for i in range(n_items):
            for r in range(reps):
                x = rng.uniform(-1, 1)
                cond = "A" if (s + i + r) % 2 == 0 else "B"
                y = 5 + 0.8 * x + (0.5 if cond == "A" else -0.5) + u0 + u1 * x + 0.3 * i + rng.normal()
                rows.append({"subject": f"S{s:02d}", "item": f"I{i}", "cond": cond, "x": x, "y": y})
    return pd.DataFrame(rows)


def _X(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(float),
                            np.where(frame["cond"] == "A", 1.0, -1.0)])


def _model(frame: pd.DataFrame, random: list, reml: bool = True) -> MixedModel:
    blocks = build_blocks(frame, parse_random(random, "subject"), {"cond": ["A", "B"]})
    return MixedModel(_X(frame), frame["y"].to_numpy(float), blocks, reml=reml)


def _dense_deviance(model: MixedModel, X: np.ndarray, y: np.ndarray, theta: np.ndarray, sigma: float) -> float:
    """-2 log-likelihood (REML or ML), straight from V = sigma^2 (Z L L'Z' + I):
    log|V| [+ log|X'V^-1 X|] + r'V^-1 r + dof log(2 pi), with beta the GLS
    estimate. No penalized least squares, no Cholesky of the model."""
    n, p = X.shape
    Z = np.zeros((n, model.q))
    for b in model.blocks:
        for j in range(b.p):
            Z[np.arange(n), b.q0 + b.codes * b.p + j] = b.Xk[:, j]
    Lam = np.zeros((model.q, model.q))
    for b in model.blocks:
        T = model.template(b, theta)
        for lv in range(b.n_levels):
            s = b.q0 + lv * b.p
            Lam[s:s + b.p, s:s + b.p] = T
    V = sigma ** 2 * (Z @ Lam @ Lam.T @ Z.T + np.eye(n))
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
    r = y - X @ beta
    out = np.linalg.slogdet(V)[1] + float(r @ Vi @ r)
    if model.reml:
        return out + np.linalg.slogdet(XtViX)[1] + (n - p) * np.log(2 * np.pi)
    return out + n * np.log(2 * np.pi)


STRUCTURES = {
    "intercept": [{"group": "subject", "terms": ["(Intercept)"]}],
    "slope": [{"group": "subject", "terms": ["(Intercept)", "x"]}],
    "uncorrelated": [{"group": "subject", "terms": ["(Intercept)", "x"], "correlated": False}],
    "factor_slope": [{"group": "subject", "terms": ["(Intercept)", "cond"]}],
    "crossed": [{"group": "subject"}, {"group": "item"}],
    "interaction": [{"group": "subject"}, {"group": ["subject", "item"]}],
}


@pytest.mark.parametrize("name", list(STRUCTURES))
@pytest.mark.parametrize("reml", [True, False])
def test_unprofiled_criterion_equals_the_dense_likelihood(name, reml):
    """deviance_at(theta, sigma) is -2 log L computed the textbook way, at an
    arbitrary point (not the optimum: the identity holds everywhere)."""
    frame = _frame()
    model = _model(frame, STRUCTURES[name], reml)
    rng = np.random.default_rng(1)
    theta = np.where(model.theta_lower == 0, rng.uniform(0.2, 1.5, model.n_theta), rng.uniform(-0.5, 0.5, model.n_theta))
    sigma = 0.9
    ours = model.deviance_at(np.r_[theta, sigma])
    want = _dense_deviance(model, _X(frame), frame["y"].to_numpy(float), theta, sigma)
    assert ours == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_profiled_criterion_is_the_minimum_over_sigma(name):
    """deviance(theta) = deviance_at(theta, sigma_hat) with sigma_hat^2 = r2 / dof,
    and any other sigma does worse."""
    frame = _frame()
    model = _model(frame, STRUCTURES[name])
    theta = np.where(model.theta_lower == 0, 0.7, 0.2)
    sigma_hat = np.sqrt(model.parts(theta)["r2"] / model.dof)
    assert model.deviance(theta) == pytest.approx(model.deviance_at(np.r_[theta, sigma_hat]), rel=1e-12)
    assert model.deviance_at(np.r_[theta, sigma_hat * 1.05]) > model.deviance(theta)


def test_one_random_intercept_equals_the_closed_form():
    """The former _RandomIntercept criterion, re-derived here:
    sum log(1 + n_i theta^2) + log|X'WX| + (n - p)(1 + log(2 pi q / (n - p))),
    W = I - c_i J per subject, c_i = theta^2 / (1 + n_i theta^2)."""
    frame = _frame()
    X, y = _X(frame), frame["y"].to_numpy(float)
    model = _model(frame, STRUCTURES["intercept"])
    codes = pd.factorize(frame["subject"], sort=True)[0]
    n, p = X.shape
    for theta in [0.0, 0.3, 1.0, 4.0]:
        A = np.zeros((p, p)); Xty = np.zeros(p); q = 0.0
        for s in np.unique(codes):
            Xi, yi = X[codes == s], y[codes == s]
            c = theta ** 2 / (1 + len(yi) * theta ** 2)
            W = np.eye(len(yi)) - c * np.ones((len(yi), len(yi)))
            A += Xi.T @ W @ Xi; Xty += Xi.T @ W @ yi; q += float(yi @ W @ yi)
        beta = np.linalg.solve(A, Xty)
        q -= float(Xty @ beta)
        ni = np.bincount(codes)
        want = (np.log1p(ni * theta ** 2).sum() + np.linalg.slogdet(A)[1]
                + (n - p) * (1 + np.log(2 * np.pi * q / (n - p))))
        assert model.deviance(np.array([theta])) == pytest.approx(want, rel=1e-10), f"theta={theta}"


def test_uncorrelated_equals_correlated_with_zero_covariance():
    """(1 + x || subject) at (a, b) is (1 + x | subject) at (a, 0, b)."""
    frame = _frame()
    unc = _model(frame, STRUCTURES["uncorrelated"])
    cor = _model(frame, STRUCTURES["slope"])
    for a, b in [(0.5, 0.3), (1.2, 0.0), (0.0, 0.8)]:
        assert unc.deviance(np.array([a, b])) == pytest.approx(cor.deviance(np.array([a, 0.0, b])), rel=1e-12)


def test_theta_layout_follows_lme4():
    """Column-major lower triangle per block; lower bounds 0 on the diagonal;
    lme4's names; || splits into one block per column."""
    frame = _frame()
    m = _model(frame, STRUCTURES["slope"])
    assert m.theta_names == ["subject.(Intercept)", "subject.x.(Intercept)", "subject.x"]
    assert m.theta_lower.tolist() == [0.0, -np.inf, 0.0]
    m = _model(frame, STRUCTURES["uncorrelated"])
    assert m.theta_names == ["subject.(Intercept)", "subject.x"]
    assert [b.columns for b in m.blocks] == [["(Intercept)"], ["x"]]
    m = _model(frame, STRUCTURES["factor_slope"])
    assert m.theta_names == ["subject.(Intercept)", "subject.cond1.(Intercept)", "subject.cond1"]


def test_blocks_reorder_by_decreasing_levels_only_when_needed():
    """lme4's convention: reorder only if a later block has MORE levels."""
    frame = _frame(n_subjects=4, n_items=9)
    m = _model(frame, [{"group": "subject"}, {"group": "item"}])
    assert [b.label for b in m.blocks] == ["item", "subject"]
    frame = _frame(n_subjects=9, n_items=4)
    m = _model(frame, [{"group": "subject"}, {"group": "item"}])
    assert [b.label for b in m.blocks] == ["subject", "item"]
    m = _model(_frame(), STRUCTURES["interaction"])
    assert [b.label for b in m.blocks] == ["subject:item", "subject"]


def test_random_formula_spelling():
    terms = parse_random(STRUCTURES["uncorrelated"] + STRUCTURES["interaction"][1:], "subject")
    assert random_formula(terms) == "(1 + x || subject) + (1 | subject:item)"
    assert parse_random(None, "pid") == [RandomTerm(("pid",))]


def test_uncorrelated_factor_slope_is_refused():
    with pytest.raises(ValueError, match="uncorrelated"):
        _model(_frame(), [{"group": "subject", "terms": ["(Intercept)", "cond"], "correlated": False}])


def test_a_text_slope_that_is_not_a_model_factor_is_refused():
    frame = _frame().assign(label=lambda d: d["cond"] + "!")
    blocks_for = lambda: build_blocks(frame, parse_random([{"group": "subject", "terms": ["label"]}], "subject"), {})  # noqa: E731
    with pytest.raises(ValueError, match="neither numeric nor a model factor"):
        blocks_for()


def test_fit_recovers_a_random_slope():
    """Large simulated data: the estimated SDs land near the truth (intercept
    1.0, slope 0.5, residual 1.0) and the fit is not singular."""
    frame = _frame(n_subjects=150, n_items=4, reps=3, seed=3)
    rec = lmm(frame, "y", within=["cond"], subject="subject", covariates=["x"],
              random=[{"group": "subject", "terms": ["(Intercept)", "x"]}])
    comps = {(c["group"], c["var1"], c["var2"]): c for c in rec["details"]["variance_components"]}
    assert comps[("subject", "(Intercept)", None)]["sdcor"] == pytest.approx(1.0, rel=0.25)
    assert comps[("subject", "x", None)]["sdcor"] == pytest.approx(0.5, rel=0.4)
    assert comps[("Residual", None, None)]["sdcor"] == pytest.approx(1.0, rel=0.1)
    assert "singular_fit" not in {w["code"] for w in rec["warnings"]}
    assert rec["details"]["formula"] == "y ~ cond * x + (1 + x | subject)"
    assert rec["details"]["term_tests"] == "type3"
    assert {c["name"] for c in rec["details"]["coefficients"]} == {"(Intercept)", "cond1", "x", "cond1:x"}


def test_a_zero_slope_variance_is_a_singular_fit():
    """Every subject has the same slope (1.0): the slope SD is at or near the
    boundary, and the singular flag agrees with theta either way (lme4's
    rule: a diagonal theta below 1e-4)."""
    rng = np.random.default_rng(5)
    rows = []
    for s in range(20):
        u = rng.normal(0, 1)
        for _ in range(8):
            x = rng.uniform(-1, 1)
            rows.append({"subject": f"S{s:02d}", "x": x, "y": 2 + x + u + rng.normal(0, 0.2)})
    frame = pd.DataFrame(rows)
    rec = lmm(frame, "y", subject="subject", covariates=["x"],
              random=[{"group": "subject", "terms": ["(Intercept)", "x"]}])
    theta = dict(zip(rec["details"]["theta_names"], rec["details"]["theta"]))
    if theta["subject.x"] < 1e-4:
        assert "singular_fit" in {w["code"] for w in rec["warnings"]}
    else:  # the noise can give a small positive slope variance; then no flag
        assert "singular_fit" not in {w["code"] for w in rec["warnings"]}


def test_intercept_only_fixed_part_is_allowed():
    frame = _frame()
    rec = lmm(frame, "y", subject="subject", random=[{"group": "subject"}, {"group": "item"}])
    assert rec["effects"] == []
    assert [c["name"] for c in rec["details"]["coefficients"]] == ["(Intercept)"]
    assert rec["details"]["formula"] == "y ~ 1 + (1 | subject) + (1 | item)"


def test_a_rank_deficient_design_is_refused():
    frame = _frame().assign(x2=lambda d: 2 * d["x"] + 1)
    with pytest.raises(ValueError, match="rank deficient"):
        lmm(frame, "y", subject="subject", covariates=["x", "x2"])


@pytest.mark.parametrize("name", list(STRUCTURES))
def test_whitened_theta_maps_to_the_same_covariance(name):
    """The optimizer's theta (centred, scaled columns) maps to lme4's theta
    with the same random-effect covariance, and back exactly."""
    frame = _frame().assign(x=lambda d: d["x"] * 10 + 60)  # uncentred, so W is not the identity
    model = _model(frame, STRUCTURES[name])
    rng = np.random.default_rng(3)
    tw = np.where(model.theta_lower == 0, rng.uniform(0.2, 2.0, model.n_theta), rng.normal(0, 1, model.n_theta))
    raw = model._recast(tw, "raw")
    assert np.all(raw[model.theta_lower == 0] >= 0)
    for b in model.blocks:
        Tw, Traw = model.template(b, tw), model.template(b, raw)
        np.testing.assert_allclose(Traw @ Traw.T, b.Winv @ Tw @ Tw.T @ b.Winv.T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model._recast(raw, "whitened"), tw, rtol=1e-9, atol=1e-12)


def test_a_singular_whitened_theta_stays_singular():
    """A zero variance in the whitened columns is a rank-1 covariance in
    lme4's theta too (a zero diagonal theta: lme4's isSingular)."""
    frame = _frame().assign(x=lambda d: d["x"] * 10 + 60)
    model = _model(frame, STRUCTURES["slope"])
    raw = model._recast(np.array([1.0, 0.4, 0.0]), "raw")
    assert raw[0] > 0 and abs(raw[2]) < 1e-12


def test_an_uncentred_covariate_reaches_the_same_optimum():
    """Shifting x by 60 only reparametrizes y ~ x + cond + (1 + x | subject):
    the REML criterion and the slope variance must not move. Raw-theta
    L-BFGS-B stopped short here before the whitening (RIS scenarios,
    2026-10-20)."""
    frame = _frame(n_subjects=15, seed=4)
    shifted = frame.assign(x=frame["x"] + 60)
    fits = [_model(f, STRUCTURES["slope"]).fit() for f in (frame, shifted)]
    assert fits[1].criterion == pytest.approx(fits[0].criterion, abs=1e-6)
    slope = [next(r["vcov"] for r in f.model.variance_components(f.theta, f.sigma2)
                  if r["var1"] == "x" and r["var2"] is None) for f in fits]
    assert slope[1] == pytest.approx(slope[0], rel=1e-4)
