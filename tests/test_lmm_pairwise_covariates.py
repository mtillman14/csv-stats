"""lmm_pairwise with covariates (emmeans' reference grid), checked without R.

Stage 1 of .claude/plan-lmm-covariate-contrasts.md (scistack repo). emmeans
is the oracle in test_lmm_scenarios.py; these pin what follows from the
reference grid itself:
- a contrast is the coefficient combination at the covariate's mean
- shifting the covariate leaves every contrast unchanged (the mean moves
  with it), unlike lmm()'s type-III F for the factor, which sits at 0
- ``covariates_at`` moves the estimate exactly along the interaction
"""

from __future__ import annotations

import pytest

from csvstats.lmm import lmm, lmm_pairwise
from test_lmm_type3 import _data

KW = dict(between=["group"], subject="subject", covariates=["x"])


def _coefs(data):
    return {c["name"]: c["estimate"] for c in lmm(data, "y", **KW)["details"]["coefficients"]}


def test_a_contrast_is_the_coefficients_at_the_covariate_mean():
    """Sum coding, levels ctl < trt: ctl - trt = 2 (b_group1 + b_group1:x * mean x)."""
    data = _data()
    rec = lmm_pairwise(data, "y", factor="group", **KW)
    (e,) = rec["effects"]
    assert (e["a"], e["b"]) == ("ctl", "trt")
    xbar = float(data["x"].mean())
    assert rec["details"]["covariates_at"] == {"x": {"value": pytest.approx(xbar), "source": "mean"}}
    b = _coefs(data)
    assert e["estimate"]["value"] == pytest.approx(2 * (b["group1"] + b["group1:x"] * xbar), rel=1e-8)
    assert "at x = " in rec["details"]["contrasts"] and rec["details"]["formula"].startswith("y ~ group * x")


def test_shifting_the_covariate_moves_lmm_f_but_not_the_contrasts():
    """x + 10: the grid mean shifts too, so estimate, SE and df stay; lmm()'s
    group F (at x = 0) changes. The two conventions, side by side."""
    a = lmm_pairwise(_data(), "y", factor="group", **KW)["effects"][0]
    b = lmm_pairwise(_data(offset=10.0), "y", factor="group", **KW)["effects"][0]
    assert b["estimate"]["value"] == pytest.approx(a["estimate"]["value"], rel=1e-6)
    assert b["estimate"]["se"] == pytest.approx(a["estimate"]["se"], rel=1e-6)
    assert b["df"][0] == pytest.approx(a["df"][0], rel=1e-5)
    fa, fb = (next(e for e in lmm(d, "y", **KW)["effects"] if e["term"] == "group")["statistic"]
              for d in (_data(), _data(offset=10.0)))
    assert fb != pytest.approx(fa, rel=1e-3)


def test_covariates_at_moves_the_estimate_along_the_interaction():
    data = _data()
    xbar = float(data["x"].mean())
    default = lmm_pairwise(data, "y", factor="group", **KW)["effects"][0]["estimate"]["value"]
    same = lmm_pairwise(data, "y", factor="group", covariates_at={"x": xbar}, **KW)
    assert same["effects"][0]["estimate"]["value"] == pytest.approx(default, rel=1e-10)
    assert same["details"]["covariates_at"]["x"]["source"] == "given"
    moved = lmm_pairwise(data, "y", factor="group", covariates_at={"x": xbar + 1}, **KW)
    assert moved["effects"][0]["estimate"]["value"] - default == pytest.approx(2 * _coefs(data)["group1:x"], rel=1e-8)


def test_within_factor_contrasts_carry_the_covariate_in_every_context():
    rec = lmm_pairwise(_data(), "y", between=["group"], within=["time"], subject="subject", covariates=["x"],
                       factor="time")
    assert len(rec["effects"]) == 2 * 3  # 2 group contexts x 3 time pairs
    assert all(e["df_method"] == "satterthwaite" for e in rec["effects"])
    assert set(rec["details"]["covariates_at"]) == {"x"}


def test_refusals():
    with pytest.raises(ValueError, match="covariates alone have no pairs"):
        lmm_pairwise(_data(), "y", subject="subject", covariates=["x"])
    with pytest.raises(ValueError, match="covariates_at names"):
        lmm_pairwise(_data(), "y", factor="group", covariates_at={"age": 50}, **KW)
