"""csvstats.lmm: linear mixed models that use every observation (§3.5).

Terms and coefficients are Satterthwaite F / t since stage 3-4; lmm_pairwise is
still Wald (stage 5). Every effect says its ``df_method`` and Wald never
implies t/F df. Validated here on synthetic data with a known effect; parity
with R (lme4 / lmerTest) is in test_lmm_reference.py.
"""

import json

import numpy as np
import pandas as pd
import pytest

from csvstats.lmm import lmm, lmm_pairwise


def _sessions(n_subjects=16, drop=(), effect=1.0, seed=3):
    """y per subject x session (pre/post/follow), with a subject intercept,
    a true session effect, and the (subject, session) cells in ``drop`` left
    out, as when later sessions have not been recorded yet."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        intercept = rng.normal(0, 2)
        for k, session in enumerate(["pre", "post", "follow"]):
            if (f"S{s:02d}", session) in drop:
                continue
            for trial in range(2):
                rows.append({"subject": f"S{s:02d}", "session": session,
                             "y": 10 + intercept + effect * k + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_omnibus_record_is_satterthwaite_f_and_json():
    """Terms are F(q, ddf) with Satterthwaite ddf since stage 4; the Wald chi2
    of the same hypothesis travels alongside."""
    rec = lmm(_sessions(), "y", within=["session"], subject="subject")
    assert rec["test"] == "lmm"
    (effect,) = rec["effects"]
    assert effect["term"] == "session"
    assert effect["statistic_name"] == "F" and effect["df"][0] == 2.0 and effect["df"][1] > 2
    assert effect["df_method"] == "satterthwaite"
    assert effect["wald"]["df"] == 2.0 and effect["wald"]["chi2"] > 0
    assert effect["p"] < 1e-6, "a true effect of 1 SD-unit per session is detected"
    json.dumps(rec)  # C2: plain JSON
    assert rec["details"]["coding"] == "sum-to-zero"
    assert 0 < rec["details"]["icc"] < 1


def test_incomplete_subjects_are_kept_and_named_not_dropped():
    drop = {("S01", "follow"), ("S02", "post"), ("S02", "follow")}
    data = _sessions(drop=drop)
    rec = lmm(data, "y", within=["session"], subject="subject")
    assert rec["n"]["subjects"] == 16, "nobody is dropped"
    assert rec["n"]["excluded_detail"] == []
    assert rec["n"]["total"] == len(data)
    (note,) = [w for w in rec["warnings"] if w["code"] == "incomplete_within_used"]
    assert "S01" in note["message"] and "S02" in note["message"]


def test_two_factor_terms_include_the_interaction():
    data = _sessions()
    data["speed"] = np.where(data.index % 2 == 0, "slow", "fast")
    rec = lmm(data, "y", within=["session", "speed"], subject="subject")
    assert [e["term"] for e in rec["effects"]] == ["session", "speed", "session:speed"]


def test_pairwise_estimate_is_a_minus_b_and_keeps_unpaired_subjects():
    drop = {("S03", "post")}
    data = _sessions(drop=drop)
    data = data[data["session"].isin(["pre", "post"])]
    rec = lmm_pairwise(data, "y", within=["session"], subject="subject", levels={"session": ["post", "pre"]})
    (effect,) = rec["effects"]
    assert (effect["a"], effect["b"]) == ("post", "pre")
    assert effect["statistic_name"] == "z" and effect["df"] == [] and effect["df_method"] == "wald"
    est = effect["estimate"]
    assert est["value"] == pytest.approx(1.0, abs=0.3), "true post - pre = 1"
    assert est["ci"][0] < est["value"] < est["ci"][1]
    assert rec["n"]["subjects"] == 16, "S03 (pre only) is kept"


def test_pairwise_tails():
    data = _sessions()
    data = data[data["session"].isin(["pre", "post"])]
    common = dict(within=["session"], subject="subject", levels={"session": ["post", "pre"]})
    greater = lmm_pairwise(data, "y", tails="greater", **common)["effects"][0]["p"]
    less = lmm_pairwise(data, "y", tails="less", **common)["effects"][0]["p"]
    assert greater < 0.01 < less


def test_one_row_per_subject_is_refused_with_a_reason():
    data = pd.DataFrame({"subject": [f"S{i}" for i in range(8)], "g": ["a", "b"] * 4, "y": np.arange(8.0)})
    with pytest.raises(ValueError, match="not identifiable"):
        lmm(data, "y", between=["g"], subject="subject")


def test_subject_is_required():
    with pytest.raises(ValueError, match="subject="):
        lmm(_sessions(), "y", within=["session"])


def test_a_zero_subject_variance_is_a_warning_not_an_error():
    """Every subject has exactly the same mean, so the subject variance is
    zero by construction (a singular fit). Before, this test used random data
    with "no subject effect", and the estimate was 0.24 by chance."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        e = rng.normal(0, 1, 2)
        # Within-subject deviations that sum to zero: each subject's mean is 5.
        for level, dev in zip(["a", "b", "a", "b"], [e[0], -e[0], e[1], -e[1]]):
            rows.append({"subject": f"S{i}", "c": level, "y": 5.0 + dev})
    data = pd.DataFrame(rows)
    assert data.groupby("subject")["y"].mean().round(12).nunique() == 1
    rec = lmm(data, "y", within=["c"], subject="subject")
    assert rec["effects"][0]["p"] is not None, "a result, not an exception"
    codes = {w["code"] for w in rec["warnings"]}
    assert "singular_fit" in codes
    assert rec["details"]["icc"] < 1e-3


def test_version_strings_agree():
    """__init__.__version__ said 0.2.0 while pyproject said 0.2.1, so
    scidb.log reported the wrong engine version."""
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib

    import csvstats

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        assert tomllib.load(f)["project"]["version"] == csvstats.__version__


def test_zapsmall_matches_r():
    """R: zapsmall(c(1, 1e-20)) is c(1, 0); with max 123.45..., 7 significant
    digits leave 4 decimal places."""
    from csvstats.lmm import _zapsmall

    assert _zapsmall(np.array([1.0, 1e-20])).tolist() == [1.0, 0.0]
    assert _zapsmall(np.array([123.456789012, 0.1])).tolist() == [123.4568, 0.1]
    assert _zapsmall(np.zeros(2)).tolist() == [0.0, 0.0]


def test_doolittle_reproduces_the_matrix():
    """L U = x for a positive-definite x (no pivoting needed), L unit lower."""
    from csvstats.lmm import _doolittle_L

    x = np.array([[4.0, 2.0, 0.6], [2.0, 3.0, 0.5], [0.6, 0.5, 2.0]])
    L = _doolittle_L(x)
    assert np.allclose(np.diag(L), 1.0) and np.allclose(np.triu(L, 1), 0.0)
    U = np.linalg.solve(L, x)
    assert np.allclose(np.tril(U, -1), 0.0)
