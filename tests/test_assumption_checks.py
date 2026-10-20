"""csvstats.core.assumption_checks: the evidence behind a test recommendation."""

import numpy as np
import pandas as pd

from csvstats.core import assumption_checks


def _within(skewed: bool):
    rng = np.random.default_rng(1)
    rows = []
    for s in range(20):
        base = rng.normal(0, 1)
        for level in ("a", "b", "c"):
            noise = rng.exponential(3) ** 2 if skewed else rng.normal(0, 1)
            rows.append({"subject": f"S{s}", "c": level, "y": base + noise})
    return pd.DataFrame(rows)


def test_normal_residuals_pass_and_skewed_fail():
    ok = assumption_checks(_within(False), "y", within=["c"], subject="subject")["assumptions"]
    bad = assumption_checks(_within(True), "y", within=["c"], subject="subject")["assumptions"]
    assert ok[0]["name"] == "normality" and ok[0]["violated"] is False
    assert bad[0]["violated"] is True


def test_levene_only_with_a_between_factor():
    data = _within(False)
    assert [c["name"] for c in assumption_checks(data, "y", within=["c"], subject="subject")["assumptions"]] == ["normality"]
    names = [c["name"] for c in assumption_checks(data, "y", between=["c"])["assumptions"]]
    assert names == ["normality", "homogeneity_of_variance"]


def test_incomplete_subjects_are_not_dropped():
    data = _within(False)
    data = data[~((data.subject == "S3") & (data.c == "b"))]
    out = assumption_checks(data, "y", within=["c"], subject="subject")
    assert out["n"]["total"] == len(data) and out["n"]["excluded_detail"] == []
