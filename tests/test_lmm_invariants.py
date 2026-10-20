"""Properties csvstats.lmm must have, checked with NO R fixtures.

Stage 4 of .claude/plan-lmm-scenario-generator.md (scistack repo). These
catch bugs a fixture comparison can miss: anything that depends on row
order, subject labels, the units of y or level order is wrong whatever R
says. The balanced-design checks cross two independent csv-stats code paths:
a random-intercept LMM on balanced, complete data must reproduce the
classical (repeated-measures / mixed) ANOVA F and df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csvstats.core import anova
from csvstats.lmm import lmm

#: Same data, different bookkeeping: only floating point may differ.
SAME_RTOL = 1e-7
#: Satterthwaite df use numerical derivatives (relative steps); as the
#: reference tests.
DF_RTOL = 1e-3


def _data(n: int = 14, seed: int = 7, drop: float = 0.15, subject_sd: float = 1.5) -> pd.DataFrame:
    """B2 x W3, random intercept per subject; ``drop`` of the rows removed at
    random (every subject keeps >= 2), so the data are unbalanced."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal(0.0, subject_sd)
        group = "ctl" if i % 2 == 0 else "trt"
        for k, time in enumerate(["t1", "t2", "t3"]):
            y = 10 + 0.5 * k + (0.4 * k if group == "trt" else 0.0) + u + rng.normal()
            rows.append({"subject": f"S{i:02d}", "group": group, "time": time, "y": y})
    frame = pd.DataFrame(rows)
    if drop:
        cut = rng.random(len(frame)) < drop
        keep = ~cut | frame.groupby("subject").cumcount().lt(2).to_numpy()
        frame = frame[keep].reset_index(drop=True)
    return frame


def _fit(frame: pd.DataFrame, **kw) -> dict:
    kw.setdefault("levels", {"group": ["ctl", "trt"], "time": ["t1", "t2", "t3"]})
    return lmm(frame, "y", between=["group"], within=["time"], subject="subject", **kw)


def _assert_same_inference(a: dict, b: dict, df_rtol: float = SAME_RTOL) -> None:
    ea = {e["term"]: e for e in a["effects"]}
    eb = {e["term"]: e for e in b["effects"]}
    assert set(ea) == set(eb)
    for term in ea:
        assert eb[term]["statistic"] == pytest.approx(ea[term]["statistic"], rel=SAME_RTOL), f"{term} F"
        assert eb[term]["df"][0] == ea[term]["df"][0], f"{term} NumDF"
        assert eb[term]["df"][1] == pytest.approx(ea[term]["df"][1], rel=df_rtol), f"{term} DenDF"
        assert eb[term]["p"] == pytest.approx(ea[term]["p"], rel=1e-5, abs=1e-12), f"{term} p"


def _coefs(record: dict) -> dict:
    return {c["name"]: c for c in record["details"]["coefficients"]}


def test_row_order_changes_nothing():
    frame = _data()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a, b = _fit(frame), _fit(shuffled)
    _assert_same_inference(a, b)
    ca, cb = _coefs(a), _coefs(b)
    for name in ca:
        assert cb[name]["estimate"] == pytest.approx(ca[name]["estimate"], rel=SAME_RTOL, abs=1e-10), name
        assert cb[name]["se"] == pytest.approx(ca[name]["se"], rel=SAME_RTOL), name
    assert b["details"]["log_likelihood"] == pytest.approx(a["details"]["log_likelihood"], rel=1e-10)


def test_subject_labels_change_nothing():
    """Renamed subjects sort in the opposite order, so their codes differ."""
    frame = _data()
    renamed = frame.assign(subject=frame["subject"].map(lambda s: f"P{99 - int(s[1:]):02d}"))
    a, b = _fit(frame), _fit(renamed)
    _assert_same_inference(a, b)
    assert b["details"]["subject_intercept_variance"] == pytest.approx(
        a["details"]["subject_intercept_variance"], rel=SAME_RTOL)


@pytest.mark.parametrize("shift, factor", [(3.0, 2.5), (-50.0, 0.01), (1e4, 100.0), (2.0, -0.5)])
def test_affine_y_scales_estimates_and_leaves_inference(shift, factor):
    """``y -> shift + factor * y``: estimates scale by factor (the intercept
    also shifts), SEs by |factor|, t flips with the sign; df, F and p are
    unchanged; the REML log-likelihood moves by -(n - p) log|factor|.

    The intercept's t is NOT invariant: the shift moves its estimate but not
    its SE (first run, 2026-10-19). It is checked as estimate / SE instead."""
    frame = _data()
    moved = frame.assign(y=shift + factor * frame["y"])
    a, b = _fit(frame), _fit(moved)
    _assert_same_inference(a, b, df_rtol=DF_RTOL)
    ca, cb = _coefs(a), _coefs(b)
    for name in ca:
        want = shift + factor * ca[name]["estimate"] if name == "(Intercept)" else factor * ca[name]["estimate"]
        assert cb[name]["estimate"] == pytest.approx(want, rel=1e-6, abs=1e-9 * abs(factor)), f"{name} estimate"
        assert cb[name]["se"] == pytest.approx(abs(factor) * ca[name]["se"], rel=1e-6), f"{name} SE"
        want_t = (cb[name]["estimate"] / cb[name]["se"] if name == "(Intercept)"
                  else np.sign(factor) * ca[name]["t"])
        assert cb[name]["t"] == pytest.approx(want_t, rel=1e-6, abs=1e-9), f"{name} t"
        assert cb[name]["df"] == pytest.approx(ca[name]["df"], rel=DF_RTOL), f"{name} df"
    n, p = len(frame), len(ca)
    assert b["details"]["log_likelihood"] == pytest.approx(
        a["details"]["log_likelihood"] - (n - p) * np.log(abs(factor)), abs=1e-6)


def test_level_order_leaves_every_f_unchanged():
    """Reordering levels changes the coefficients (sum coding drops the LAST
    level), not the hypotheses: every term's F and NumDF are the same. The
    Fai–Cornelius DenDF depends on the contrast basis (as lmerTest), so it
    is not compared."""
    frame = _data()
    a = _fit(frame)
    b = _fit(frame, levels={"group": ["trt", "ctl"], "time": ["t3", "t1", "t2"]})
    ea = {e["term"]: e for e in a["effects"]}
    eb = {e["term"]: e for e in b["effects"]}
    for term in ea:
        assert eb[term]["statistic"] == pytest.approx(ea[term]["statistic"], rel=1e-6), f"{term} F"
        assert eb[term]["df"][0] == ea[term]["df"][0], f"{term} NumDF"
    # One-df terms have one basis up to sign, so they match in full.
    assert eb["group"]["df"][1] == pytest.approx(ea["group"]["df"][1], rel=DF_RTOL)


def _uncorrected(effect: dict) -> list[float]:
    """An ANOVA effect's df before any Greenhouse–Geisser correction."""
    correction = effect.get("sphericity_correction")
    if correction:
        return [d / correction["epsilon"] for d in effect["df"]]
    return list(effect["df"])


def _assert_lmm_equals_anova(lmm_rec: dict, anova_rec: dict) -> None:
    assert not any(w["code"] == "singular_fit" for w in lmm_rec["warnings"]), "needs a positive subject variance"
    ours = {e["term"]: e for e in lmm_rec["effects"]}
    classical = {e["term"]: e for e in anova_rec["effects"]}
    assert set(ours) == set(classical)
    for term, c in classical.items():
        df1, df2 = _uncorrected(c)
        assert ours[term]["statistic"] == pytest.approx(c["statistic"], rel=1e-6), f"{term} F"
        assert ours[term]["df"][0] == pytest.approx(df1), f"{term} NumDF"
        assert ours[term]["df"][1] == pytest.approx(df2, rel=DF_RTOL), f"{term} DenDF"


def test_balanced_within_design_reproduces_repeated_measures_anova():
    """Balanced, complete W3 with a clear subject effect: compound symmetry
    holds in the model, so the LMM's F(2, 2(n-1)) is the RM ANOVA's."""
    frame = _data(n=12, drop=0.0, subject_sd=2.0)
    frame = frame[frame["group"] == "ctl"].reset_index(drop=True)
    levels = {"time": ["t1", "t2", "t3"]}
    ours = lmm(frame, "y", within=["time"], subject="subject", levels=levels)
    classical = anova(frame, "y", within=["time"], subject="subject", levels=levels)
    _assert_lmm_equals_anova(ours, classical)
    n = frame["subject"].nunique()
    assert {e["term"]: e for e in ours["effects"]}["time"]["df"][1] == pytest.approx(2 * (n - 1), rel=DF_RTOL)


def test_balanced_mixed_design_reproduces_mixed_anova():
    """Balanced, complete B2 x W3: the between term's F(1, n-2) and the
    within terms' F(., 2(n-2)) are the mixed ANOVA's."""
    frame = _data(n=14, drop=0.0, subject_sd=2.0)
    levels = {"group": ["ctl", "trt"], "time": ["t1", "t2", "t3"]}
    ours = lmm(frame, "y", between=["group"], within=["time"], subject="subject", levels=levels)
    classical = anova(frame, "y", between=["group"], within=["time"], subject="subject", levels=levels)
    _assert_lmm_equals_anova(ours, classical)
