"""Shared by the R-oracle LMM tests:

- test_lmm_reference.py: the curated set (tests/reference, make_lmm_reference.R)
- test_lmm_scenarios.py: the sampled set (tests/reference/scenarios,
  make_lmm_scenarios.R)

ONE owner for the tolerances, the JSON quirks and the comparisons, so the two
suites can never hold csv-stats to different standards. Not collected by
pytest (no ``test_`` prefix). Doc: docs/claude/lmm-r-oracle-testing.md in the
scistack repo.
"""

from __future__ import annotations

from typing import Callable, Mapping

import pytest

# Tolerances. Fixed effects and SEs are scaled by the data's SD, so the
# datasets around 1e-3 and 1e4 are held to the same relative standard.
EST_RTOL = 1e-5
SE_RTOL = 1e-4
VAR_RTOL = 1e-3
LOGLIK_ATOL = 1e-3
# A singular fit sits on a boundary; two optimizers approach it differently.
SINGULAR_LOOSEN = 100
DF_RTOL = 1e-3
# t and F are also compared with an ABSOLUTE floor of the same size: a
# relative difference of a statistic near 0 is meaningless (NESTED_006:
# F = 0.0034, abs diff 2e-6 was a 6e-4 relative miss; 2026-10-20).
T_RTOL = 1e-5
P_ATOL = 1e-4
F_RTOL = 1e-5
# Inference (df, t, F, p) is loosened by this on a singular fit, or where
# lme4 stopped short of the optimum.
INFERENCE_LOOSEN = 10.0
# csv-stats' log-likelihood this much HIGHER than lme4's means lme4 stopped
# before the optimum.
STOPPED_SHORT_GAP = 1e-10


def as_list(value) -> list:
    """jsonlite's auto_unbox wrote a one-factor list as a bare string and an
    empty one as null (fixtures of 2026-10-19); a list either way here."""
    if value is None:
        return []
    # A one-element NUMBER is unboxed too (covariates_at, 2026-10-20).
    return [value] if isinstance(value, (str, int, float)) else list(value)


def as_dict(value) -> dict:
    """jsonlite writes an empty named list as ``[]``."""
    return dict(value) if isinstance(value, Mapping) else {}


def stopped_short(our_loglik: float, r_loglik: float) -> bool:
    """csv-stats' REML log-likelihood is HIGHER than lme4's: lme4's optimizer
    stopped before the optimum, so its estimates (and everything computed
    from them) carry that error. W3rep_mcar15 (2026-10-19): loglik 1e-8
    apart, SEs 4e-6, so the intercept's t = 50.4 differed by 2.4e-5.
    Tolerances are relaxed ONLY for such datasets, never across the board."""
    return our_loglik - r_loglik > STOPPED_SHORT_GAP


def r_variances(ref: dict) -> tuple[float, float]:
    """lme4's ``(subject intercept variance, residual variance)``."""
    rows = ref["varcorr"]
    subject = next(r["vcov"] for r in rows if r["grp"] == "subject" and r["var1"] == "(Intercept)" and r["var2"] is None)
    residual = next(r["vcov"] for r in rows if r["grp"] == "Residual")
    return subject, residual


def rel(a, b, floor):
    return abs(a - b) / max(abs(b), floor)


def assert_fixed_effects(coefficients: list[dict], ref: dict, scale: float, loosen: float,
                         floors: Mapping[str, Mapping[str, float]] | None = None) -> None:
    """Every coefficient's estimate and SE against lme4's. ``floors``
    (``{"fixef": {name: abs}, "se": {name: abs}}``) widens the absolute
    tolerance per coefficient: the sampled set passes 10 x the spread between
    R's own optimizers there."""
    floors = floors or {}
    ours = {c["name"]: c for c in coefficients}
    assert set(ours) == {r["name"] for r in ref["fixed"]}, "same model terms and coding (contr.sum)"
    for r in ref["fixed"]:
        mine = ours[r["name"]]
        est_abs = max(EST_RTOL * loosen * scale, floors.get("fixef", {}).get(r["name"], 0.0))
        se_abs = max(SE_RTOL * loosen * scale, floors.get("se", {}).get(r["name"], 0.0))
        assert mine["estimate"] == pytest.approx(r["estimate"], rel=EST_RTOL * loosen, abs=est_abs), \
            f"{r['name']} estimate"
        assert mine["se"] == pytest.approx(r["se"], rel=SE_RTOL * loosen, abs=se_abs), _vs(f"{r['name']} SE", mine["se"], r["se"], SE_RTOL * loosen)


def assert_variance_components(details: dict, ref: dict, scale: float, loosen: float,
                               floors: tuple[float, float] = (0.0, 0.0)) -> None:
    """Subject and residual variance (random intercept) against lme4's;
    ``floors`` = absolute ``(subject, residual)`` tolerance floors."""
    subject, residual = r_variances(ref)
    var = scale ** 2
    assert details["residual_variance"] == pytest.approx(
        residual, rel=VAR_RTOL * loosen, abs=floors[1] or None)
    assert details["subject_intercept_variance"] == pytest.approx(
        subject, rel=VAR_RTOL * loosen, abs=max(VAR_RTOL * loosen * var, floors[0]))


def _component_key(group, var1, var2) -> tuple:
    """(group, {var1, var2}) for matching csv-stats' variance components with
    R's VarCorr rows. lme4 suffixes a repeated group name (``||`` makes two
    blocks of one group) with ``.1``; stripped here."""
    import re

    group = re.sub(r"\.\d+$", "", str(group))
    return group, frozenset(v for v in (var1, var2) if v is not None), var2 is None


def assert_variance_rows(components: list[dict], ref: dict, scale: float, loosen: float,
                         floors: list[float] | None = None) -> None:
    """Every variance and covariance (and the residual) against R's
    ``as.data.frame(VarCorr)`` rows, matched on (group, variables).
    Absolute tolerance VAR_RTOL x the data's variance, or ``floors[i]`` for
    R's row i (10 x the optimizer spread), whichever is larger."""
    ours = {_component_key(c["group"], c["var1"], c["var2"]): c for c in components}
    theirs = ref["varcorr"]
    assert len(ours) == len(theirs), f"{len(ours)} components vs R's {len(theirs)}"
    for i, r in enumerate(theirs):
        key = _component_key(r["grp"], r["var1"], r["var2"])
        assert key in ours, f"R's {r['grp']} {r['var1']} {r['var2']} has no csv-stats counterpart: {sorted(ours, key=str)}"
        floor = floors[i] if floors and i < len(floors) else 0.0
        tol = max(VAR_RTOL * loosen * scale ** 2, floor)
        assert ours[key]["vcov"] == pytest.approx(r["vcov"], rel=VAR_RTOL * loosen, abs=tol), \
            f"{r['grp']} {r['var1']} {r['var2']} vcov"


def _vs(label: str, ours, theirs, tol: float) -> str:
    """An assertion message with both values: a custom message replaces
    pytest.approx's own detail, so without this a failure shows no numbers."""
    try:
        r = abs(ours - theirs) / abs(theirs) if theirs else float("nan")
    except TypeError:
        r = float("nan")
    return f"{label}: csv-stats {ours!r} vs R {theirs!r} (rel diff {r:.2e}, tol {tol:.2e})"


def assert_singular_agrees(record: dict, ref: dict) -> None:
    """lme4's isSingular and csv-stats' singular_fit warning agree, both ways."""
    flagged = any(w["code"] == "singular_fit" for w in record["warnings"])
    assert flagged == bool(ref["singular"]), "lme4 singular" if ref["singular"] else "lme4 NOT singular"


def assert_coefficient_satterthwaite(record: dict, ref: dict, loosen: float) -> None:
    """Every coefficient's Satterthwaite df, t and p (lmerTest's summary()).
    A fallback to Wald is allowed only with its warning, and then the
    comparison is skipped for that coefficient, never silent."""
    ours = {c["name"]: c for c in record["details"]["coefficients"]}
    if any(c["df_method"] != "satterthwaite" for c in ours.values()):
        assert any(w["code"] == "satterthwaite_unavailable" for w in record["warnings"])
    # A fallback is a failure only where lmerTest itself produced a df.
    fallbacks = [r["name"] for r in ref["fixed"]
                 if r["df"] is not None and ours[r["name"]]["df_method"] != "satterthwaite"]
    for r in ref["fixed"]:
        mine = ours[r["name"]]
        if mine["df_method"] != "satterthwaite" or r["df"] is None:
            continue
        assert mine["df"] == pytest.approx(r["df"], rel=DF_RTOL * loosen), _vs(f"{r['name']} df", mine["df"], r["df"], DF_RTOL * loosen)
        assert mine["t"] == pytest.approx(r["t"], rel=T_RTOL * loosen, abs=T_RTOL * loosen), _vs(f"{r['name']} t", mine["t"], r["t"], T_RTOL * loosen)
        assert mine["p"] == pytest.approx(r["p"], abs=P_ATOL * loosen), _vs(f"{r['name']} p", mine["p"], r["p"], P_ATOL * loosen)
    assert not fallbacks, f"Wald fallback where lmerTest has a df: {fallbacks}"


def assert_term_f_tests(record: dict, ref: dict, loosen: float) -> None:
    """Every term's type-III F, NumDF, DenDF (Satterthwaite, Fai–Cornelius)
    and p against lmerTest's anova(type = 3). Same names ("group:time")."""
    if not isinstance(ref["anova_satterthwaite"], list):
        pytest.skip(f"lmerTest anova: {ref['anova_satterthwaite']}")
    ours = {e["term"]: e for e in record["effects"]}
    assert set(ours) == {r["term"] for r in ref["anova_satterthwaite"]}
    for r in ref["anova_satterthwaite"]:
        mine = ours[r["term"]]
        assert mine["df_method"] == "satterthwaite", f"{r['term']}: fell back to Wald ({record['warnings']})"
        assert mine["df"][0] == r["num_df"], f"{r['term']} NumDF"
        assert mine["statistic"] == pytest.approx(r["F"], rel=F_RTOL * loosen, abs=F_RTOL * loosen), _vs(f"{r['term']} F", mine["statistic"], r["F"], F_RTOL * loosen)
        assert mine["df"][1] == pytest.approx(r["den_df"], rel=DF_RTOL * loosen), _vs(f"{r['term']} DenDF", mine["df"][1], r["den_df"], DF_RTOL * loosen)
        assert mine["p"] == pytest.approx(r["p"], abs=P_ATOL * loosen), _vs(f"{r['term']} p", mine["p"], r["p"], P_ATOL * loosen)


def assert_wald_alongside(record: dict, ref: dict) -> None:
    """The Wald chi2 kept under each term's "wald" equals F * NumDF from R."""
    if not isinstance(ref["wald_terms"], list):
        pytest.skip(f"lmerTest anova: {ref['wald_terms']}")
    ours = {e["term"]: e for e in record["effects"]}
    for r in ref["wald_terms"]:
        assert ours[r["term"]]["wald"]["chi2"] == pytest.approx(r["chi2"], rel=F_RTOL * 10)


def contrast_key(row: dict, factor: str, factors: list[str]) -> tuple:
    context = tuple(sorted((f, str(row[f])) for f in factors if f != factor))
    return context, str(row["a"]), str(row["b"])


def reference_contrasts(ref: dict, which: str) -> dict[str, list[dict]]:
    """``{factor: rows}``; a factor emmeans could not do is a string (skip)."""
    return {f: rows for f, rows in as_dict(ref[which]).items() if isinstance(rows, list)}


def _ours_by_key(pairwise: dict) -> dict:
    return {(tuple(sorted(e["context"].items())), e["a"], e["b"]): e for e in pairwise["effects"]}


def assert_pairwise_contrasts(pairwise_for: Callable[[str], dict], ref: dict, scale: float, loosen: float) -> None:
    """Every full-model pairwise contrast of every factor, within every
    combination of the others, against emmeans (lmer.df = "satterthwaite"):
    estimate, SE, df, t, p. ``pairwise_for(factor)`` is csv-stats'
    ``lmm_pairwise`` record for that factor."""
    for factor, rows in reference_contrasts(ref, "contrasts_satterthwaite").items():
        ours = _ours_by_key(pairwise_for(factor))
        assert len(ours) == len(rows), f"{factor}: {len(ours)} contrasts vs emmeans {len(rows)}"
        for row in rows:
            key = contrast_key(row, factor, ref["factors"])
            mine = ours[key]
            assert mine["df_method"] == "satterthwaite", f"{factor} {key}: fell back to Wald"
            est = mine["estimate"]
            assert est["value"] == pytest.approx(row["estimate"], rel=EST_RTOL * loosen, abs=EST_RTOL * loosen * scale), \
                f"{factor} {key} estimate"
            assert est["se"] == pytest.approx(row["SE"], rel=SE_RTOL * loosen, abs=SE_RTOL * loosen * scale), \
                f"{factor} {key} SE"
            assert mine["df"][0] == pytest.approx(row["df"], rel=DF_RTOL * loosen), _vs(f"{factor} {key} df", mine["df"][0], row["df"], DF_RTOL * loosen)
            assert mine["statistic"] == pytest.approx(row["statistic"], rel=T_RTOL * loosen, abs=T_RTOL * loosen), _vs(f"{factor} {key} t", mine["statistic"], row["statistic"], T_RTOL * loosen)
            assert mine["p"] == pytest.approx(row["p.value"], abs=P_ATOL * loosen), _vs(f"{factor} {key} p", mine["p"], row["p.value"], P_ATOL * loosen)


def assert_pairwise_wald(pairwise_for: Callable[[str], dict], ref: dict, loosen: float) -> None:
    """The Wald z kept with each contrast equals emmeans' asymptotic z."""
    for factor, rows in reference_contrasts(ref, "contrasts_wald").items():
        ours = _ours_by_key(pairwise_for(factor))
        for row in rows:
            mine = ours[contrast_key(row, factor, ref["factors"])]
            assert mine["wald"]["z"] == pytest.approx(row["statistic"], rel=T_RTOL * loosen, abs=T_RTOL * loosen)
            assert mine["wald"]["p"] == pytest.approx(row["p.value"], abs=P_ATOL * loosen)
