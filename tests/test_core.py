"""csvstats.core: the C3 core record, checked against scipy and hand values."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from csvstats.core import (
    NotSupported,
    anova,
    correct_pvalues,
    friedman,
    kruskal,
    rank_test,
    ttest,
)


def _two_groups():
    return pd.DataFrame({
        "g": ["A"] * 5 + ["B"] * 6,
        "y": [5.1, 4.8, 6.0, 5.5, 5.9, 3.9, 4.2, 4.0, 4.6, 3.5, 4.4],
    })


def _paired():
    rows = []
    for s, (pre, post) in enumerate([(5.0, 6.1), (4.2, 4.9), (6.3, 6.8), (5.5, 6.9), (4.9, 5.0), (5.8, 6.6)]):
        rows += [(f"S{s}", "pre", pre), (f"S{s}", "post", post)]
    return pd.DataFrame(rows, columns=["subject", "time", "y"])


# --- correction -----------------------------------------------------------


def test_correct_pvalues_reference_values():
    assert correct_pvalues([0.01, 0.04, 0.03], "holm") == pytest.approx([0.03, 0.06, 0.06])
    assert correct_pvalues([0.01, 0.04, 0.03], "fdr_bh") == pytest.approx([0.03, 0.04, 0.04])
    assert correct_pvalues([0.01, 0.4], "bonferroni") == pytest.approx([0.02, 0.8])
    assert correct_pvalues([0.2, None, 0.6], "bonferroni") == [0.4, None, 1.0]
    with pytest.raises(ValueError):
        correct_pvalues([0.1], "tukey")


# --- t-tests --------------------------------------------------------------


def test_welch_matches_scipy_and_states_the_direction():
    data = _two_groups()
    out = ttest(data, "y", between=["g"])
    (effect,) = out["effects"]
    a = data[data.g == "A"].y.to_numpy()
    b = data[data.g == "B"].y.to_numpy()
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert out["test"] == "ttest_ind_welch"
    assert (effect["a"], effect["b"]) == ("A", "B")
    assert effect["statistic"] == pytest.approx(ref.statistic)
    assert effect["p"] == pytest.approx(ref.pvalue)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    assert effect["df"] == [pytest.approx(df)]
    est = effect["estimate"]
    assert est["value"] == pytest.approx(a.mean() - b.mean())
    half = stats.t.ppf(0.975, df) * math.sqrt(va + vb)
    assert est["ci"] == pytest.approx([est["value"] - half, est["value"] + half])


def test_levels_reverse_the_sign():
    out = ttest(_two_groups(), "y", between=["g"], levels={"g": ["B", "A"]})
    (effect,) = out["effects"]
    assert (effect["a"], effect["b"]) == ("B", "A")
    assert effect["estimate"]["value"] < 0


def test_student_effect_sizes_and_their_ci():
    data = _two_groups()
    out = ttest(data, "y", between=["g"], equal_var=True)
    (effect,) = out["effects"]
    a = data[data.g == "A"].y.to_numpy()
    b = data[data.g == "B"].y.to_numpy()
    assert effect["statistic"] == pytest.approx(stats.ttest_ind(a, b).statistic)
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    sizes = effect["effect_sizes"]
    assert sizes["d_s"]["value"] == pytest.approx(d)
    assert sizes["g_s"]["value"] < sizes["d_s"]["value"], "J < 1 shrinks g"
    lo, hi = sizes["d_s"]["ci"]
    assert lo < d < hi
    # The noncentral-t bounds put the observed t at the 97.5% / 2.5% points.
    scale = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    df = len(a) + len(b) - 2
    assert stats.nct.cdf(d * scale, df, lo * scale) == pytest.approx(0.975, abs=1e-6)
    assert stats.nct.cdf(d * scale, df, hi * scale) == pytest.approx(0.025, abs=1e-6)
    assert effect["effect_size"]["name"] == "g_s"


def test_paired_t_matches_scipy_and_d_z():
    data = _paired()
    out = ttest(data, "y", within=["time"], subject="subject", levels={"time": ["post", "pre"]})
    (effect,) = out["effects"]
    wide = data.pivot(index="subject", columns="time", values="y")
    ref = stats.ttest_rel(wide["post"], wide["pre"])
    assert out["test"] == "ttest_paired"
    assert effect["statistic"] == pytest.approx(ref.statistic)
    assert effect["p"] == pytest.approx(ref.pvalue)
    assert effect["df"] == [5.0]
    d = (wide["post"] - wide["pre"]).to_numpy()
    assert effect["estimate"]["sd_of_differences"] == pytest.approx(d.std(ddof=1))
    assert effect["effect_sizes"]["d_z"]["value"] == pytest.approx(ref.statistic / math.sqrt(6))
    assert effect["effect_size"]["name"] == "d_z"


def test_incomplete_subjects_are_dropped_and_reported():
    data = _paired()
    data = data[~((data.subject == "S0") & (data.time == "post"))]
    out = ttest(data, "y", within=["time"], subject="subject")
    assert out["n"]["subjects"] == 5
    assert {"subject": "S0", "reason": "missing a within-subject level"} in out["n"]["excluded_detail"]
    assert any(w["code"] == "subjects_dropped_incomplete" for w in out["warnings"])


def test_missing_values_are_reported_and_the_input_is_not_mutated():
    data = _two_groups()
    data.loc[0, "y"] = np.nan
    before = data.copy()
    out = ttest(data, "y", between=["g"])
    pd.testing.assert_frame_equal(data, before)
    assert out["n"]["total"] == 10
    assert any(w["code"] == "rows_dropped_missing" for w in out["warnings"])


def test_results_are_json_and_deterministic():
    one = ttest(_two_groups(), "y", between=["g"])
    two = ttest(_two_groups(), "y", between=["g"])
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert "date" not in one


def test_bad_input_raises_naming_the_column():
    with pytest.raises(ValueError, match="nope"):
        ttest(_two_groups(), "nope", between=["g"])
    with pytest.raises(NotSupported):
        ttest(_two_groups(), "y", between=["g"], within=["g"])


# --- rank tests -----------------------------------------------------------


def test_mann_whitney_matches_scipy_with_hodges_lehmann():
    data = _two_groups()
    out = rank_test(data, "y", between=["g"])
    (effect,) = out["effects"]
    a = data[data.g == "A"].y.to_numpy()
    b = data[data.g == "B"].y.to_numpy()
    ref = stats.mannwhitneyu(a, b)
    assert effect["statistic"] == pytest.approx(ref.statistic)
    assert effect["p"] == pytest.approx(ref.pvalue)
    assert effect["estimate"]["value"] == pytest.approx(np.median((a[:, None] - b[None, :]).ravel()))
    assert effect["effect_size"]["value"] == pytest.approx(2 * ref.statistic / (len(a) * len(b)) - 1)
    assert effect["effect_size"]["value"] == pytest.approx(1.0), "every A exceeds every B"


def test_wilcoxon_matches_scipy():
    data = _paired()
    out = rank_test(data, "y", within=["time"], subject="subject", levels={"time": ["post", "pre"]})
    (effect,) = out["effects"]
    wide = data.pivot(index="subject", columns="time", values="y")
    ref = stats.wilcoxon(wide["post"] - wide["pre"])
    assert effect["statistic"] == pytest.approx(ref.statistic)
    assert effect["p"] == pytest.approx(ref.pvalue)
    assert effect["effect_size"]["value"] == pytest.approx(1.0), "every difference is positive"


# --- ANOVA family ---------------------------------------------------------


def _three_groups():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "g": np.repeat(["A", "B", "C"], 8),
        "y": np.concatenate([rng.normal(m, 1, 8) for m in (5, 6, 7.5)]),
    })


def test_one_way_anova_matches_scipy():
    data = _three_groups()
    out = anova(data, "y", between=["g"])
    (effect,) = out["effects"]
    ref = stats.f_oneway(*[data[data.g == g].y for g in "ABC"])
    assert effect["statistic"] == pytest.approx(ref.statistic)
    assert effect["p"] == pytest.approx(ref.pvalue)
    assert effect["df"] == [2.0, 21.0]
    f = ref.statistic
    assert effect["effect_sizes"]["eta2_p"]["value"] == pytest.approx(2 * f / (2 * f + 21))
    assert effect["effect_sizes"]["omega2_p"]["value"] == pytest.approx(2 * (f - 1) / (2 * (f - 1) + 24))


def _rm(levels=("t1", "t2", "t3"), groups=None):
    rng = np.random.default_rng(5)
    rows = []
    for s in range(8):
        base = rng.normal(10, 2)
        for i, t in enumerate(levels):
            rows.append({"subject": f"S{s}", "time": t, "y": base + i + rng.normal(0, 0.5),
                         **({"group": "ctl" if s < 4 else "trt"} if groups else {})})
    return pd.DataFrame(rows)


def test_rm_anova_applies_greenhouse_geisser_with_three_levels():
    out = anova(_rm(), "y", within=["time"], subject="subject")
    (effect,) = out["effects"]
    assert out["test"] == "anova_rm"
    assert effect["term"] == "time"
    assert effect["sphericity_correction"]["method"] == "greenhouse_geisser"
    assert effect["effect_size"]["name"] == "eta2_g"
    assert 0 <= effect["effect_sizes"]["eta2_g"]["value"] <= effect["effect_sizes"]["eta2_p"]["value"]


def test_mixed_anova_terms_and_generalized_eta():
    out = anova(_rm(groups=True), "y", between=["group"], within=["time"], subject="subject")
    terms = [e["term"] for e in out["effects"]]
    assert terms == ["group", "time", "group:time"]
    for effect in out["effects"]:
        g = effect["effect_sizes"]["eta2_g"]["value"]
        assert 0 <= g <= effect["effect_sizes"]["eta2_p"]["value"] + 1e-12


def test_unsupported_anova_design_says_so():
    data = _rm(groups=True)
    data["side"] = "L"
    with pytest.raises(NotSupported, match="not yet in csv-stats"):
        anova(data, "y", between=["group", "side"], within=["time"], subject="subject")


def test_kruskal_and_friedman():
    data = _three_groups()
    k = kruskal(data, "y", between=["g"])
    ref = stats.kruskal(*[data[data.g == g].y for g in "ABC"])
    assert k["effects"][0]["statistic"] == pytest.approx(ref.statistic)
    f = friedman(_rm(), "y", within=["time"], subject="subject")
    wide = _rm().pivot(index="subject", columns="time", values="y")
    ref = stats.friedmanchisquare(wide.t1, wide.t2, wide.t3)
    assert f["effects"][0]["statistic"] == pytest.approx(ref.statistic)
    assert f["effects"][0]["effect_size"]["value"] == pytest.approx(ref.statistic / (8 * 2))


def test_pingouin_p_columns_are_read_under_both_spellings():
    """Regression: pingouin 0.6 renamed p-unc -> p_unc (KeyError)."""
    from csvstats.core import _p_unc

    assert _p_unc(pd.Series({"p_unc": 0.2})) == 0.2
    assert _p_unc(pd.Series({"p-unc": 0.3})) == 0.3


def _curves(paired: bool):
    rng = np.random.default_rng(11)
    rows = []
    for s in range(8):
        base = np.sin(np.linspace(0, np.pi, 101)) + rng.normal(0, 0.05, 101)
        for level, shift in (("pre", 0.0), ("post", 0.6)):
            bump = shift * np.exp(-((np.arange(101) - 50) ** 2) / 50.0)
            rows.append({"subject": f"S{s}" if paired else f"{level}{s}", "time": level,
                         "y": list(base + bump + rng.normal(0, 0.05, 101))})
    return pd.DataFrame(rows)


def test_spm_refuses_unequal_lengths_before_needing_spm1d():
    from csvstats.core import _curve_matrix

    with pytest.raises(ValueError, match="unequal length"):
        _curve_matrix([[1, 2, 3], [1, 2]], "x")


def test_spm_paired_finds_the_middle_cluster():
    pytest.importorskip("spm1d", reason="SPM tests need spm1d (pip install spm1d)")
    from csvstats.core import spm_ttest

    out = spm_ttest(_curves(True), "y", within=["time"], subject="subject", levels={"time": ["post", "pre"]})
    (effect,) = out["effects"]
    assert out["test"] == "spm_ttest_paired"
    assert effect["clusters"], "the bump at node 50 is supra-threshold"
    cluster = effect["clusters"][0]
    assert cluster["start"] < 50 < cluster["end"]
    assert effect["p"] is not None and effect["p"] < 0.05
    assert len(effect["curve"]) == 101
