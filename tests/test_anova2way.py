
from csvstats.anova import anova2way
import numpy as np
import pandas as pd


def test_anova2way_independent():
    """Test two-way independent samples ANOVA with known data."""

    np.random.seed(42)
    n = 20

    # Create data with two factors: treatment (A/B) and sex (M/F)
    # Factor 1 (treatment) has a significant effect, factor 2 (sex) does not
    data = pd.DataFrame({
        'treatment': np.repeat(['A', 'B'], n * 2),
        'sex': np.tile(np.repeat(['M', 'F'], n), 2),
        'score': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=n),   # A, M
            np.random.normal(loc=5.0, scale=1.0, size=n),   # A, F
            np.random.normal(loc=8.0, scale=1.0, size=n),   # B, M
            np.random.normal(loc=8.0, scale=1.0, size=n),   # B, F
        ])
    })

    result = anova2way(data, group_column1='treatment', group_column2='sex',
                       data_column='score', filename=None)

    # Verify result structure
    assert result is not None
    assert "test" in result
    assert result["test"] == "Two-way Independent Samples ANOVA"
    assert "date" in result
    assert "main_effects" in result
    assert "interaction" in result
    assert "treatment" in result["main_effects"]
    assert "sex" in result["main_effects"]

    # Treatment should be significant (large mean difference)
    assert "F" in result["main_effects"]["treatment"]
    assert "p" in result["main_effects"]["treatment"]
    assert result["main_effects"]["treatment"]["p"] < 0.05

    # Sex should not be significant (no mean difference)
    assert "F" in result["main_effects"]["sex"]
    assert "p" in result["main_effects"]["sex"]

    # Interaction
    assert "F" in result["interaction"]
    assert "p" in result["interaction"]

    # Summary statistics
    assert "summary_statistics_treatment" in result
    assert "summary_statistics_sex" in result
    assert "summary_statistics_interaction" in result

    # Assumption tests
    assert "normality_test" in result
    assert "homogeneity_of_variance_test" in result
    assert "sphericity_test" in result

    # Post-hoc tests
    assert "post_hoc" in result
    assert "treatment" in result["post_hoc"]
    # Treatment is significant, so post-hoc should be a dict with results
    assert result["post_hoc"]["treatment"] != "Not applicable"


def test_anova2way_skip_same_column():
    """Test that anova2way returns None when data_column matches a group column."""

    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'value': [7, 8, 9]
    })

    result = anova2way(data, group_column1='A', group_column2='B',
                       data_column='A', filename=None)
    assert result is None


def test_anova2way_repeated_measures():
    """Test two-way repeated measures ANOVA."""

    np.random.seed(7)
    n_subjects = 15

    # Each subject is measured under all 4 conditions (2 x 2 within-subject design)
    subjects = np.tile(np.arange(n_subjects), 4)
    factor1 = np.repeat(['low', 'high'], n_subjects * 2)
    factor2 = np.tile(np.repeat(['pre', 'post'], n_subjects), 2)

    # factor1 has a large effect, factor2 has a moderate effect
    scores = np.concatenate([
        np.random.normal(loc=10, scale=2, size=n_subjects),  # low, pre
        np.random.normal(loc=12, scale=2, size=n_subjects),  # low, post
        np.random.normal(loc=16, scale=2, size=n_subjects),  # high, pre
        np.random.normal(loc=18, scale=2, size=n_subjects),  # high, post
    ])

    data = pd.DataFrame({
        'subject': subjects,
        'dose': factor1,
        'time': factor2,
        'score': scores,
    })

    result = anova2way(data, group_column1='dose', group_column2='time',
                       data_column='score', repeated_measures_column='subject',
                       filename=None)

    assert result is not None
    assert result["test"] == "Two-way Repeated Measures ANOVA"
    assert "main_effects" in result
    assert "dose" in result["main_effects"]
    assert "time" in result["main_effects"]
    assert "interaction" in result

    # Both main effects should be significant given the large mean differences
    assert result["main_effects"]["dose"]["p"] < 0.05
    assert result["main_effects"]["time"]["p"] < 0.05

    # Assumption tests
    assert "normality_test" in result
    assert result["sphericity_test"] != "Not applicable"
    assert result["homogeneity_of_variance_test"] == "Not applicable"

    # Post-hoc
    assert "post_hoc" in result


if __name__ == "__main__":
    test_anova2way_independent()
    test_anova2way_skip_same_column()
    test_anova2way_repeated_measures()
    print("All tests passed.")
