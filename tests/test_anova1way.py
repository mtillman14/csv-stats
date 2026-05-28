
from csvstats.anova import anova1way
import numpy as np
import pandas as pd


def test_anova1way_independent_significant():
    """Test independent samples one-way ANOVA with a significant result."""

    np.random.seed(0)
    group_A = np.random.normal(loc=5.0, scale=1.0, size=30)
    group_B = np.random.normal(loc=8.0, scale=1.0, size=30)
    data = pd.DataFrame({
        'group': np.array(['A'] * 30 + ['B'] * 30),
        'value': np.concatenate([group_A, group_B]),
    })

    result = anova1way(data, group_column='group', data_column='value', filename=None)

    assert result is not None
    assert result["test"] == "One-way Independent Samples ANOVA"
    assert "date" in result
    assert result["group_column"] == "group"
    assert result["data_column"] == "value"
    assert result["repeated_measures_column"] == ""

    # With a large mean difference, the result should be significant
    assert "F" in result
    assert "p" in result
    assert result["p"] < 0.05

    # Degrees of freedom
    assert result["df_between"] == 1  # 2 groups - 1
    assert result["df_within"] == 58  # 60 observations - 2 groups

    # Summary statistics
    assert "summary_statistics" in result

    # Assumption tests
    assert "normality_test" in result
    assert "homogeneity_of_variance_test" in result
    assert result["homogeneity_of_variance_test"] != "Not applicable"
    assert result["sphericity_test"] == "Not applicable"

    # Post-hoc should be performed for significant result
    assert "post_hoc" in result
    assert result["post_hoc"] != "Not applicable"
    assert "correction_method" in result["post_hoc"]
    assert "significant_pairs" in result["post_hoc"]
    assert "posthoc_results" in result["post_hoc"]


def test_anova1way_independent_nonsignificant():
    """Test independent samples one-way ANOVA with a non-significant result."""

    np.random.seed(42)
    # Same mean for both groups -> non-significant
    group_A = np.random.normal(loc=5.0, scale=1.0, size=30)
    group_B = np.random.normal(loc=5.0, scale=1.0, size=30)
    data = pd.DataFrame({
        'group': np.array(['A'] * 30 + ['B'] * 30),
        'value': np.concatenate([group_A, group_B]),
    })

    result = anova1way(data, group_column='group', data_column='value', filename=None)

    assert result is not None
    assert result["test"] == "One-way Independent Samples ANOVA"
    assert result["p"] >= 0.05
    assert result["post_hoc"] == "Not applicable"


def test_anova1way_independent_three_groups():
    """Test independent samples one-way ANOVA with three groups."""

    np.random.seed(10)
    n = 25
    data = pd.DataFrame({
        'group': np.repeat(['A', 'B', 'C'], n),
        'value': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
            np.random.normal(loc=11.0, scale=1.0, size=n),
        ]),
    })

    result = anova1way(data, group_column='group', data_column='value', filename=None)

    assert result is not None
    assert result["df_between"] == 2  # 3 groups - 1
    assert result["df_within"] == 72  # 75 observations - 3 groups
    assert result["p"] < 0.05
    assert result["post_hoc"] != "Not applicable"


def test_anova1way_repeated_measures_significant():
    """Test repeated measures one-way ANOVA with a significant result."""

    np.random.seed(5)
    n_subjects = 20

    subjects = np.tile(np.arange(n_subjects), 3)
    condition = np.repeat(['low', 'medium', 'high'], n_subjects)
    scores = np.concatenate([
        np.random.normal(loc=10, scale=2, size=n_subjects),
        np.random.normal(loc=14, scale=2, size=n_subjects),
        np.random.normal(loc=18, scale=2, size=n_subjects),
    ])

    data = pd.DataFrame({
        'subject': subjects,
        'condition': condition,
        'score': scores,
    })

    result = anova1way(data, group_column='condition', data_column='score',
                       repeated_measures_column='subject', filename=None)

    assert result is not None
    assert result["test"] == "One-way Repeated Measures ANOVA"
    assert result["repeated_measures_column"] == "subject"
    assert result["p"] < 0.05

    # Assumption tests for repeated measures
    assert "normality_test" in result
    assert result["sphericity_test"] != "Not applicable"
    assert result["homogeneity_of_variance_test"] == "Not applicable"

    # Post-hoc should be performed
    assert result["post_hoc"] != "Not applicable"
    assert "correction_method" in result["post_hoc"]


def test_anova1way_repeated_measures_nonsignificant():
    """Test repeated measures one-way ANOVA with a non-significant result."""

    np.random.seed(99)
    n_subjects = 15

    subjects = np.tile(np.arange(n_subjects), 3)
    condition = np.repeat(['A', 'B', 'C'], n_subjects)
    # Same mean across conditions -> non-significant
    scores = np.concatenate([
        np.random.normal(loc=10, scale=2, size=n_subjects),
        np.random.normal(loc=10, scale=2, size=n_subjects),
        np.random.normal(loc=10, scale=2, size=n_subjects),
    ])

    data = pd.DataFrame({
        'subject': subjects,
        'condition': condition,
        'score': scores,
    })

    result = anova1way(data, group_column='condition', data_column='score',
                       repeated_measures_column='subject', filename=None)

    assert result is not None
    assert result["test"] == "One-way Repeated Measures ANOVA"
    assert result["p"] >= 0.05
    assert result["post_hoc"] == "Not applicable"


def test_anova1way_skip_same_column_as_group():
    """Test that anova1way returns None when data_column matches group_column."""

    data = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    result = anova1way(data, group_column='group', data_column='group', filename=None)
    assert result is None


def test_anova1way_skip_same_column_as_repeated_measures():
    """Test that anova1way returns None when data_column matches repeated_measures_column."""

    data = pd.DataFrame({'subject': [1, 2, 3], 'group': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    result = anova1way(data, group_column='group', data_column='subject',
                       repeated_measures_column='subject', filename=None)
    assert result is None


def test_anova1way_none_repeated_measures_column():
    """Test that passing None for repeated_measures_column is handled (converted to empty string)."""

    np.random.seed(0)
    data = pd.DataFrame({
        'group': np.array(['A'] * 30 + ['B'] * 30),
        'value': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=30),
            np.random.normal(loc=8.0, scale=1.0, size=30),
        ]),
    })

    result = anova1way(data, group_column='group', data_column='value',
                       repeated_measures_column=None, filename=None)

    assert result is not None
    assert result["test"] == "One-way Independent Samples ANOVA"
    assert result["repeated_measures_column"] == ""


def test_anova1way_all_columns():
    """Test the all-columns path (data_column == '_')."""

    np.random.seed(3)
    n = 20
    data = pd.DataFrame({
        'group': np.repeat(['A', 'B'], n),
        'measure1': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
        ]),
        'measure2': np.concatenate([
            np.random.normal(loc=10.0, scale=1.0, size=n),
            np.random.normal(loc=10.0, scale=1.0, size=n),
        ]),
    })

    results = anova1way(data, group_column='group', data_column='_', filename=None)

    assert results is not None
    assert isinstance(results, dict)
    assert 'measure1' in results
    assert 'measure2' in results
    # Each sub-result should be a full anova result dict
    assert results['measure1']['test'] == "One-way Independent Samples ANOVA"
    assert results['measure2']['test'] == "One-way Independent Samples ANOVA"


if __name__ == "__main__":
    test_anova1way_independent_significant()
    test_anova1way_independent_nonsignificant()
    test_anova1way_independent_three_groups()
    test_anova1way_repeated_measures_significant()
    test_anova1way_repeated_measures_nonsignificant()
    test_anova1way_skip_same_column_as_group()
    test_anova1way_skip_same_column_as_repeated_measures()
    test_anova1way_none_repeated_measures_column()
    test_anova1way_all_columns()
    print("All tests passed.")
