
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
    assert result["main_effects"]["sex"]["p"] >= 0.05

    # Interaction
    assert "F" in result["interaction"]
    assert "p" in result["interaction"]

    # Summary statistics
    assert "summary_statistics_treatment" in result
    assert "summary_statistics_sex" in result
    assert "summary_statistics_interaction" in result

    # Assumption tests for independent samples
    assert "normality_test" in result
    assert "homogeneity_of_variance_test" in result
    assert result["homogeneity_of_variance_test"] != "Not applicable"
    assert result["sphericity_test"] == "Not applicable"

    # Post-hoc tests
    assert "post_hoc" in result
    assert "treatment" in result["post_hoc"]
    assert "sex" in result["post_hoc"]
    assert "interaction" in result["post_hoc"]
    # Treatment is significant, so post-hoc should be a dict with results
    assert result["post_hoc"]["treatment"] != "Not applicable"
    # Sex is not significant, so post-hoc should be "Not applicable"
    assert result["post_hoc"]["sex"] == "Not applicable"


def test_anova2way_independent_nonsignificant():
    """Test two-way independent samples ANOVA where no effects are significant."""

    np.random.seed(42)
    n = 20

    # All groups have the same mean -> nothing significant
    data = pd.DataFrame({
        'factor1': np.repeat(['A', 'B'], n * 2),
        'factor2': np.tile(np.repeat(['X', 'Y'], n), 2),
        'score': np.random.normal(loc=10.0, scale=1.0, size=n * 4),
    })

    result = anova2way(data, group_column1='factor1', group_column2='factor2',
                       data_column='score', filename=None)

    assert result is not None
    assert result["test"] == "Two-way Independent Samples ANOVA"

    # All post-hocs should be "Not applicable"
    assert result["post_hoc"]["factor1"] == "Not applicable"
    assert result["post_hoc"]["factor2"] == "Not applicable"
    assert result["post_hoc"]["interaction"] == "Not applicable"


def test_anova2way_skip_same_column_group1():
    """Test that anova2way returns None when data_column matches group_column1."""

    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'value': [7, 8, 9]
    })

    result = anova2way(data, group_column1='A', group_column2='B',
                       data_column='A', filename=None)
    assert result is None


def test_anova2way_skip_same_column_group2():
    """Test that anova2way returns None when data_column matches group_column2."""

    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'value': [7, 8, 9]
    })

    result = anova2way(data, group_column1='A', group_column2='B',
                       data_column='B', filename=None)
    assert result is None


def test_anova2way_skip_same_column_repeated_measures():
    """Test that anova2way returns None when data_column matches repeated_measures_column."""

    data = pd.DataFrame({
        'subject': [1, 2, 3],
        'A': ['x', 'y', 'z'],
        'B': ['a', 'b', 'c'],
        'value': [7, 8, 9]
    })

    result = anova2way(data, group_column1='A', group_column2='B',
                       data_column='subject', repeated_measures_column='subject',
                       filename=None)
    assert result is None


def test_anova2way_none_repeated_measures_column():
    """Test that passing None for repeated_measures_column is handled correctly."""

    np.random.seed(42)
    n = 20

    data = pd.DataFrame({
        'treatment': np.repeat(['A', 'B'], n * 2),
        'sex': np.tile(np.repeat(['M', 'F'], n), 2),
        'score': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
        ])
    })

    result = anova2way(data, group_column1='treatment', group_column2='sex',
                       data_column='score', repeated_measures_column=None,
                       filename=None)

    assert result is not None
    assert result["test"] == "Two-way Independent Samples ANOVA"
    assert result["repeated_measures_column"] == ""


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
    assert result["repeated_measures_column"] == "subject"
    assert "main_effects" in result
    assert "dose" in result["main_effects"]
    assert "time" in result["main_effects"]
    assert "interaction" in result

    # Both main effects should be significant given the large mean differences
    assert result["main_effects"]["dose"]["p"] < 0.05
    assert result["main_effects"]["time"]["p"] < 0.05

    # Assumption tests for repeated measures
    assert "normality_test" in result
    assert result["sphericity_test"] != "Not applicable"
    assert result["homogeneity_of_variance_test"] == "Not applicable"

    # Post-hoc for significant effects
    assert "post_hoc" in result
    assert result["post_hoc"]["dose"] != "Not applicable"
    assert result["post_hoc"]["time"] != "Not applicable"


def test_anova2way_repeated_measures_nonsignificant():
    """Test repeated measures two-way ANOVA with non-significant effects."""

    np.random.seed(88)
    n_subjects = 15

    subjects = np.tile(np.arange(n_subjects), 4)
    factor1 = np.repeat(['low', 'high'], n_subjects * 2)
    factor2 = np.tile(np.repeat(['pre', 'post'], n_subjects), 2)

    # Same mean across all conditions -> non-significant
    scores = np.random.normal(loc=10, scale=2, size=n_subjects * 4)

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

    # All post-hocs should be "Not applicable" when nothing is significant
    assert result["post_hoc"]["dose"] == "Not applicable"
    assert result["post_hoc"]["time"] == "Not applicable"
    assert result["post_hoc"]["interaction"] == "Not applicable"


def test_anova2way_all_columns():
    """Test the all-columns path (data_column == '_')."""

    np.random.seed(3)
    n = 20

    data = pd.DataFrame({
        'treatment': np.repeat(['A', 'B'], n * 2),
        'sex': np.tile(np.repeat(['M', 'F'], n), 2),
        'measure1': np.concatenate([
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=5.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
            np.random.normal(loc=8.0, scale=1.0, size=n),
        ]),
        'measure2': np.random.normal(loc=10.0, scale=1.0, size=n * 4),
    })

    results = anova2way(data, group_column1='treatment', group_column2='sex',
                        data_column='_', filename=None)

    assert results is not None
    assert isinstance(results, dict)
    assert 'measure1' in results
    assert 'measure2' in results
    assert results['measure1']['test'] == "Two-way Independent Samples ANOVA"
    assert results['measure2']['test'] == "Two-way Independent Samples ANOVA"


def test_anova2way_all_columns_with_filename_placeholder():
    """Test all-columns path with {data_column} placeholder in filename."""

    np.random.seed(3)
    n = 20

    data = pd.DataFrame({
        'treatment': np.repeat(['A', 'B'], n * 2),
        'sex': np.tile(np.repeat(['M', 'F'], n), 2),
        'measure1': np.random.normal(loc=5.0, scale=1.0, size=n * 4),
    })

    # Use a filename with {data_column} placeholder - the function should format it
    # without error. Use None-equivalent by checking no exception is raised.
    results = anova2way(data, group_column1='treatment', group_column2='sex',
                        data_column='_',
                        filename='/tmp/test_{data_column}_results.pdf')

    assert results is not None
    assert 'measure1' in results


def test_anova2way_all_columns_without_filename_placeholder():
    """Test all-columns path with a filename that has no {data_column} placeholder
    (exercises the KeyError/IndexError fallback)."""

    np.random.seed(3)
    n = 20

    data = pd.DataFrame({
        'treatment': np.repeat(['A', 'B'], n * 2),
        'sex': np.tile(np.repeat(['M', 'F'], n), 2),
        'measure1': np.random.normal(loc=5.0, scale=1.0, size=n * 4),
    })

    # Plain filename without placeholder -> triggers the except branch
    results = anova2way(data, group_column1='treatment', group_column2='sex',
                        data_column='_',
                        filename='/tmp/test_results.pdf')

    assert results is not None
    assert 'measure1' in results


def test_anova2way_interaction_posthoc():
    """Test that interaction post-hoc tests are performed when interaction is significant."""

    np.random.seed(123)
    n = 30

    # Create a crossover interaction: A+X high, A+Y low, B+X low, B+Y high
    data = pd.DataFrame({
        'factor1': np.repeat(['A', 'B'], n * 2),
        'factor2': np.tile(np.repeat(['X', 'Y'], n), 2),
        'score': np.concatenate([
            np.random.normal(loc=12.0, scale=1.0, size=n),  # A, X (high)
            np.random.normal(loc=5.0, scale=1.0, size=n),   # A, Y (low)
            np.random.normal(loc=5.0, scale=1.0, size=n),   # B, X (low)
            np.random.normal(loc=12.0, scale=1.0, size=n),  # B, Y (high)
        ])
    })

    result = anova2way(data, group_column1='factor1', group_column2='factor2',
                       data_column='score', filename=None)

    assert result is not None
    # The interaction should be significant due to the crossover pattern
    assert result["interaction"]["p"] < 0.05
    assert result["post_hoc"]["interaction"] != "Not applicable"
    assert "correction_method" in result["post_hoc"]["interaction"]
    assert "posthoc_results" in result["post_hoc"]["interaction"]


if __name__ == "__main__":
    test_anova2way_independent()
    test_anova2way_independent_nonsignificant()
    test_anova2way_skip_same_column_group1()
    test_anova2way_skip_same_column_group2()
    test_anova2way_skip_same_column_repeated_measures()
    test_anova2way_none_repeated_measures_column()
    test_anova2way_repeated_measures()
    test_anova2way_repeated_measures_nonsignificant()
    test_anova2way_all_columns()
    test_anova2way_all_columns_with_filename_placeholder()
    test_anova2way_all_columns_without_filename_placeholder()
    test_anova2way_interaction_posthoc()
    print("All tests passed.")
