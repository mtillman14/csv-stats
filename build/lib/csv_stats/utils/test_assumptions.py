from typing import Union

from scipy import stats
import pandas as pd

def test_normality_assumption(model) -> dict:
    """Test the normality assumption of residuals using the Shapiro-Wilk test.

    Args:
        model (_type_): The fitted statistical model from statsmodels.

    Returns:
        dict: The result of the Shapiro-Wilk test containing the test statistic and p-value.
    """

    # Test for normality
    residuals = model.resid
    normality_test_statistic, p_value_normality = stats.shapiro(residuals)

    normality_result = {}
    normality_result["shapiro_wilk"] = {"test_statistic": normality_test_statistic, "p_value": p_value_normality}

    if p_value_normality < 0.05:
        print(f"Warning: The Shapiro-Wilk test for normality returned a p-value of {p_value_normality:.4f}. A p-value less than 0.05 suggests that the residuals may not be normally distributed. Consider using a non-parametric test if normality is violated.")
    
    return normality_result


def test_variance_homogeneity_assumption(data: pd.DataFrame, group_columns: Union[str, list], data_column: str) -> dict:
    """Test the homogeneity of variances using Levene's test. For 2 or more factors, uses the interaction of all factors.

    Args:
        data (pd.DataFrame): The input data containing the groups and values.
        group_columns (Union[str, list]): The name(s) of the column(s) containing group labels.
        data_column (str): The name of the column containing data values.

    Returns:
        dict: The result of Levene's test containing the test statistic and p-value.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]    
    
    # Test for homogeneity of variances
    data['combined_group'] = data[group_columns].astype(str).agg('_'.join, axis=1)
    grouped_data = [group[data_column].values for name, group in data.groupby('combined_group')]
    levene_statistic, p_value_levene = stats.levene(*grouped_data)

    homogeneity_variances_result = {}
    homogeneity_variances_result["levene"] = {"test_statistic": levene_statistic, "p_value": p_value_levene}
    
    if p_value_levene < 0.05:
        print(f"Warning: The Levene's test for homogeneity of variances returned a p-value of {p_value_levene:.4f}. A p-value less than 0.05 suggests that the variances across groups may not be equal. Consider using a non-parametric test if homogeneity of variances is violated.")

    return homogeneity_variances_result