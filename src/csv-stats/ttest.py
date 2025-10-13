from typing import Union
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

from .anova import anova1way
from .utils.load_data import load_data_from_path
from .utils.save_stats import dict_to_pdf
from .utils.summary_stats import calculate_summary_statistics
from .utils.test_assumptions import test_normality_assumption, test_variance_homogeneity_assumption

def ttest_ind(data: Union[Path, str, pd.DataFrame], group_column: str, data_column: str, filename: Union[str, None] = 'ttest_ind_results.pdf') -> dict:
    """Conduct a two-sample t-test (independent or paired) and save results to a PDF.

    Args:
        data (Union[Path, str, pd.DataFrame]): The input data, which can be a file path, string, or DataFrame.
        group_column (str): The name of the column containing group labels. If this is a one sample t-test, this should be an empty string.
        data_column (str): The name of the column containing the data values.
        repeated_measures_column (str, optional): The name of the. Defaults to "".
        filename (str, optional): The filename to save the results to. Defaults to 'ttest_ind_results.pdf'. If None, results are not saved to a file.
    """

    result = {} # Initialize result dictionary

    # Load the data
    data = load_data_from_path(data)

    # Make sure there is a 'group_column' even for one-sample t-tests    
    if group_column is None:
        group_column = ""
    if group_column == "":
        group_column = "one_sample"
        data[group_column] = "all"

    # Store metadata
    result["group_column"] = group_column
    result["data_column"] = data_column

    groups = data[group_column].unique()
    num_groups = len(groups)
    if num_groups == 2:
        result = anova1way(data, group_column, data_column, group_column, filename)
        dict_to_pdf(result, filename=filename)
        return result
    
    if num_groups != 1:
        raise ValueError("Data must contain exactly one or two groups for a t-test.")
    
    t_stat, p_value = stats.ttest_1samp(data[data_column], popmean=0)
    result["t_statistic"] = round(t_stat, 4)
    result["p_value"] = round(p_value, 4)

    summary_stats = calculate_summary_statistics(data, group_column, data_column)
    result["summary_statistics"] = summary_stats

    # Get the model for normality assumption testing
    formula = f"{data_column} ~ C({group_column})"
    model = ols(formula, data=data).fit()

    # Test model assumptions
    normality_test_result = test_normality_assumption(model)
    homogeneity_variances_result = test_variance_homogeneity_assumption(data, group_column, data_column)
    result["normality_test"] = normality_test_result
    result["homogeneity_of_variance_test"] = homogeneity_variances_result

    if filename is not None:
        dict_to_pdf(result, filename=filename)
    return result


def ttest_dep(data: Union[Path, str, pd.DataFrame], group_column: str, data_column: str, repeated_measures_column: str, filename: Union[str, None] = 'ttest_dep_results.pdf') -> dict:
    """Conduct a paired t-test and save results to a PDF.

    Args:
        data (Union[Path, str, pd.DataFrame]): The input data, which can be a file path, string, or DataFrame.
        group_column (str): The name of the column containing group labels. If this is a one sample t-test, this should be an empty string.
        data_column (str): The name of the column containing the data values.
        repeated_measures_column (str, optional): The name of the column containing the repeated measures identifiers. Defaults to "".
        filename (str, optional): The filename to save the results to. Defaults to 'ttest_dep_results.pdf'. If None, results are not saved to a file.
    """

    # Load the data
    data = load_data_from_path(data)

    groups = data[group_column].unique()
    num_groups = len(groups)
    if num_groups != 2:
        raise ValueError("Data must contain exactly two groups for a paired t-test.")

    # Pivot the data to have one column per group
    pivot_data = data.pivot(index=repeated_measures_column, columns=group_column, values=data_column)
    delta_column_name = f"{groups[0]}_minus_{groups[1]}"
    pivot_data[delta_column_name] = pivot_data[groups[0]] - pivot_data[groups[1]]

    # Perform paired t-test on the difference column
    ttest_ind_result = ttest_ind(pivot_data, "", delta_column_name, filename=None)
    
    # Calculate and add summary statistics for the repeated measures
    # i.e. if there was a pre and post test per subject, this will provide a mean per subject
    # repeated_measures_summary = calculate_summary_statistics(data, repeated_measures_column, delta_column_name)
    # ttest_ind_result["repeated_measures_summary_statistics"] = repeated_measures_summary

    if filename is not None:
        dict_to_pdf(ttest_ind_result, filename=filename)
    return ttest_ind_result