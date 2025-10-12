from typing import Union
from pathlib import Path

import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats

from .utils.summary_stats import calculate_summary_statistics
from .utils.load_data import load_data_from_path
from .utils.test_assumptions import test_normality_assumption, test_variance_homogeneity_assumption
from .utils.save_stats import dict_to_pdf

def anova1way(data: Union[Path, str, pd.DataFrame], group_column: str, data_column: str):
    """
    Perform one-way ANOVA.

    Parameters:
    data : file path, str, or pd.DataFrame
        The input data containing the groups and values.
    group_column : str
        The name of the column containing group labels.
    data_column : str
        The name of the column containing data values.

    Returns:
    result: dict
        A dictionary containing:        
        "F" : float
            The computed F-statistic.
        "p" : float
            The associated p-value, rounded to four decimal places.
    """

    result = {}

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # Fit the one-way ANOVA model using statsmodels
    formula = f"{data_column} ~ C({group_column})"
    model = ols(formula, data=data).fit()
    
    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Extract F-statistic and p-value
    F = anova_table.loc[f"C({group_column})", "F"]
    p = anova_table.loc[f"C({group_column})", "PR(>F)"]

    # Calculate degrees of freedom
    df_between = len(data[group_column].unique()) - 1
    df_within = len(data) - len(data[group_column].unique())
        
    summary_stats = calculate_summary_statistics(data, group_column, data_column)

    normality_result = test_normality_assumption(model)
    homogeneity_variances_result = test_variance_homogeneity_assumption(data, group_column, data_column)

    # Store results in the dictionary
    result["F"] = F
    result["p"] = round(p, 4)
    result["df_between"] = df_between
    result["df_within"] = df_within
    result["summary_stats"] = summary_stats    
    result["normality_test"] = normality_result
    result["homogeneity_of_variance_test"] = homogeneity_variances_result   

    dict_to_pdf(result, filename='anova1way_results.pdf')     

    return result


def anova2way(data: Union[Path, str, pd.DataFrame], group_column1: str, group_column2: str, data_column: str):
    """
    Perform two-way ANOVA.

    Parameters:
    data : file path, str, or pd.DataFrame
        The input data containing the groups and values.
    group_column1 : str
        The name of the first column containing group labels.
    group_column2 : str
        The name of the second column containing group labels.
    data_column : str
        The name of the column containing data values.

    Returns:
    result: dict
        A dictionary containing:
        "F1" : float
            The computed F-statistic for the first factor.
        "p1" : float
            The associated p-value for the first factor, rounded to four decimal places.
        "F2" : float
            The computed F-statistic for the second factor.
        "p2" : float
            The associated p-value for the second factor, rounded to four decimal places.
        "F_interaction" : float
            The computed F-statistic for the interaction between factors.
        "p_interaction" : float
            The associated p-value for the interaction, rounded to four decimal places.
    """

    result = {}

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # Fit the model with interaction
    formula = f"{data_column} ~ C({group_column1}) + C({group_column2}) + C({group_column1}):C({group_column2})"
    model = ols(formula, data=data).fit()
    
    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)    
        
    summary_stats_group1 = calculate_summary_statistics(data, group_column1, data_column) 
    summary_stats_group2 = calculate_summary_statistics(data, group_column2, data_column)
    # Prepare the dataframe to calculate the summary statistics for the interaction effect
    interaction_column_name = f"{group_column1}_{group_column2}"
    data[interaction_column_name] = data[group_column1].astype(str) + '_' + data[group_column2].astype(str)
    summary_stats_interaction = calculate_summary_statistics(data, interaction_column_name, data_column)

    # Store results in the dictionary
    result["main_effects"] = {}
    result["main_effects"][group_column1] = {}
    result["main_effects"][group_column2] = {}
    result["interaction"] = {}

    result["main_effects"][group_column1]["F"] = anova_table.loc[f"C({group_column1})", "F"]
    result["main_effects"][group_column1]["p"] = round(anova_table.loc[f"C({group_column1})", "PR(>F)"], 4)
    result["main_effects"][group_column2]["F"] = anova_table.loc[f"C({group_column2})", "F"]
    result["main_effects"][group_column2]["p"] = round(anova_table.loc[f"C({group_column2})", "PR(>F)"], 4)
    
    interaction_key = f"C({group_column1}):C({group_column2})"
    result["interaction"]["F"] = anova_table.loc[interaction_key, "F"]
    result["interaction"]["p"] = round(anova_table.loc[interaction_key, "PR(>F)"], 4)
    result[f"summary_stats_{group_column1}"] = summary_stats_group1
    result[f"summary_stats_{group_column2}"] = summary_stats_group2
    result["summary_stats_interaction"] = summary_stats_interaction

    normality_result = test_normality_assumption(model)
    homogeneity_variances_result = test_variance_homogeneity_assumption(data, [group_column1, group_column2], data_column)

    result["normality_test"] = normality_result
    result["homogeneity_of_variance_test"] = homogeneity_variances_result

    return result