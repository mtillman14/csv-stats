from typing import Union
from pathlib import Path

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pandas as pd

from .utils.summary_stats import calculate_summary_statistics
from .utils.load_data import load_data_from_path
from .utils.test_assumptions import test_normality_assumption, test_variance_homogeneity_assumption, test_sphericity_assumption
from .utils.save_stats import dict_to_pdf

def anova1way(data: Union[Path, str, pd.DataFrame], group_column: str, data_column: str, repeated_measures_column: str = "", filename: Union[str, None] = 'anova1way_results.pdf') -> dict:
    """
    Perform one-way ANOVA.

    Parameters:
    data : file path, str, or pd.DataFrame
        The input data containing the groups and values.
    group_column : str
        The name of the column containing group labels.
    data_column : str
        The name of the column containing data values.
    repeated_measures_column : str, optional
        The name of the column containing the repeated measures identifiers. Defaults to "".
    filename : str, optional
        The filename to save the results to. Defaults to 'anova1way_results.pdf'. If None, results are not saved to a file.

    Returns:
    result: dict
        A dictionary containing:        
        "F" : float
            The computed F-statistic.
        "p" : float
            The associated p-value, rounded to four decimal places.
    """

    if repeated_measures_column is None:
        repeated_measures_column = ""

    # Define a boolean flag for repeated measures ANOVA
    is_repeated_measures = repeated_measures_column == ""

    result = {}
    result["group_column"] = group_column
    result["data_column"] = data_column
    result["repeated_measures_column"] = repeated_measures_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)    
    
    # Perform ANOVA
    formula = f"{data_column} ~ C({group_column})"
    anova_result = _perform_anova(data, formula, group_column, data_column, repeated_measures_column, is_repeated_measures)
    
    # Extract F-statistic and p-value
    F = anova_result["anova_table"].loc[f"C({group_column})", "F"]
    p = anova_result["anova_table"].loc[f"C({group_column})", "PR(>F)"]

    # Calculate degrees of freedom
    df_between = len(data[group_column].unique()) - 1
    df_within = len(data) - len(data[group_column].unique())
        
    summary_stats = calculate_summary_statistics(data, group_column, data_column)    

    # Store results in the dictionary
    result["F"] = F
    result["p"] = round(p, 4)
    result["df_between"] = df_between
    result["df_within"] = df_within
    result["summary_stats"] = summary_stats    
    result["normality_test"] = anova_result["normality_test"]
    result["homogeneity_of_variance_test"] = anova_result["homogeneity_of_variance_test"]   
    result["sphericity_test"] = anova_result["sphericity_test"]

    if filename is not None:
        dict_to_pdf(result, filename=filename)     

    return result


def anova2way(data: Union[Path, str, pd.DataFrame], group_column1: str, group_column2: str, data_column: str, repeated_measures_column: str = "", filename: Union[str, None] = 'anova2way_results.pdf') -> dict:
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
    repeated_measures_column : str, optional
        The name of the column containing the repeated measures identifiers. Defaults to "".
    filename : str, optional
        The filename to save the results to. Defaults to 'anova2way_results.pdf'. If None, results are not saved to a file.

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

    if repeated_measures_column is None:
        repeated_measures_column = ""

    # Define a boolean flag for repeated measures ANOVA
    is_repeated_measures = repeated_measures_column == ""

    result = {}
    result["group_column1"] = group_column1
    result["group_column2"] = group_column2    
    result["data_column"] = data_column
    result["repeated_measures_column"] = repeated_measures_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # Fit the model with interaction
    formula = f"{data_column} ~ C({group_column1}) + C({group_column2}) + C({group_column1}):C({group_column2})"
    anova_result = _perform_anova(data, formula, [group_column1, group_column2], data_column, repeated_measures_column, is_repeated_measures)

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

    anova_table = anova_result["anova_table"]
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

    result["normality_test"] = anova_result["normality_test"]
    result["homogeneity_of_variance_test"] = anova_result["homogeneity_of_variance_test"]
    result["sphericity_test"] = anova_result["sphericity_test"]

    if filename is not None:
        dict_to_pdf(result, filename='anova2way_results.pdf')  

    return result


def anova3way(data: Union[Path, str, pd.DataFrame], group_column1: str, group_column2: str, group_column3: str, data_column: str, repeated_measures_column: str, filename: Union[str, None] = 'anova3way_results.pdf') -> dict:
    """
    Perform three-way ANOVA.

    Parameters:
    data : file path, str, or pd.DataFrame
        The input data containing the groups and values.
    group_column1 : str
        The name of the first column containing group labels.
    group_column2 : str
        The name of the second column containing group labels.
    group_column3 : str
        The name of the third column containing group labels.
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
        "F3" : float
            The computed F-statistic for the third factor.
        "p3" : float
            The associated p-value for the third factor, rounded to four decimal places.
        "F_interaction" : float
            The computed F-statistic for the interaction between factors.
        "p_interaction" : float
            The associated p-value for the interaction, rounded to four decimal places.
    """

    result = {}
    result["group_column1"] = group_column1
    result["group_column2"] = group_column2
    result["group_column3"] = group_column3
    result["data_column"] = data_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # Perform ANOVA
    formula = f"{data_column} ~ C({group_column1}) * C({group_column2}) * C({group_column3})"
    anova_result = _perform_anova(data, formula, [group_column1, group_column2, group_column3], data_column, repeated_measures_column, False) 
        
    summary_stats_group1 = calculate_summary_statistics(data, group_column1, data_column) 
    summary_stats_group2 = calculate_summary_statistics(data, group_column2, data_column)
    summary_stats_group3 = calculate_summary_statistics(data, group_column3, data_column)
    # Prepare the dataframe to calculate the summary statistics for the interaction effect
    interaction_column_name = f"{group_column1}_{group_column2}+{group_column3}"
    data[interaction_column_name] = data[group_column1].astype(str) + '_' + data[group_column2].astype(str) + '_' + data[group_column3].astype(str)
    summary_stats_interaction = calculate_summary_statistics(data, interaction_column_name, data_column)

    # Store results in the dictionary
    result["main_effects"] = {}
    result["main_effects"][group_column1] = {}
    result["main_effects"][group_column2] = {}
    result["main_effects"][group_column3] = {}
    result["interaction"] = {}

    anova_table = anova_result["anova_table"]
    result["main_effects"][group_column1]["F"] = anova_table.loc[f"C({group_column1})", "F"]
    result["main_effects"][group_column1]["p"] = round(anova_table.loc[f"C({group_column1})", "PR(>F)"], 4)
    result["main_effects"][group_column2]["F"] = anova_table.loc[f"C({group_column2})", "F"]
    result["main_effects"][group_column2]["p"] = round(anova_table.loc[f"C({group_column2})", "PR(>F)"], 4)
    result["main_effects"][group_column3]["F"] = anova_table.loc[f"C({group_column3})", "F"]
    result["main_effects"][group_column3]["p"] = round(anova_table.loc[f"C({group_column3})", "PR(>F)"], 4)
    
    interaction_key = f"C({group_column1}):C({group_column2}):C({group_column3})"
    result["interaction"]["F"] = anova_table.loc[interaction_key, "F"]
    result["interaction"]["p"] = round(anova_table.loc[interaction_key, "PR(>F)"], 4)
    result[f"summary_stats_{group_column1}"] = summary_stats_group1
    result[f"summary_stats_{group_column2}"] = summary_stats_group2
    result[f"summary_stats_{group_column3}"] = summary_stats_group3
    result["summary_stats_interaction"] = summary_stats_interaction    

    result["normality_test"] = anova_result["normality_test"]
    result["homogeneity_of_variance_test"] = anova_result["homogeneity_of_variance_test"]
    result["sphericity_test"] = anova_result["sphericity_test"]

    if filename is not None:
        dict_to_pdf(result, filename=filename)  

    return result


def _perform_anova(data: pd.DataFrame, formula: str, group_column: Union[list, str], data_column: str, repeated_measures_column: str, is_repeated_measures: bool) -> dict:
    """Perform the actual ANOVA computation. Works for all types of ANOVA.

    Args:
        data (pd.DataFrame): The input data
        group_column (str): The name of the column containing group labels.
        data_column (str): The name of the column containing data values.
        repeated_measures_column (str): The name of the column containing the repeated measures identifiers.
        is_repeated_measures (bool): The boolean flag indicating if this is a repeated measures ANOVA.

    Returns:
        dict: The ANOVA results including F-statistic and p-value.
    """
    result = {}
    if not is_repeated_measures:
        # Fit the one-way ANOVA model using statsmodels        
        model = ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        homogeneity_variances_result = test_variance_homogeneity_assumption(data, group_column, data_column)        
        mauchly_test_result = "Not applicable"
    else:
        model = AnovaRM(data, 
                        depvar=data_column, 
                        subject=repeated_measures_column, 
                        within=[group_column]).fit()
        anova_table = model.anova_table
        mauchly_test_result = test_sphericity_assumption(data, group_column, repeated_measures_column, data_column)
        homogeneity_variances_result = "Not applicable"        

    result["model"] = model
    result["anova_table"] = anova_table
    result["homogeneity_of_variance_test"] = homogeneity_variances_result
    result["sphericity_test"] = mauchly_test_result

    normality_result = test_normality_assumption(model)
    result["normality_test"] = normality_result

    return result