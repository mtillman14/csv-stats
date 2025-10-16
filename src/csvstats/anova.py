from typing import Union
from pathlib import Path

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pandas as pd
import pingouin as pg

from .utils.summary_stats import calculate_summary_statistics
from .utils.load_data import load_data_from_path
from .utils.test_assumptions import test_normality_assumption, test_variance_homogeneity_assumption, test_sphericity_assumption
from .utils.save_stats import dict_to_pdf
from .utils.run_all_columns import _run_all_columns

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
    is_repeated_measures = repeated_measures_column != ""

    result = {}
    result["date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_repeated_measures:
        result["test"] = "One-way Repeated Measures ANOVA"
    else:
        result["test"] = "One-way Independent Samples ANOVA"    
    result["group_column"] = group_column
    result["data_column"] = data_column
    result["repeated_measures_column"] = repeated_measures_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data) 

    # "_" is the special character indicating to loop through all columns
    if data_column == "_":
        results = _run_all_columns(anova1way, data, group_column, repeated_measures_column, filename)
        return results   
    
    # Perform ANOVA
    formula = f"{data_column} ~ C({group_column})"
    anova_result = _perform_anova(data, formula, group_column, data_column, repeated_measures_column, is_repeated_measures)
    
    # Extract F-statistic and p-value
    if is_repeated_measures:
        # Index into the output of pg.rm_anova()
        F = anova_result["anova_table"]['F'].iloc[0]
        p = anova_result["anova_table"]['p-unc'].iloc[0]
    else:
        # Index into the output of sm.stats.anova_lm()
        F = anova_result["anova_table"].loc[f"{group_column}", "F Value"]
        p = anova_result["anova_table"].loc[f"{group_column}", "Pr > F"]

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

    if p >= 0.05:
        result["post_hoc"] = 'Not applicable'
    else:
        # Perform post-hoc testing
        posthoc_result = _perform_posthoc_tests(data, group_column, data_column, repeated_measures_column, is_repeated_measures)
        result['post_hoc'] = posthoc_result

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
    is_repeated_measures = repeated_measures_column != ""

    result = {}
    result["group_column1"] = group_column1
    result["group_column2"] = group_column2    
    result["data_column"] = data_column
    result["repeated_measures_column"] = repeated_measures_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # "_" is the special character indicating to loop through all columns
    if data_column == "_":
        results = _run_all_columns(anova2way, data, group_column, repeated_measures_column, filename)
        return results   

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
    # Extract F-statistic and p-value
    if is_repeated_measures:
        # Index into the output of pg.rm_anova()
        F1 = anova_result["anova_table"]['F'].iloc[0]
        p1 = anova_result["anova_table"]['p-unc'].iloc[0]
        F2 = anova_result["anova_table"]['F'].iloc[1]
        p2 = anova_result["anova_table"]['p-unc'].iloc[1]
    else:
        # Index into the output of sm.stats.anova_lm()
        F1 = anova_result["anova_table"].loc[f"{group_column1}", "F Value"]
        p1 = anova_result["anova_table"].loc[f"{group_column1}", "Pr > F"]
        F2 = anova_result["anova_table"].loc[f"{group_column2}", "F Value"]
        p2 = anova_result["anova_table"].loc[f"{group_column2}", "Pr > F"]

    result["main_effects"][group_column1]["F"] = F1
    result["main_effects"][group_column1]["p"] = round(p1, 4)
    result["main_effects"][group_column2]["F"] = F2
    result["main_effects"][group_column2]["p"] = round(p2, 4)
    
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

    if repeated_measures_column is None:
        repeated_measures_column = ""

    # Define a boolean flag for repeated measures ANOVA
    is_repeated_measures = repeated_measures_column != ""

    result = {}
    result["group_column1"] = group_column1
    result["group_column2"] = group_column2
    result["group_column3"] = group_column3
    result["data_column"] = data_column
    result["repeated_measures_column"] = repeated_measures_column

    # Load the data as a pandas DataFrame
    data = load_data_from_path(data)

    # "_" is the special character indicating to loop through all columns
    if data_column == "_":
        results = _run_all_columns(anova3way, data, group_column, repeated_measures_column, filename)
        return results   

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
    # Extract F-statistic and p-value
    if is_repeated_measures:
        # Index into the output of pg.rm_anova()
        F1 = anova_result["anova_table"]['F'].iloc[0]
        p1 = anova_result["anova_table"]['p-unc'].iloc[0]
        F2 = anova_result["anova_table"]['F'].iloc[1]
        p2 = anova_result["anova_table"]['p-unc'].iloc[1]
        F3 = anova_result["anova_table"]['F'].iloc[2]
        p3 = anova_result["anova_table"]['p-unc'].iloc[2]
    else:
        # Index into the output of sm.stats.anova_lm()
        F1 = anova_result["anova_table"].loc[f"{group_column1}", "F Value"]
        p1 = anova_result["anova_table"].loc[f"{group_column1}", "Pr > F"]
        F2 = anova_result["anova_table"].loc[f"{group_column2}", "F Value"]
        p2 = anova_result["anova_table"].loc[f"{group_column2}", "Pr > F"]
        F3 = anova_result["anova_table"].loc[f"{group_column3}", "F Value"]
        p3 = anova_result["anova_table"].loc[f"{group_column3}", "Pr > F"]
    result["main_effects"][group_column1]["F"] = F1
    result["main_effects"][group_column1]["p"] = round(p1, 4)
    result["main_effects"][group_column2]["F"] = F2
    result["main_effects"][group_column2]["p"] = round(p2, 4)
    result["main_effects"][group_column3]["F"] = F3
    result["main_effects"][group_column3]["p"] = round(p3, 4)
    
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
        residuals = model.resid
    else:
        anova_table = pg.rm_anova(
            data=data,
            dv=data_column,
            within=group_column,
            subject=repeated_measures_column,
            detailed=True
        )

        # Calculate residuals for repeated measures
        grand_mean = data[data_column].mean()
        
        # Subject effects: deviation of each subject's overall mean from grand mean
        subject_means = data.groupby(repeated_measures_column)[data_column].transform('mean')
        subject_effects = subject_means - grand_mean

        # Group effects:        
        group_means = data.groupby(group_column)[data_column].transform('mean')
        group_effects = group_means - grand_mean
        predicted = grand_mean + subject_effects + group_effects
        residuals = data[data_column] - predicted

        # Perform Mauchly test of sphericity
        mauchly_test_result = test_sphericity_assumption(data, group_column, repeated_measures_column, data_column)
        homogeneity_variances_result = "Not applicable"

    result["anova_table"] = anova_table
    result["homogeneity_of_variance_test"] = homogeneity_variances_result
    result["sphericity_test"] = mauchly_test_result

    normality_result = test_normality_assumption(residuals)
    result["normality_test"] = normality_result

    return result

def _perform_posthoc_tests(data: pd.DataFrame, group_column: Union[list, str], data_column: str, repeated_measures_column: str, is_repeated_measures: bool, correction_method: str = "holm") -> dict:
    
    # Ensure group_column is a list
    if isinstance(group_column, str):
        group_column = [group_column]

    # Map correction method names to pingouin-compatible names
    correction_map = {
        "bonferroni": "bonf",
        "holm": "holm",
        "fdr_bh": "fdr_bh",
        "none": "none"
    }

    padjust = correction_map[correction_method]

    if is_repeated_measures:
        # Perform repeated measures post-hoc tests
        posthoc_result = pg.pairwise_tests(
            data=data,
            dv=data_column,
            within=group_column,
            subject=repeated_measures_column,
            parametric=True,
            padjust=padjust,
            effsize="cohen"
        )
    else:
        # Perform independent samples post-hoc tests
        if len(group_column) == 1:
            # Single factor
            posthoc_result = pg.pairwise_tests(
                data=data,
                dv=data_column,
                between=group_column[0],
                parametric=True,
                padjust=padjust,
                effsize='cohen'
            )
        else:
            # Multiple factors - perform tests for each factor
            posthoc_result = pg.pairwise_tests(
                data=data,
                dv=data_column,
                between=group_column,
                parametric=True,
                padjust=padjust,
                effsize='cohen'
            )

    # Extract significant pairs (p-adj < 0.05)
    significant_pairs = []
    if 'p-corr' in posthoc_result.columns:
        sig_rows = posthoc_result[posthoc_result['p-corr'] < 0.05]
        for _, row in sig_rows.iterrows():
            if 'A' in row and 'B' in row:
                significant_pairs.append((row['A'], row['B']))

    # Convert posthoc_result DataFrame to the desired dictionary format
    posthoc_dict = {}
    for _, row in posthoc_result.iterrows():
        # Create the key from A and B columns
        key = f"{row['A']} - {row['B']}"
        
        # Create a dictionary with all other columns (excluding A and B)
        value_dict = {col: row[col] for col in posthoc_result.columns if col not in ['A', 'B']}
        
        posthoc_dict[key] = value_dict

    return {
        'correction_method': correction_method,
        'significant_pairs': significant_pairs,
        'posthoc_results': posthoc_dict              
    }