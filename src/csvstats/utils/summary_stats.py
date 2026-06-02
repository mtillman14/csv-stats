
import numpy as np
import pandas as pd
from scipy import stats

def calculate_summary_statistics(data: pd.DataFrame, group_column: str, data_column: str) -> dict:
    """
    Calculate summary statistics for each group in the data. If there is only one group, grouped statistics will still be calculated.

    Parameters:
    data : pd.DataFrame
        The input data containing the groups and values.
    group_column : str
        The name of the column containing group labels.
    data_column : str
        The name of the column containing data values.

    Returns:
    summary_stats: dict
        A dictionary where keys are group names and values are dictionaries of summary statistics:
        "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    """
    summary_stats = {}
    grouped = data.groupby(group_column)[data_column]

    # Calculate summary statistics for each group
    means = {name: group.mean() for name, group in grouped}    
    std_devs = {name: group.std() for name, group in grouped}
    counts = {name: len(group) for name, group in grouped}
    variances = {name: group.var() for name, group in grouped}
    n_missing = {name: group.isnull().sum() for name, group in grouped}   
    mins = {name: group.min() for name, group in grouped}
    maxs = {name: group.max() for name, group in grouped}
            
    # Calculate quartiles
    quartiles = {name: group.quantile([0.25, 0.5, 0.75]).to_dict() for name, group in grouped}

    # Calculate 95% confidence intervals for the mean
    ci_low = {}
    ci_high = {}
    for name in means:
        n = counts[name]
        if n >= 2:
            se = std_devs[name] / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_low[name] = means[name] - t_crit * se
            ci_high[name] = means[name] + t_crit * se
        else:
            ci_low[name] = np.nan
            ci_high[name] = np.nan

    # Store group-wise statistics
    summary_stats_grouped = {}
    summary_stats_grouped["mean"] = means
    summary_stats_grouped["std_dev"] = std_devs
    summary_stats_grouped["count"] = counts
    summary_stats_grouped["variance"] = variances
    summary_stats_grouped["n_missing"] = n_missing
    summary_stats_grouped["quartiles"] = quartiles
    summary_stats_grouped["min"] = mins
    summary_stats_grouped["max"] = maxs
    summary_stats_grouped["ci_95_low"] = ci_low
    summary_stats_grouped["ci_95_high"] = ci_high

    # Store overall statistics
    overall_n = len(data)
    overall_mean = data[data_column].mean()
    overall_std = data[data_column].std()
    if overall_n >= 2:
        overall_se = overall_std / np.sqrt(overall_n)
        overall_t_crit = stats.t.ppf(0.975, df=overall_n - 1)
        overall_ci_low = overall_mean - overall_t_crit * overall_se
        overall_ci_high = overall_mean + overall_t_crit * overall_se
    else:
        overall_ci_low = np.nan
        overall_ci_high = np.nan

    summary_stats_overall = {}
    summary_stats_overall["mean"] = overall_mean
    summary_stats_overall["std_dev"] = overall_std
    summary_stats_overall["count"] = overall_n
    summary_stats_overall["n_missing"] = data[data_column].isnull().sum()
    summary_stats_overall["variance"] = data[data_column].var()
    summary_stats_overall["quartiles"] = data[data_column].quantile([0.25,0.5, 0.75]).to_dict()
    summary_stats_overall["min"] = data[data_column].min()
    summary_stats_overall["max"] = data[data_column].max()
    summary_stats_overall["ci_95_low"] = overall_ci_low
    summary_stats_overall["ci_95_high"] = overall_ci_high

    summary_stats["grouped"] = summary_stats_grouped
    summary_stats["overall"] = summary_stats_overall

    return summary_stats