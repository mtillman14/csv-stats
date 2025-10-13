
import numpy as np
import pandas as pd

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

    # Store overall statistics
    summary_stats_overall = {}
    summary_stats_overall["mean"] = data[data_column].mean()
    summary_stats_overall["std_dev"] = data[data_column].std()
    summary_stats_overall["count"] = len(data)
    summary_stats_overall["n_missing"] = data[data_column].isnull().sum()
    summary_stats_overall["variance"] = data[data_column].var()
    summary_stats_overall["quartiles"] = data[data_column].quantile([0.25,0.5, 0.75]).to_dict()
    summary_stats_overall["min"] = data[data_column].min()
    summary_stats_overall["max"] = data[data_column].max()

    summary_stats["grouped"] = summary_stats_grouped
    summary_stats["overall"] = summary_stats_overall

    return summary_stats