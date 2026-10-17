import pandas as pd

from ..descriptive import summarize


def _summary(values: pd.Series) -> dict:
    """One group's statistics in this module's key names, from `descriptive`."""
    s = summarize(values)
    return {
        "mean": s["mean"],
        "std_dev": s["sd"],
        # Non-missing values only; missing ones are in n_missing. (Before
        # 0.2.0 this was len(group), which counted NaN, so n and the CI
        # disagreed with the mean and SD whenever a value was missing.)
        "count": s["n"],
        "variance": s["variance"],
        "n_missing": s["n_missing"],
        "quartiles": {0.25: s["q1"], 0.5: s["median"], 0.75: s["q3"]},
        "min": s["min"],
        "max": s["max"],
        "ci_95_low": s["ci_low"],
        "ci_95_high": s["ci_high"],
    }


def calculate_summary_statistics(data: pd.DataFrame, group_column: str, data_column: str) -> dict:
    """
    Calculate summary statistics for each group in the data. If there is only one group, grouped statistics will still be calculated.

    Every number comes from `csvstats.descriptive` (the one definition of each
    formula: NaN skipped, SD with n - 1, t-based 95% CI).

    Parameters:
    data : pd.DataFrame
        The input data containing the groups and values.
    group_column : str
        The name of the column containing group labels.
    data_column : str
        The name of the column containing data values.

    Returns:
    summary_stats: dict
        {"grouped": {statistic: {group: value}}, "overall": {statistic: value}}
        with statistics "mean", "std_dev", "count" (non-missing), "variance",
        "n_missing", "quartiles", "min", "max", "ci_95_low", "ci_95_high".
    """
    per_group = {name: _summary(group) for name, group in data.groupby(group_column)[data_column]}
    keys = next(iter(per_group.values())).keys() if per_group else _summary(pd.Series(dtype=float)).keys()
    grouped = {key: {name: stats[key] for name, stats in per_group.items()} for key in keys}
    return {"grouped": grouped, "overall": _summary(data[data_column])}
