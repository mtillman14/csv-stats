from typing import Callable
from typing import Union
from pathlib import Path

import pandas as pd

def _run_all_columns(test_to_run: Callable, data: pd.DataFrame, group_column: str, repeated_measures_column: str, filename: Union[str, Path]):
    """Helper function to loop through all data columns if `data_column == "_"` is True"""
    results = {}
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    filename = str(filename)
    for col in numeric_cols:
        try:
            filename_formatted = filename.format(data_column=col)
        except:
            # There is nothing to format in the string
            filename_formatted = filename
        results[col] = test_to_run(data, group_column, col, repeated_measures_column=repeated_measures_column, filename=filename_formatted)
    return results