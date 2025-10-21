from typing import Callable
import inspect

import pandas as pd

def _run_all_columns(test_to_run: Callable, data: pd.DataFrame, group_column: str, filename: str, **optional_params):
    """Helper function to loop through all data columns if `data_column == "_"` is True"""
    results = {}
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if filename is not None:
        filename = str(filename)

    # Get the function signature once
    sig = inspect.signature(test_to_run)
    param_names = set(sig.parameters.keys())

    for col in numeric_cols:
        try:
            filename_formatted = filename.format(data_column=col)
        except:
            # There is nothing to format in the string
            filename_formatted = filename

        # Build kwargs with only the parameters the function accepts
        kwargs = {'filename': filename_formatted}
        for key, value in optional_params.items():
            if key in param_names:
                kwargs[key] = value

        results[col] = test_to_run(data, group_column, col, **kwargs)
    return results