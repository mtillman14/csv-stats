from typing import Callable, Union
import inspect

import pandas as pd

def _run_all_columns(test_to_run: Callable, data: pd.DataFrame, group_column: Union[str, list], filename: str, **optional_params):
    """Helper function to loop through all data columns if `data_column == "_"` is True.

    group_column may be a single string (1-way ANOVA) or a list of strings (2-way+ ANOVA).
    """
    results = {}
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if filename is not None:
        filename = str(filename)

    # Get the function signature once
    sig = inspect.signature(test_to_run)
    param_names = set(sig.parameters.keys())

    # Normalise to a list so the call site is uniform
    group_columns = group_column if isinstance(group_column, list) else [group_column]

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

        results[col] = test_to_run(data, *group_columns, col, **kwargs)
    return results