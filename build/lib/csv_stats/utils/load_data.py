from typing import Union
from pathlib import Path

import pandas as pd

def load_data_from_path(data: Union[Path, str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load data from a file path or return the DataFrame if already provided.

    Parameters:
    data : Union[Path, str, pd.DataFrame]
        The file path to load data from or a DataFrame.

    Returns:
    pd.DataFrame
        The loaded DataFrame.
    """

    if isinstance(data, pd.DataFrame):
        return data
    
    # Convert string paths to Path objects
    if isinstance(data, str):
        data = Path(data)
    
    if isinstance(data, Path):
        check_file_exists(data)
        return pd.read_csv(data)
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")


def check_file_exists(path: Union[Path, str]):
    """
    Check that the provided file path exists.

    Parameters:
    path : Union[Path, str]
        The file path to check.
    """
    if isinstance(path, str):
        path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Unable to find a file at {path}")