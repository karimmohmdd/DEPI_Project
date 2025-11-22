
# file_loader.py
import pandas as pd

def load_data(path: str):
    """
    Load processed CSV file and return DataFrame.
    """
    df = pd.read_csv(path)
    pd.set_option('display.max_columns', None)
    return df
