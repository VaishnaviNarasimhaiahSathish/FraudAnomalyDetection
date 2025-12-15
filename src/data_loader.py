import pandas as pd

def load_dataset(path):
    """Load raw credit card fraud CSV file."""
    return pd.read_csv(path)
