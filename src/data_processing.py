import pandas as pd
from sklearn.model_selection import train_test_split
from .config import FEATURE_NAME_MAPPING, EXCLUDE_COLUMNS, LABEL_COLUMN, FEATURE_COLUMNS

def load_data(file_path):
    """
    Load data and perform initial processing
    
    Args:
        file_path: Path to the data file
    
    Returns:
        X: Feature data
        y: Label data
        feature_cols: Renamed feature column names
    """
    data = pd.read_csv(file_path, sep='\t', header=0)
    
    if EXCLUDE_COLUMNS:
        data = data.drop(columns=EXCLUDE_COLUMNS)
    
    X = data[FEATURE_COLUMNS].copy()
    y = data[LABEL_COLUMN]
    
    X.rename(columns=FEATURE_NAME_MAPPING, inplace=True)
    
    feature_cols = [FEATURE_NAME_MAPPING[col] for col in FEATURE_COLUMNS]
    
    return X, y, feature_cols

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets
    
    Args:
        X: Feature data
        y: Label data
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Training and test sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
