import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Save directory
SAVE_DIR = os.path.join(ROOT_DIR, "results")

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "max_features": "sqrt"
}

# Feature name mapping
FEATURE_NAME_MAPPING = {
    'croppot': 'Quality',
    'tem': 'Tem',
    'pop': 'Pop',
    'pre': 'Pre',
    'solar': 'Solar',
    'ENNMN': 'ENNMN',
    'PARAMN': 'PARAMN',
    'gdp': 'GDP',
    'PD': 'PD',
    'AI': 'AI',
    'dem': 'DEM',
    'slope': 'Slope',
    'som': 'SOM',
    'igg': 'Clrr',
    'riverdis': 'Riverdis',
    'roaddis': 'Roaddis',
    'setdis': 'Setdis'
}

# Columns to exclude
EXCLUDE_COLUMNS = ["Year"]

# Label column
LABEL_COLUMN = "Cropabon"

# Feature columns derived from FEATURE_NAME_MAPPING
FEATURE_COLUMNS = list(FEATURE_NAME_MAPPING.keys())
