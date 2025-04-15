# CARin2050

This repository hosts the code related to the manuscript "CARin2050".

## Usage

### Clone the repository:
```bash
git clone https://github.com/impala3/CARin2050.git
cd CARin2050
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the program:

#### Process a single file:
```bash
python main.py path/to/your/file
```

#### Process all files in a directory:
```bash
python main.py path/to/your/directory
```

## Project Structure

- **data/** - Contains the input data files for model training
  - Data files should be in .txt format with tab-separated values

- **results/** - Contains saved models and analysis outputs
  - Models are saved as .pkl files
  - SHAP analysis results and plots are saved here
  - When existing models are found, they will be loaded instead of retraining

- **src/** - Source code for the project
  - data_processing.py: Functions for loading and preprocessing data
  - models.py: Model training, evaluation, and feature importance
  - analysis.py: SHAP analysis and visualization
  - utils.py: Helper functions for formatting and displaying results
  - config.py: Configuration parameters and settings

## Command Line Arguments

- `path`: Path to a single file or directory to process

## Notes

- The code will automatically create the necessary directories if they don't exist
- Results from each model run are saved in the results directory
- When an existing model is found, it will be loaded to save computation time
- SHAP values are also cached to avoid recalculation
- When processing a directory, all .txt files in the directory will be analyzed