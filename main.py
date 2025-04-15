import os
import sys
import argparse
from src.config import ROOT_DIR, SAVE_DIR
from src.data_processing import load_data, split_train_test
from src.models import train_or_load_model, evaluate_model, get_feature_importance
from src.analysis import analyze_model
from src.utils import (
    get_base_filename, create_directories, print_header, print_result, 
    print_success, print_info, print_warning, print_table
)

def process_file(file_path):
    """Process a single data file"""
    base_name = get_base_filename(file_path)
    print_header(f"Processing file: {file_path}")
    
    # Load data
    X, y, feature_cols = load_data(file_path)
    print_info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Model save path
    model_path = os.path.join(SAVE_DIR, f"{base_name}_model.pkl")
    
    # Train or load model
    model = train_or_load_model(X_train, y_train, model_path)
    
    # Evaluate model
    print_header("Model Evaluation", char='-', color=None)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    importance_df = get_feature_importance(model, feature_cols)
    print_table(importance_df, title="Feature Importance")
    
    # SHAP analysis
    print_header("SHAP Analysis", char='-', color=None)
    analyze_model(model, X, feature_cols, base_name)
    
    print_success(f"\nProcessing complete for {base_name}")
    return metrics

def process_directory(directory_path):
    """Process all data files in a directory"""
    files_processed = 0
    results = {}
    
    print_header(f"Processing directory: {directory_path}")
    
    if not os.path.isdir(directory_path):
        print_warning(f"Warning: Directory not found: {directory_path}")
        return results
        
    for file in os.listdir(directory_path):
        if file.endswith('.txt'):
            file_path = os.path.join(directory_path, file)
            try:
                results[get_base_filename(file)] = process_file(file_path)
                files_processed += 1
            except Exception as e:
                print_warning(f"Error processing {file}: {str(e)}")
    
    if files_processed == 0:
        print_warning(f"No .txt files found in {directory_path}")
    else:
        print_success(f"Successfully processed {files_processed} files in {directory_path}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process data files for crop prediction models')
    parser.add_argument('path', help='Path to file or directory to process')
    args = parser.parse_args()
    
    print_header("Starting Model Training and Analysis", char='*')
    
    create_directories(SAVE_DIR)
    
    results = {}
    
    path = args.path
    if os.path.isfile(path):
        results[get_base_filename(path)] = process_file(path)
    elif os.path.isdir(path):
        results.update(process_directory(path))
    else:
        print_warning(f"Error: Path not found: {path}")
        parser.print_help()
        return
    
    if results:
        print_header("Summary of Results", char='*')
        for name, metrics in results.items():
            print_header(f"Results for {name}", char='-')
            print_result("R²", f"{metrics['r2']:.4f}")
            print_result("RMSE", f"{metrics['rmse']:.4f}")
        
        print_success("\nAll files processed successfully!")
    else:
        print_warning("No results to report. No files were processed successfully.")
    
    return results

if __name__ == "__main__":
    main()
