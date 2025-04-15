import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from .config import SAVE_DIR
from .utils import print_info, print_success

def analyze_model(model, data, feature_cols, file_basename):
    """
    Analyze model feature importance and perform SHAP analysis
    
    Args:
        model: Trained model
        data: DataFrame containing features
        feature_cols: List of feature column names
        file_basename: Base filename for naming output files
    
    Returns:
        None
    """
    shap_values_path = os.path.join(SAVE_DIR, f"{file_basename}_shap_values.npy")
    model_save_path = os.path.join(SAVE_DIR, f"{file_basename}_model.pkl")
    plt_save_path = os.path.join(SAVE_DIR, f"{file_basename}_shap_summary.png")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    if os.path.exists(shap_values_path):
        print_info(f"Loading existing SHAP values from {shap_values_path}")
        shap_values = np.load(shap_values_path)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data[feature_cols])
        
        np.save(shap_values_path, shap_values)
        print_success(f"SHAP values saved to {shap_values_path}")
    
    print_info(f"SHAP values shape: {shap_values.shape}")
    
    data = data.copy()
    data['pred'] = model.predict(data[feature_cols])
    
    plt.rcParams['font.family'] = 'Arial'
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, data[feature_cols], show=False)
    plt.savefig(plt_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print_success(f"SHAP summary plot saved to {plt_save_path}")
    
    joblib.dump(model, model_save_path)
    print_success(f"Model saved to {model_save_path}")
    
    return shap_values
