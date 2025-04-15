import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .config import MODEL_PARAMS, SAVE_DIR
from .utils import print_info, print_success, print_result

def train_or_load_model(X_train, y_train, model_path):
    """
    Train a new model or load existing model
    
    Args:
        X_train: Training feature data
        y_train: Training label data
        model_path: Model save path
    
    Returns:
        Trained model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if os.path.exists(model_path):
        print_info(f"Loading existing model from {model_path}")
        return joblib.load(model_path)
    
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    print_success(f"Model trained and saved to {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test feature data
        y_test: Test label data
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print_result("R²", f"{r2:.4f}")
    print_result("RMSE", f"{rmse:.4f}")
    
    return {
        "r2": r2,
        "rmse": rmse,
        "predictions": y_pred
    }

def get_feature_importance(model, feature_names):
    """
    Get feature importance
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        Feature importance DataFrame
    """
    import pandas as pd
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': importance
    })
    return importance_df.sort_values(by='Importance', ascending=False)
