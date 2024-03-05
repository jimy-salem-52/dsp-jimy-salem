from preprocess import feature_selection, scaling_func, path, hot_encoding_func
import numpy as np
import pandas as pd
import joblib, os
from train import features_list, categorical_features

def make_prediction_refactored(input_data: pd.DataFrame) -> np.ndarray:
    is_test = True
    # Feature selection
    new_data = feature_selection(input_data, is_test, features_list, [])
    # Encoding
    new_data = hot_encoding_func(new_data, is_test, categorical_features, [])
    # Scaling
    new_data = scaling_func(new_data, is_test, categorical_features, [])
 
    model_unload = joblib.load(os.path.join(path, 'model.joblib'))
    y_pred_test = model_unload.predict(new_data)
    y_pred_test = np.abs(y_pred_test)
 
    return y_pred_test

