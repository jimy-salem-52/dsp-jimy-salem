# from preprocess import feature_selection, scaling_func, path, hot_encoding_func
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_log_error
# import joblib, os

# #avoid repetition
# features_list = ["LotArea", "Neighborhood", "TotalBsmtSF", "GrLivArea", "BldgType", "GarageArea"]
# y = ["SalePrice"]
# categorical_features = ["Neighborhood", "BldgType"]

# def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
#     rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
#     return round(rmsle, precision)

# def build_model_refactored(data: pd.DataFrame) -> dict[str, str]:
#     result = {}
#     is_test = False
#     new_data = feature_selection(data, is_test, features_list, y)
#     X_train_categorical, X_test_categorical, y_train, y_test = hot_encoding_func(new_data, is_test, categorical_features, y)
#     X_train_scaled, X_test_scaled, _, _ = scaling_func(new_data, is_test, categorical_features, y)

#     X_train_final = np.hstack((X_train_scaled, X_train_categorical))
#     X_test_final = np.hstack((X_test_scaled, X_test_categorical))

#     model = joblib.load(os.path.join(path, 'model.joblib'))
#     model.fit(X_train_final, y_train)
#     y_pred = model.predict(X_test_final)
#     y_pred = np.clip(y_pred, 0, None)

#     result['rmse'] = compute_rmsle(y_test, y_pred)

#     return result

import pandas as pd
import joblib
from Preprocess_new import preprocessing, compute_rmsle
from __init__ import MODEL_PATH

def build_model(data:pd.DataFrame):
    X_train, X_test,y_train,y_test=preprocessing(data, False)
    model = joblib.load(MODEL_PATH)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return dict({"rmsle":compute_rmsle(y_pred,y_test)})

