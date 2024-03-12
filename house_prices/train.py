import pandas as pd
import joblib
from preprocess import preprocessing, compute_rmsle
from __init__ import MODEL_PATH

def build_model(data:pd.DataFrame):
    X_train, X_test,y_train,y_test=preprocessing(data, False)
    model = joblib.load(MODEL_PATH)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return dict({"rmsle":compute_rmsle(y_pred,y_test)})

