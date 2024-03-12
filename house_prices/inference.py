import pandas as pd
import numpy as np
from preprocess import preprocessing
from __init__ import MODEL_PATH
import joblib

def make_prediction(data:pd.DataFrame)->np.ndarray:
    new_data=preprocessing(data,True)
    model=joblib.load(MODEL_PATH)
    y_pred=model.predict(new_data)
    return np.abs(y_pred)
