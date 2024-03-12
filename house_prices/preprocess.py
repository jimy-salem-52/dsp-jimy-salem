import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from __init__ import FEATURES_LIST, Y, CATEGORICAL_FEATURES, IMPUTER_PATH, ENCODER_PATH,SCALER_PATH

def selection(data:pd.DataFrame,is_inference:bool):
    selected_data=[]
    if is_inference is False:
        selected_data=data[FEATURES_LIST+Y]
    else:
        selected_data=data[FEATURES_LIST]
    return selected_data


def feature_selection(data:pd.DataFrame,is_inference:bool):
    imputer=joblib.load(IMPUTER_PATH)
    selected_data=selection(data,is_inference)
    if is_inference:
        imputed_data=imputer.transform(selected_data) #changed 
    else:
        imputer.fit(selected_data)
        imputed_data=imputer.transform(selected_data)
    imputed_data_df=pd.DataFrame(imputed_data,columns=selected_data.columns)
    return imputed_data_df

def encoding_function(data:pd.DataFrame, is_inference:bool):
    encoder=joblib.load(ENCODER_PATH)
    selected_data=feature_selection(data, is_inference)
    if is_inference is False:
        X=selected_data.drop(Y, axis=1)
        y=selected_data[Y]
        X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)
        encoder.fit(X_train[CATEGORICAL_FEATURES])
        X_train_categorical=encoder.transform(X_train[CATEGORICAL_FEATURES])
        X_test_categorical=encoder.transform(X_test[CATEGORICAL_FEATURES])
        X_train_categorical = pd.DataFrame(X_train_categorical, index=X_train.index, columns=encoder.get_feature_names_out())
        X_test_categorical = pd.DataFrame(X_test_categorical, index=X_test.index, columns=encoder.get_feature_names_out())
        return X_train_categorical, X_test_categorical, y_train, y_test
    else:
        if CATEGORICAL_FEATURES:
            encoded_features=encoder.transform(selected_data[CATEGORICAL_FEATURES])
            encoded_columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            encoded_df=pd.DataFrame(encoded_features,columns=encoded_columns,index=selected_data.index)
            data=data.drop(columns=CATEGORICAL_FEATURES)
            data=pd.concat([data,encoded_df], axis=1)
            return data
        else:
            return data

def scaling_function(data:pd.DataFrame,is_inference:bool):
    scaler=joblib.load(SCALER_PATH)
    selected_data=feature_selection(data,is_inference)
    if is_inference is False:
        X=selected_data.drop(Y, axis=1)
        y=selected_data[Y]
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        scaler.fit(X_train.drop(CATEGORICAL_FEATURES, axis=1))
        X_train_scaled=scaler.transform(X_train.drop(CATEGORICAL_FEATURES, axis=1))
        X_test_scaled=scaler.transform(X_test.drop(CATEGORICAL_FEATURES, axis=1))
        return X_train_scaled,X_test_scaled,y_train,y_test
    else:
        numeric_data=data.drop(columns=CATEGORICAL_FEATURES, errors='ignore')
        scaled_data=scaler.transform(numeric_data)
        return pd.DataFrame(scaled_data, columns=numeric_data.columns, index=data.index)

def preprocessing(data:pd.DataFrame, is_inference:bool):
    selected_data=feature_selection(data, is_inference)
    if is_inference is False:
        X_train_categorical,X_test_categorical,y_train,y_test=encoding_function(selected_data,is_inference)
        X_train_scaled,X_test_scaled,_,_ = scaling_function(selected_data, is_inference)
        X_train_final=np.hstack((X_train_scaled,X_train_categorical))
        X_test_final=np.hstack((X_test_scaled,X_test_categorical))
        return X_train_final,X_test_final,y_train,y_test
    else:
        selected_data_categorical=encoding_function(selected_data,is_inference)
        selected_data_scaled=scaling_function(selected_data,is_inference)
        combined_data=pd.concat([selected_data_scaled, selected_data_categorical],axis=1)
        combined_data=combined_data.loc[:,~combined_data.columns.duplicated()]
        return combined_data

def compute_rmsle(y_test:np.ndarray, y_pred:np.ndarray, precision:int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle,precision)
