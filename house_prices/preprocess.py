import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split

path = r'C:\Users\SADEK COMPUTER\Desktop\Epita\01 - Semester 2\Data Science Production\Github Assignment\dsp-jimy-salem\models'

def feature_selection(data: pd.DataFrame, is_test: bool, features: list, y_out: list):
    if is_test:
        selected_data = data[features].copy()
        selected_data.dropna(subset=features, inplace=True)
        return selected_data
    else:
        selected_data = data[features + y_out].copy()
        selected_data.dropna(subset=features + y_out, inplace=True)
        X = selected_data[features]
        y = selected_data[y_out]
        return pd.concat([X, y], axis=1)

def train_split_func(data: pd.DataFrame, is_test: bool, y_out: list):
    if is_test == False:
        y = data[y_out]
        X = data.drop(columns=y_out, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        pass

def hot_encoding_func(data: pd.DataFrame, is_test: bool, categorical_features: list, y_out: list):
    onehot_encoder = joblib.load(os.path.join(path, 'one_hot_encoder.joblib'))
    if is_test == False:
        X_train, X_test, y_train, y_test = train_split_func(data, is_test, y_out)
        X_train_categorical = onehot_encoder.fit_transform(X_train[categorical_features])
        X_test_categorical = onehot_encoder.transform(X_test[categorical_features])
        return X_train_categorical, X_test_categorical, y_train, y_test
    else:
        if categorical_features:
            encoded_features = onehot_encoder.transform(data[categorical_features])
            categories = onehot_encoder.categories_
            encoded_columns = [f"{column}_{category}" for column, category_list in zip(categorical_features, categories) for category in category_list[1:]]
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
            data.drop(columns=categorical_features, inplace=True)
            return pd.concat([data, encoded_df], axis=1)
        else:
            return data

def scaling_func(data: pd.DataFrame, is_test: bool, categorical_features: list, y_out: list):
    scaler = joblib.load(os.path.join (path, 'scaler.joblib'))
    #categorical_features = ['Neighborhood', 'BldgType']
    if is_test == False:
        X_train, X_test, y_train, y_test = train_split_func(data, is_test, y_out)
        X_train_scaled = scaler.fit_transform(X_train.drop(categorical_features, axis=1))
        X_test_scaled = scaler.transform(X_test.drop(categorical_features, axis=1))
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        data.dropna(inplace=True)
        data = scaler.fit_transform(data)
        return data

