import os
FEATURES_LIST= ["LotArea", "Neighborhood", "TotalBsmtSF", "GrLivArea", "BldgType", "GarageArea"]
Y = ["SalePrice"]
CATEGORICAL_FEATURES = ["Neighborhood", "BldgType"]
PATH=r'../models'
IMPUTER_PATH=os.path.join(PATH,'imputer.joblib')
ENCODER_PATH=os.path.join(PATH,'one_hot_encoder.joblib')
SCALER_PATH=os.path.join(PATH,'scaler.joblib')
MODEL_PATH=os.path.join(PATH,'model.joblib')

