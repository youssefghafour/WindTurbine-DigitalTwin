# model_logic.py
import os
import xgboost as xgb

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_CURRENT_DIR, "wind_final_full_train.json")

_booster = xgb.Booster()
_booster.load_model(_MODEL_PATH)

def predict_proba(df_features):
    d = xgb.DMatrix(df_features)
    return _booster.predict(d)

def get_prediction(data_row):
    features_only = data_row.drop(columns=["Target"], errors="ignore")
    prob = float(predict_proba(features_only)[0])
    status = "Failure Predicted" if prob >= 0.5 else "Healthy"
    return status, prob
