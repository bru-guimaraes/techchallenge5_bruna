import pandas as pd
from typing import List
from .paths import PATH_MODEL, load_feature_names
import joblib

_model = joblib.load(PATH_MODEL)
_feature_names = load_feature_names()

def preprocess(req: dict) -> pd.DataFrame:
    df = pd.DataFrame([req])
    df_enc = pd.get_dummies(df)
    return df_enc.reindex(columns=_feature_names, fill_value=0)

def predict_single(data: dict, threshold: float) -> (int, float):
    df = preprocess(data)
    probs = _model.predict_proba(df)[:, 1]
    p = float(probs[0])
    return int(p >= threshold), p

def predict_batch(list_req: List[dict], threshold: float) -> List[dict]:
    df = pd.DataFrame(list_req)
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=_feature_names, fill_value=0)
    probs = _model.predict_proba(df_aligned)[:, 1]
    return [{"prediction": int(p >= threshold), "probability": float(p)} for p in probs]
