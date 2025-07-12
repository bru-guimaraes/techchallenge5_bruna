import os
import json
import joblib
from utils.paths import PATH_MODEL

def test_model_and_features_exist():
    assert os.path.isfile(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL),"features.json")
    assert os.path.isfile(features_path)
    with open(features_path) as f:
        feats = json.load(f)
    assert isinstance(feats, list) and len(feats)>0
    mdl = joblib.load(PATH_MODEL)
    assert hasattr(mdl, "predict")
