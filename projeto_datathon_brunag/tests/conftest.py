# tests/conftest.py

import os
import json
import joblib
import numpy as np
import shap
import pytest

from utils.paths import PATH_MODEL

DEFAULT_PROB = 0.7

class DummyModel:
    def predict_proba(self, X):
        n = len(X)
        return np.tile([1 - DEFAULT_PROB, DEFAULT_PROB], (n, 1))

class DummyExplainer:
    def shap_values(self, X):
        zero = np.zeros((len(X), X.shape[1]))
        return [zero, zero]

def pytest_configure(config):
    # 1) Garante que a pasta existe
    model_dir = os.path.dirname(PATH_MODEL)
    os.makedirs(model_dir, exist_ok=True)

    # 2) Salva o modelo dummy
    joblib.dump(DummyModel(), PATH_MODEL)

    # 3) Salva o features.json
    features_file = os.path.join(model_dir, "features.json")
    base_features = ["area_atuacao","nivel_ingles","nivel_espanhol","nivel_academico"]
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(base_features, f, ensure_ascii=False)

    # 4) **Stub do mapeamento de área** para o processar_features_inference
    mapping_file = os.path.join(model_dir, "area_atuacao_mlb.joblib")
    joblib.dump({}, mapping_file)

    # 5) Stub do explainer
    shap.TreeExplainer = lambda model: DummyExplainer()
