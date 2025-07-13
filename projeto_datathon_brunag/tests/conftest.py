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
        return [np.zeros((len(X), X.shape[1])), np.zeros((len(X), X.shape[1]))]

def pytest_sessionstart(session):
    model_dir = os.path.dirname(PATH_MODEL)
    os.makedirs(model_dir, exist_ok=True)

    # cria o modelo dummy
    joblib.dump(DummyModel(), PATH_MODEL)

    # cria o features.json real
    features_file = os.path.join(model_dir, "features.json")
    fake_features = ["area_atuacao", "nivel_ingles", "nivel_espanhol", "nivel_academico"]
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(fake_features, f)

    # stuba o explainer
    shap.TreeExplainer = lambda model: DummyExplainer()
