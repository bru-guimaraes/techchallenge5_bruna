# tests/conftest.py

import os
import json
import joblib
import numpy as np
import shap
import pytest

from utils.paths import PATH_MODEL

# Probabilidade fixa para predict_proba
DEFAULT_PROB = 0.7

class DummyModel:
    def predict_proba(self, X):
        # sempre retorna [1-DEFAULT_PROB, DEFAULT_PROB]
        n = len(X)
        return np.tile([1 - DEFAULT_PROB, DEFAULT_PROB], (n, 1))

class DummyExplainer:
    def shap_values(self, X):
        # retorna zeros na forma esperada (2 classes, n amostras × m features)
        zero = np.zeros((len(X), X.shape[1]))
        return [zero, zero]

def pytest_configure(config):
    """
    Executado antes de carregar os módulos de teste.
    Aqui criamos:
      1) O arquivo PATH_MODEL (modelo dummy)
      2) O features.json (lista base de features)
      3) Fazemos monkey-patch no shap.TreeExplainer para usar DummyExplainer
    """
    # 1) Garante que a pasta do modelo existe
    model_dir = os.path.dirname(PATH_MODEL)
    os.makedirs(model_dir, exist_ok=True)

    # 2) Cria e salva o modelo dummy
    dummy_model = DummyModel()
    joblib.dump(dummy_model, PATH_MODEL)

    # 3) Cria o features.json com as 4 features básicas
    features_file = os.path.join(model_dir, "features.json")
    base_features = [
        "area_atuacao",
        "nivel_ingles",
        "nivel_espanhol",
        "nivel_academico"
    ]
    with open(features_file, "w", encoding="utf-8") as f:
        json.dump(base_features, f, ensure_ascii=False)

    # 4) Stub do explainer para que TreeExplainer retorne DummyExplainer
    shap.TreeExplainer = lambda model: DummyExplainer()
