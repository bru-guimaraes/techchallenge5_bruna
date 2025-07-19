import joblib
import pytest
from pathlib import Path
import os

# ----------------------------
# Dummy classes no nível de módulo
# ----------------------------

class DummyModel:
    def predict_proba(self, X):
        return [[0.4, 0.6] for _ in range(len(X))]

class DummyMLB:
    def __init__(self, classes):
        self.classes_ = classes
    def transform(self, series_of_lists):
        return [
            [1 if cls in item else 0 for cls in self.classes_]
            for item in series_of_lists
        ]

# ----------------------------
# Fixture global
# ----------------------------

@pytest.fixture(autouse=True)
def dummy_model_and_mlb(tmp_path, monkeypatch):
    # grava o DummyModel
    model_path = tmp_path / "modelo_classificador.joblib"
    joblib.dump(DummyModel(), model_path)

    # grava o DummyMLB
    mlb_path = tmp_path / "area_atuacao_mlb.joblib"
    joblib.dump(DummyMLB(["TI", "Dados", "Vendas", "Outro"]), mlb_path)

    # força seu código a usar esse PATH_MODEL
    monkeypatch.setenv("PATH_MODEL", str(model_path))
    return
