import os
import json
import joblib
import pytest
from run_train import main as run_main
from utils.paths import PATH_MODEL

def test_run_train_creates_artifacts(tmp_path, monkeypatch):
    # aponta o PATH_MODEL para um diretório temporário
    fake_model = tmp_path / "model.pkl"
    fake_features = tmp_path / "features.json"
    monkeypatch.setenv("PATH_MODEL", str(fake_model))

    # e também os parquet, aponte para cópias de amostra
    # monkeypatch.setenv("PATH_PARQUET_APPLICANTS", "...") etc.

    # Execute o treino
    run_main()

    # Verifica arquivos
    assert fake_model.exists(), "Modelo não foi salvo"
    assert fake_features.exists(), "features.json não foi salvo"

    # Carrega e valida
    mdl = joblib.load(str(fake_model))
    with open(str(fake_features)) as f:
        feats = json.load(f)
    assert isinstance(feats, list) and len(feats)>0
    assert hasattr(mdl, "predict")
