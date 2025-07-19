# src/tests/test_run_train.py

import os
import json
import joblib
import pytest
import yaml                              # <-- adicionado
from run_train import main as run_main
from utils.paths import PATH_MODEL

def test_run_train(tmp_path, monkeypatch):
    # prepara paths fake
    fake_model = tmp_path / "mymodel.joblib"
    fake_features = tmp_path / "features.json"

    # monkeypatcha o diretorio do modelo
    monkeypatch.setenv("PATH_MODEL", str(fake_model))

    # cria config completo para testes
    cfg = {
        "paths": {
            "parquet_treino_unificado": str(tmp_path / "data.parquet")
        },
        "features": {
            "target_column": "y"
        },
        "train": {
            "test_size": 0.2,
            "random_state": 42
        },
        "model": {
            "random_forest": {
                "n_estimators": 10
            }
        }
    }
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # cria parquet vazio com apenas a coluna target
    import pandas as pd
    pd.DataFrame({"y": []}).to_parquet(cfg["paths"]["parquet_treino_unificado"])

    # executa o pipeline
    run_main()

    # checa se arquivo do modelo e features.json existem
    assert fake_model.exists()
    assert fake_features.exists()

    # carrega e garante predict_proba existe
    m = joblib.load(fake_model)
    assert hasattr(m, "predict_proba")

    feats = json.loads(fake_features.read_text())
    assert isinstance(feats, list)
