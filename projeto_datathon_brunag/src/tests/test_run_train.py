import os
import json
import joblib
import pytest
import yaml
import pandas as pd
from pathlib import Path

from run_train import main

def test_run_train(tmp_path, monkeypatch):
    # fake PATH_MODEL
    fake_model    = tmp_path / "mymodel.joblib"
    fake_features = tmp_path / "features.json"
    monkeypatch.setenv("PATH_MODEL", str(fake_model))

    # 1) cria parquet unificado mínimo (agora com 2 linhas!)
    df = pd.DataFrame([
        {"a": 1, "y": 0},
        {"a": 2, "y": 1}
    ])
    uni = tmp_path / "unificado.parquet"
    df.to_parquet(uni)

    # 2) escreve config.yaml (adiciona train/test_size)
    cfg = {
        "paths": {
            "parquet_treino_unificado": str(uni),
            "candidatos_parquet_dir": "",
            "prospects_parquet_dir": "",
            "vagas_parquet_dir": ""
        },
        "features": {"target_column": "y"},
        "model": {"random_forest": {"n_estimators": 5, "random_state": 0}},
        "train": {"test_size": 0.5, "random_state": 0}
    }
    Path("config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    # 3) roda treino
    main()

    # 4) checa artefatos
    assert fake_model.exists(),   "Modelo não foi gerado"
    assert fake_features.exists(), "features.json não foi gerado"

    # 5) interface do modelo
    m = joblib.load(fake_model)
    assert hasattr(m, "predict_proba")

    feats = json.loads(fake_features.read_text(encoding="utf-8"))
    assert isinstance(feats, list)
    assert "a" in feats
