import os
import json
import joblib
import pytest
import yaml
import pandas as pd
from pathlib import Path

from run_train import main

def test_run_train(tmp_path, monkeypatch):
    """Testa o pipeline de treino, geração dos artefatos e interface do modelo."""

    # ---- 1) Isola diretório ----
    os.chdir(tmp_path)

    # ---- 2) Fakes para output ----
    fake_model    = tmp_path / "mymodel.joblib"
    fake_features = tmp_path / "features.json"
    monkeypatch.setenv("PATH_MODEL", str(fake_model))
    monkeypatch.setenv("FEATURES_JSON_PATH", str(fake_features))

    # ---- 3) Gera dataset desbalanceado + coluna extra ----
    df = pd.DataFrame([
        {
            "cliente": "A", "nivel_profissional": "senior", "idioma_requerido": "ingles",
            "area_atuacao": "Dados", "nivel_ingles": "alto", "nivel_espanhol": "baixo",
            "nivel_academico": "superior", "conhecimentos_tecnicos": "python", "eh_sap": True,
            "job_cliente": "A", "job_nivel_profissional": "senior", "job_idioma_requerido": "ingles",
            "job_area_atuacao": "Dados", "job_nivel_ingles": "alto", "job_nivel_espanhol": "baixo",
            "job_nivel_academico": "superior", "job_conhecimentos_tecnicos": "python", "job_eh_sap": True,
            "y": 1, "col_extra": 42  # col_extra deve ser ignorada
        },
        {
            "cliente": "B", "nivel_profissional": "junior", "idioma_requerido": "espanhol",
            "area_atuacao": "TI", "nivel_ingles": "baixo", "nivel_espanhol": "alto",
            "nivel_academico": "medio", "conhecimentos_tecnicos": "sql", "eh_sap": False,
            "job_cliente": "B", "job_nivel_profissional": "junior", "job_idioma_requerido": "espanhol",
            "job_area_atuacao": "TI", "job_nivel_ingles": "baixo", "job_nivel_espanhol": "alto",
            "job_nivel_academico": "medio", "job_conhecimentos_tecnicos": "sql", "job_eh_sap": False,
            "y": 0, "col_extra": 123  # col_extra deve ser ignorada
        },
        # Mais uma linha para garantir desbalanceamento...
        {
            "cliente": "C", "nivel_profissional": "pleno", "idioma_requerido": "frances",
            "area_atuacao": "Infra", "nivel_ingles": "medio", "nivel_espanhol": "nenhum",
            "nivel_academico": "pos", "conhecimentos_tecnicos": "aws", "eh_sap": False,
            "job_cliente": "C", "job_nivel_profissional": "pleno", "job_idioma_requerido": "frances",
            "job_area_atuacao": "Infra", "job_nivel_ingles": "medio", "job_nivel_espanhol": "nenhum",
            "job_nivel_academico": "pos", "job_conhecimentos_tecnicos": "aws", "job_eh_sap": False,
            "y": 0, "col_extra": 999  # col_extra deve ser ignorada
        },
    ])
    uni = tmp_path / "unificado.parquet"
    df.to_parquet(uni)

    # ---- 4) Gera config.yaml ----
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

    # ---- 5) Roda treino ----
    main()

    # ---- 6) Valida artefatos ----
    assert fake_model.exists(),   "Modelo não foi gerado"
    assert fake_features.exists(), "features.json não foi gerado"

    # ---- 7) Testa interface do modelo ----
    m = joblib.load(fake_model)
    assert hasattr(m, "predict_proba")
    feats = json.loads(fake_features.read_text(encoding="utf-8"))
    assert isinstance(feats, list)
    # match_score tem que estar nas features
    assert "match_score" in feats
    # col_extra NÃO pode estar nas features
    assert "col_extra" not in feats

    # ---- 8) Garante que features são exatamente as esperadas ----
    # Isso depende da sua lista FEATURES no run_train.py!
    expected = [
        "cliente",
        "nivel_profissional",
        "idioma_requerido",
        "area_atuacao",
        "nivel_ingles",
        "nivel_espanhol",
        "nivel_academico",
        "conhecimentos_tecnicos",
        "eh_sap",
        "match_score"
    ]
    for f in expected:
        assert any(f in ff for ff in feats), f"Feature {f} não encontrada nas features.json"

def test_target_col_inexistente(tmp_path, monkeypatch):
    """Garante que erro é levantado caso target_column não exista."""
    os.chdir(tmp_path)
    fake_model    = tmp_path / "mymodel.joblib"
    fake_features = tmp_path / "features.json"
    monkeypatch.setenv("PATH_MODEL", str(fake_model))
    monkeypatch.setenv("FEATURES_JSON_PATH", str(fake_features))

    # Gera dataset sem coluna alvo
    df = pd.DataFrame([{"a": 1, "b": 2}])
    uni = tmp_path / "unificado.parquet"
    df.to_parquet(uni)
    cfg = {
        "paths": {"parquet_treino_unificado": str(uni)},
        "features": {"target_column": "y"},  # coluna "y" não existe!
        "model": {"random_forest": {"n_estimators": 5, "random_state": 0}},
        "train": {"test_size": 0.5, "random_state": 0}
    }
    Path("config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    with pytest.raises(KeyError):
        main()
