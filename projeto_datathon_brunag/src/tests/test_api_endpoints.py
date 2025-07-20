import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

import application
from application import app, THRESHOLD, feature_names
from utils.paths import PATH_MODEL

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "mensagem" in body
    assert isinstance(body["mensagem"], str)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_match_vaga_sem_dados(tmp_path, monkeypatch):
    base = tmp_path / "data" / "parquet"
    apps = base / "applicants"
    pros = base / "prospects"
    vagas = base / "vagas"
    apps.mkdir(parents=True)
    pros.mkdir(parents=True)
    vagas.mkdir(parents=True)
    pd.DataFrame().to_parquet(apps / "applicants.parquet")
    pd.DataFrame().to_parquet(pros / "prospects.parquet")
    pd.DataFrame({"id_vaga":[1], "codigo_profissional":[123]}).to_parquet(vagas / "vagas.parquet")
    monkeypatch.chdir(tmp_path)

    r = client.get("/match/999")
    assert r.status_code == 404

def test_predict_via_application(monkeypatch):
    payload = {
        "cliente": "Empresa X",
        "nivel_profissional": "Senior",
        "idioma_requerido": "ingles",
        "eh_sap": True,
        "area_atuacao": "Dados",
        "nivel_ingles": "alto",
        "nivel_espanhol": "baixo",
        "formacao": "superior",
        "conhecimentos_tecnicos": "python",
        # aqui o valor deve ser um dos literal enums: 'medio','superior','pos','mestrado','doutorado'
        "nivel_academico": "superior"
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Esperado 200, mas veio {r.status_code}: {r.text}"
    d = r.json()

    # esperamos estes quatro campos
    assert set(d.keys()) == {"previsao", "probabilidade", "status", "detalhe"}

    # tipos corretos
    assert isinstance(d["previsao"], int)
    assert isinstance(d["probabilidade"], float)
    assert isinstance(d["status"], str)
    assert isinstance(d["detalhe"], str)

def test_exports():
    assert hasattr(application, "THRESHOLD")
    assert hasattr(application, "feature_names")
    assert isinstance(feature_names, list)
