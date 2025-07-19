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
    assert "mensagem" in r.json()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_feedback_and_historico(tmp_path, monkeypatch):
    content = b"year,col1\n2020,10"
    resp = client.post(
        "/historico",
        files={"file": ("data.csv", content, "text/csv")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "salvo"
    assert "caminho" in body

def test_match_vaga_sem_dados(monkeypatch):
    os.makedirs("data/parquet/applicants", exist_ok=True)
    os.makedirs("data/parquet/prospects", exist_ok=True)
    os.makedirs("data/parquet/vagas", exist_ok=True)
    pd.DataFrame().to_parquet("data/parquet/applicants/applicants.parquet")
    pd.DataFrame().to_parquet("data/parquet/prospects/prospects.parquet")
    pd.DataFrame({"id_vaga":[1], "codigo_profissional":[123]}).to_parquet("data/parquet/vagas/vagas.parquet")
    r = client.get("/match/999")
    assert r.status_code == 404

def test_predict_via_application(monkeypatch):
    payload = {
        "area_atuacao": "TI,Dados",
        "nivel_ingles": "alto",
        "nivel_espanhol": "medio",
        "nivel_academico": "mestrado"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert set(d.keys()) == {"previsao", "probabilidade"}

def test_exports():
    assert hasattr(application, "THRESHOLD")
    assert hasattr(application, "feature_names")
    assert isinstance(feature_names, list)
