import json
import pytest
from fastapi.testclient import TestClient
from application import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "mensagem" in r.json()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_valid():
    payload = {
        "area_atuacao": "Vendas",
        "nivel_ingles": "medio",
        "nivel_espanhol": "baixo",
        "nivel_academico": "superior"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in (0,1)

def test_predict_invalid():
    # missing field
    r = client.post("/predict", json={"area_atuacao":"Vendas"})
    assert r.status_code == 422  # Unprocessable Entity
