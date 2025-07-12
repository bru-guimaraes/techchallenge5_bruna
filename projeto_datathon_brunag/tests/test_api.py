import json
import pytest
from fastapi.testclient import TestClient
from application import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code==200
    assert "mensagem" in r.json()

def test_health():
    r = client.get("/health")
    assert r.status_code==200
    assert r.json()=={"status":"ok"}

def test_predict_valid():
    payload = {
      "area_atuacao":"Vendas",
      "nivel_ingles":"medio",
      "nivel_espanhol":"baixo",
      "nivel_academico":"superior"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code==200
    assert "prediction" in r.json()
