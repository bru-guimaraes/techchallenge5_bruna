from fastapi.testclient import TestClient
from application import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_predict_proba():
    payload = {
      "area_atuacao": "TI",
      "nivel_ingles": "alto",
      "nivel_espanhol": "medio",
      "nivel_academico": "superior"
    }
    r = client.post("/predict_proba", json=payload)
    assert r.status_code == 200
    assert "probability" in r.json()
