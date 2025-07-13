# tests/test_api_endpoints.py

import os
import json
import io
import zipfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

import application                # para monkeypatch do PATH_MODEL
from application import app, THRESHOLD, feature_names
from utils import paths           # caso precise de outros patches

client = TestClient(app)

# ----------------------
# Endpoints básicos
# ----------------------

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"mensagem": "API funcionando com sucesso 🚀"}

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_health_detailed():
    resp = client.get("/health_detailed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "uptime_seconds" in body
    assert "python_version" in body

def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")

def test_model_info():
    resp = client.get("/model_info")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {"version", "trained_on", "validation_accuracy"}

def test_features():
    resp = client.get("/features")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["features"]) == set(feature_names)

# ----------------------
# Payload de predição
# ----------------------

valid_payload = {
    "area_atuacao": "desenvolvimento",
    "nivel_ingles": "medio",
    "nivel_espanhol": "baixo",
    "nivel_academico": "superior"
}

@pytest.mark.parametrize("route", ["/predict", "/predict_proba"])
def test_predict_and_proba_success(route):
    resp = client.post(route, json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body and "probability" in body
    assert isinstance(body["prediction"], int)
    assert isinstance(body["probability"], float)
    assert body["prediction"] in (0, 1)

def test_predict_and_proba_validation_error():
    partial = {"area_atuacao": "vendas"}
    resp1 = client.post("/predict", json=partial)
    resp2 = client.post("/predict_proba", json=partial)
    assert resp1.status_code == 422
    assert resp2.status_code == 422

def test_batch_predict_success():
    payload = {"inputs": [valid_payload, valid_payload]}
    resp = client.post("/batch_predict", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert isinstance(results, list) and len(results) == 2
    for item in results:
        assert "prediction" in item and "probability" in item

def test_batch_predict_validation_error():
    resp = client.post("/batch_predict", json={"inputs": [{}]})
    assert resp.status_code == 422

# ----------------------
# Endpoints de explicação
# ----------------------

def test_explain_success():
    resp = client.post("/explain", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"prediction", "probability", "explanation"}
    assert isinstance(body["explanation"], dict)
    assert set(body["explanation"].keys()) == set(feature_names)

def test_global_explain():
    resp = client.get("/global_explain")
    assert resp.status_code == 200
    body = resp.json()
    assert "global_importance" in body
    assert isinstance(body["global_importance"], dict)
    assert set(body["global_importance"].keys()) == set(feature_names)

def test_compare():
    payload = {
        "cand_a": valid_payload,
        "cand_b": {**valid_payload, "nivel_ingles": "alto"}
    }
    resp = client.post("/compare", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"a", "b", "delta"}
    assert set(body["delta"].keys()) == set(feature_names)

# ----------------------
# Endpoint de threshold
# ----------------------

def test_get_and_set_threshold():
    resp1 = client.get("/threshold")
    orig = resp1.json()["threshold"]

    new_val = 0.75
    resp2 = client.post("/threshold", json={"threshold": new_val})
    assert resp2.status_code == 200
    assert resp2.json()["threshold"] == new_val

    # restaura valor original para não impactar outros testes
    client.post("/threshold", json={"threshold": orig})

# ----------------------
# Endpoint de feedback
# ----------------------

def test_feedback_append(tmp_path, monkeypatch):
    """
    POST /feedback deve gravar JSONL ao lado do modelo (PATH_MODEL).
    Aqui monkeypatchamos application.PATH_MODEL para um arquivo vazio
    dentro de tmp_path e, depois, validamos o log gerado.
    """
    # 1) Prepara diretório fake e arquivo de modelo vazio
    fake_dir = tmp_path / "model_dir"
    fake_dir.mkdir()
    fake_model = fake_dir / "modelo_classificador.pkl"
    fake_model.write_bytes(b"")  # evita erro de load

    # 2) Monkeypatcha o PATH_MODEL usado pela aplicação
    monkeypatch.setattr(application, "PATH_MODEL", str(fake_model))

    # 3) Chama endpoint de feedback
    payload = {"input": valid_payload, "prediction": 1, "actual": 0}
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    # 4) Determina onde a aplicação teria gravado o log
    log_path = Path(os.path.dirname(application.PATH_MODEL)) / "feedback_log.jsonl"
    assert log_path.exists(), f"Esperava log em {log_path}, mas não encontrei"

    # 5) Verifica conteúdo do último registro
    lines = log_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec == payload

# ----------------------
# Endpoint de upload de histórico
# ----------------------

def test_upload_historical(tmp_path):
    data = "col1,col2\n1,2\n3,4"
    file_obj = io.BytesIO(data.encode())
    file_obj.name = "hist.csv"

    resp = client.post(
        "/historical_data",
        files={"file": ("hist.csv", file_obj, "text/csv")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "saved"

    path = Path(body["path"])
    assert path.exists()
    df = pd.read_csv(path)
    assert not df.empty
