import pytest
from fastapi.testclient import TestClient

from application import app, THRESHOLD, feature_names

client = TestClient(app)

def test_health():
    """Testa o endpoint /health."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict():
    """Testa o endpoint /predict com payload mínimo válido."""
    # Payload alinhado ao novo modelo: lista de pares candidate+job
    payload = {
        "data": [
            {
                "candidate": {
                    "area_atuacao": "TI - Desenvolvimento",
                    "cliente": "Empresa X",
                    "conhecimentos_tecnicos": "Python, AWS",
                    "eh_sap": False,
                    "idioma_requerido": "Inglês",
                    "nivel_academico": "Superior",
                    "nivel_espanhol": "Nenhum",
                    "nivel_ingles": "Avançado",
                    "nivel_profissional": "Sênior"
                },
                "job": {
                    "area_atuacao": "TI - Desenvolvimento",
                    "cliente": "Empresa X",
                    "conhecimentos_tecnicos": "Python, AWS",
                    "eh_sap": False,
                    "idioma_requerido": "Inglês",
                    "nivel_academico": "Superior",
                    "nivel_espanhol": "Nenhum",
                    "nivel_ingles": "Avançado",
                    "nivel_profissional": "Sênior"
                }
            }
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Esperado 200, mas veio {r.status_code}: {r.text}"
    d = r.json()
    # Espera 'results' como resposta, contendo uma lista
    assert "results" in d
    assert isinstance(d["results"], list)
    # Checa campos do primeiro resultado
    res = d["results"][0]
    for campo in ["mensagem", "status", "nota", "corte"]:
        assert campo in res

def test_exports():
    """Garante que constantes estão exportadas."""
    assert THRESHOLD is not None
    assert feature_names is not None
    assert isinstance(feature_names, list)
