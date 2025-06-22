from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "cliente": "EmpresaX",
        "nivel_profissional": "Pleno",
        "idioma_requerido": "Inglês",
        "eh_sap": True,
        "area_atuacao": "Desenvolvimento",
        "nivel_ingles": "Avançado",
        "nivel_espanhol": "Básico",
        "formacao": "Superior Completo",
        "conhecimentos_tecnicos": "Python, SQL"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "contratado" in response.json()
