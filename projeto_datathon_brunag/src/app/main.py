from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import json
import os

from app.esquema import DadosEntrada
from utils.paths import PATH_MODEL

# Caminho para o features.json
FEATURES_PATH = os.path.join(os.path.dirname(PATH_MODEL), "features.json")

app = FastAPI()

# Carrega modelo e features uma única vez
model = joblib.load(PATH_MODEL)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API Recruitment running!"}

@app.post("/predict-candidato")
def predict_candidato(dados: DadosEntrada):
    """
    Prediz a probabilidade de contratação de um candidato a partir de um corpo JSON estruturado.
    """
    # 1. Recebe os dados como DataFrame
    df = pd.DataFrame([dados.dict()])

    # 2. One-hot encoding nas categóricas, igual ao treino
    X = pd.get_dummies(df, drop_first=True)

    # 3. Garante que as colunas estão alinhadas com o modelo treinado
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURES]

    # 4. Prediz a probabilidade
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    status = "MATCH" if pred == 1 else "NO_MATCH"

    return {
        "previsao": pred,
        "probabilidade": proba,
        "status": status,
        "detalhe": (
            f"Candidato com probabilidade de sucesso de {proba*100:.1f}%. "
            f"Threshold: 50%."
        ),
        "input_recebido": dados.dict(),
    }
