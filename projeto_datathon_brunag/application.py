from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import json
import pandas as pd
import os

from utils.paths import PATH_MODEL

app = FastAPI()

# Carrega modelo e lista de features
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r") as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")

class PredictRequest(BaseModel):
    area_atuacao: str
    nivel_ingles: Literal["baixo", "medio", "alto"]
    nivel_espanhol: Literal["baixo", "medio", "alto"]
    nivel_academico: Literal["medio", "superior", "pos", "mestrado", "doutorado"]

@app.get("/")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df_encoded = pd.get_dummies(df)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    try:
        pred = modelo.predict(df_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    return {"prediction": int(pred[0])}
