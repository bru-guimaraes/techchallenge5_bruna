import os
import logging
import joblib
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from pythonjsonlogger import jsonlogger

# — logger JSON —
logger = logging.getLogger("recruitment_api")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Decision Recruitment API",
    version="1.0.0",
    description="Classificação binária de candidatos (match sim/não) com threshold dinâmico"
)

# Modelo de entrada
class Candidate(BaseModel):
    cliente: str
    nivel_profissional: str
    idioma_requerido: str
    eh_sap: bool
    area_atuacao: str
    nivel_ingles: str
    nivel_espanhol: str
    nivel_academico: str
    conhecimentos_tecnicos: str

# carrega pipeline + lista de features
MODEL_PATH = os.getenv("PATH_MODEL", "model/pipeline.joblib")
try:
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Pipeline carregado de {MODEL_PATH}")
except Exception as e:
    logger.error(f"Falha ao carregar pipeline em {MODEL_PATH}: {e}")
    raise

# extrai nomes das features usadas pelo preproc
try:
    feature_names = pipeline.named_steps["preproc"].get_feature_names_out().tolist()
except Exception:
    feature_names = pipeline.named_steps["preproc"].get_feature_names().tolist()

THRESHOLD = 0.5

def get_positive_proba(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError("Formato de saída de predict_proba inesperado")

@app.post("/predict")
def predict(candidate: Candidate):
    data = candidate.dict()
    df = pd.DataFrame([data])

    # Garante que todas as feature_names existem no DataFrame
    for col in feature_names:
        if col not in df.columns:
            # preenche faltantes com 0 ou False
            df[col] = 0

    # Reordena as colunas na ordem esperada pelo pipeline
    X = df[feature_names]

    try:
        proba = get_positive_proba(pipeline.predict_proba(X))
    except Exception as e:
        logger.error(f"Erro ao chamar predict_proba: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao calcular previsão")

    match = bool(proba[0] >= THRESHOLD)
    return {
        "probability": float(proba[0]),
        "match": match
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features")
def features():
    return {"features": feature_names}
