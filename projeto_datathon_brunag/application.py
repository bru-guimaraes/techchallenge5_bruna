from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import json
import pandas as pd
import os

from utils.paths import PATH_MODEL

app = FastAPI()

# ——— 1. Carregamento do modelo e da lista de features ———
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r") as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")
# ————————————————————————————————————————————————

# Exemplo de modelo de entrada (mantido)
class EntradaExemplo(BaseModel):
    nome: str
    idade: int

# Payload para /predict
class PredictRequest(BaseModel):
    area_atuacao: str
    nivel_ingles: Literal["baixo", "medio", "alto"]
    nivel_espanhol: Literal["baixo", "medio", "alto"]
    nivel_academico: Literal["medio", "superior", "pos", "mestrado", "doutorado"]

# Rota raiz
@app.get("/")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

# Rota de exemplo com POST
@app.post("/exemplo")
def exemplo_endpoint(dados: EntradaExemplo):
    return {
        "mensagem": f"Olá, {dados.nome}! Você tem {dados.idade} anos."
    }

# ——— Endpoint de predição ———
@app.post("/predict")
def predict(req: PredictRequest):
    # 1) transformar payload em DataFrame
    df = pd.DataFrame([req.dict()])

    # 2) one-hot encoding
    df_encoded = pd.get_dummies(df)

    # 3) alinhar colunas com o treino
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    # 4) predição
    try:
        pred = modelo.predict(df_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    # 5) retorno
    return {"prediction": int(pred[0])}
# —————————————————————————————————————————————————————

# ——— Health check endpoint ———
@app.get("/health")
def health():
    return {"status": "ok"}
# —————————————————————————————————————————————————————
