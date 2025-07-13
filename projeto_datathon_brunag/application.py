import time
import logging
from fastapi import FastAPI, HTTPException, Response, Request
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import Literal
import joblib
import json
import pandas as pd
import os

from utils.paths import PATH_MODEL

# ——— Logger JSON —————————————
logger = logging.getLogger("uvicorn.access")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# ——————————————————————————————

# ——— Métricas Prometheus —————————
REQUEST_COUNT = Counter(
    'request_count', 'Total HTTP requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'HTTP request latency',
    ['endpoint']
)
# ——————————————————————————————

app = FastAPI()

# ——— 1. Carregamento do modelo e features ———
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r") as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")
# ————————————————————————————————————————————————

# ——— Pydantic Models ———————————————
class EntradaExemplo(BaseModel):
    nome: str
    idade: int

class PredictRequest(BaseModel):
    area_atuacao: str
    nivel_ingles: Literal["baixo", "medio", "alto"]
    nivel_espanhol: Literal["baixo", "medio", "alto"]
    nivel_academico: Literal["medio", "superior", "pos", "mestrado", "doutorado"]
# ————————————————————————————————————————————————

# ——— Middleware para logs + métricas ———
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    # Prometheus metrics
    REQUEST_LATENCY.labels(request.url.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()

    # JSON log
    logger.info('', extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency": latency
    })

    return response
# ————————————————————————————————————————————————

# ——— Endpoint de métricas Prometheus ———
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
# ——————————————————————————————————————————————

# ——— Health check ———————————————
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}
# ——————————————————————————————————————————————

# ——— Exemplo de endpoint ———————————
@app.post("/exemplo")
def exemplo_endpoint(dados: EntradaExemplo):
    return {"mensagem": f"Olá, {dados.nome}! Você tem {dados.idade} anos."}
# ——————————————————————————————————————————————

# ——— Predict endpoint —————————————
@app.post("/predict")
def predict(req: PredictRequest):
    # transforma payload em DataFrame
    df = pd.DataFrame([req.dict()])

    # one-hot encode
    df_encoded = pd.get_dummies(df)

    # alinha colunas com o modelo
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    # predição
    try:
        pred = modelo.predict(df_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    return {"prediction": int(pred[0])}
