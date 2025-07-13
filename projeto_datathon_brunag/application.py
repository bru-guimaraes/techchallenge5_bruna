import time
import logging
from fastapi import FastAPI, HTTPException, Response, Request
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import Literal, Dict
import joblib
import json
import pandas as pd
import os
import shap

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

app = FastAPI(
    title="Decision Recruitment API",
    description="API para predição de match de candidatos, explicação via SHAP e envio de feedback.",
    version="1.0.0"
)

# ——— 1. Carregamento do modelo, features e explainer ———
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    # SHAP explainer
    explainer = shap.TreeExplainer(modelo)
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

class PredictProbaResponse(BaseModel):
    prediction: int
    probability: float

class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    explanation: Dict[str, float]

class FeedbackRequest(BaseModel):
    input: PredictRequest
    prediction: Literal[0, 1]
    actual: Literal[0, 1]
# ————————————————————————————————————————————————

# ——— Middleware para logs + métricas ———
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    # Prometheus
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
@app.get(
    "/metrics",
    summary="Métricas Prometheus",
    description="Retorna métricas para monitoramento (Prometheus format)."
)
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
# ——————————————————————————————————————————————

# ——— Health check / Root ————————
@app.get(
    "/health",
    summary="Health Check",
    description="Verifica se a API está operante."
)
def health():
    return {"status": "ok"}

@app.get(
    "/",
    summary="Root",
    description="Mensagem de boas-vindas e status da API."
)
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}
# ——————————————————————————————————————————————

# ——— Exemplo de endpoint ————————
@app.post(
    "/exemplo",
    summary="Exemplo",
    description="Recebe nome e idade e retorna saudação personalizada."
)
def exemplo_endpoint(dados: EntradaExemplo):
    return {"mensagem": f"Olá, {dados.nome}! Você tem {dados.idade} anos."}
# ——————————————————————————————————————————————

# ——— 1) /predict —————————————
@app.post(
    "/predict",
    response_model=PredictProbaResponse,
    summary="Predição",
    description="Recebe dados do candidato e retorna classe predita e probabilidade de contratação."
)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= 0.5)
    return {"prediction": pred, "probability": float(probs[0])}
# ——————————————————————————————————————————————

# ——— 2) /predict_proba ——————————
@app.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    summary="Probabilidade",
    description="Retorna apenas a classe e a probabilidade sem threshold."
)
def predict_proba(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= 0.5)
    return {"prediction": pred, "probability": float(probs[0])}
# ——————————————————————————————————————————————

# ——— 3) /explain ——————————————
@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explicação SHAP",
    description="Gera explicação de feature contributions usando SHAP."
)
def explain(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= 0.5)

    # SHAP values para classe “1”
    shap_vals = explainer.shap_values(df_aligned)[1][0]
    explanation = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}

    return {"prediction": pred, "probability": float(probs[0]), "explanation": explanation}
# ——————————————————————————————————————————————

# ——— 4) /feedback ——————————————
@app.post(
    "/feedback",
    summary="Feedback",
    description="Registra feedback entre predição e valor real para monitoramento e re-treino."
)
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # append como JSON lines
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}
