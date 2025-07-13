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
    description="API para predição de compatibilidade de candidatos, explicação via SHAP e envio de feedback.",
    version="1.0.0"
)

# ——— Carrega modelo, features e explainer ———
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    explainer = shap.TreeExplainer(modelo)
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")
# —————————————————————————————————

# ——— Pydantic Models —————————————
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
# —————————————————————————————————

# ——— Middleware para logs + métricas ———
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    REQUEST_LATENCY.labels(request.url.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()

    logger.info('', extra={
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency": latency
    })
    return response
# —————————————————————————————————

# ——— Endpoint de métricas Prometheus ———
@app.get(
    "/metrics",
    summary="Métricas Prometheus",
    description="Retorna métricas em formato Prometheus para monitoramento."
)
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
# —————————————————————————————————

# ——— Health check ————————
@app.get(
    "/health",
    summary="Verificação de saúde",
    description="Confirma se a API está operando corretamente."
)
def health():
    return {"status": "ok"}

@app.get(
    "/",
    summary="Página inicial",
    description="Mensagem de boas-vindas e status da API."
)
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}
# —————————————————————————————————

# ——— 1) /predict —————————————
@app.post(
    "/predict",
    response_model=PredictProbaResponse,
    summary="Decisão de contratação (sim/não)",
    description=(
        "Recebe dados do candidato e devolve:\n"
        "  • `prediction`: 1 se a probabilidade ≥ 50% (caso contrário 0)\n"
        "  • `probability`: valor de 0.0 a 1.0 usado para tomar essa decisão"
    )
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
# —————————————————————————————————

# ——— 2) /predict_proba ——————————
@app.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    summary="Score de compatibilidade (0–1)",
    description=(
        "Recebe dados do candidato e devolve apenas o valor de probabilidade "
        "(raw score de 0.0 a 1.0), sem converter para sim/não.\n\n"
        "Use este endpoint quando quiser:\n"
        "  • Ordenar candidatos pelo score\n"
        "  • Analisar graus de confiança em vez de decisão binária"
    )
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
# —————————————————————————————————

# ——— 3) /explain ——————————————
@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explicação via SHAP",
    description="Gera explicação das contribuições de cada característica usando SHAP."
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

    shap_vals = explainer.shap_values(df_aligned)[1][0]
    explanation = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    return {"prediction": pred, "probability": float(probs[0]), "explanation": explanation}
# —————————————————————————————————

# ——— 4) /feedback ——————————————
@app.post(
    "/feedback",
    summary="Registrar feedback",
    description="Registra a diferença entre predição e resultado real para acompanhamento e retraining."
)
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}
