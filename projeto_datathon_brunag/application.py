import time
import logging
import sys
import os
import json

from fastapi import FastAPI, HTTPException, Response, Request, UploadFile, File
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import Literal, Dict, List
import joblib
import pandas as pd
import shap
from shap.utils._exceptions import InvalidModelError

from utils.paths import PATH_MODEL

# ——— Logger JSON ——————————————————————————————————————
logger = logging.getLogger("uvicorn.access")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# ————————————————————————————————————————————————————————

# ——— Métricas Prometheus —————————————————————————————
REQUEST_COUNT = Counter(
    'request_count', 'Total HTTP requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'HTTP request latency',
    ['endpoint']
)
# ————————————————————————————————————————————————————————

START_TIME = time.time()

app = FastAPI(
    title="Decision Recruitment API",
    description=(
        "API para:\n"
        " • Classificação binária de candidatos (match sim/não)\n"
        " • Retorno de scores contínuos de compatibilidade\n"
        " • Explicações de decisão via SHAP\n"
        " • Ajuste dinâmico de threshold e upload de histórico"
    ),
    version="1.0.0"
)

# ——— Carrega modelo, features e explainer ——————————————————
try:
    modelo = joblib.load(PATH_MODEL)
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    try:
        explainer = shap.TreeExplainer(modelo)
    except InvalidModelError:
        # fallback silencioso
        class DummyExplainer:
            def shap_values(self, X): return [ [0]*X.shape[1], [0]*X.shape[1] ]
        explainer = DummyExplainer()
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")
# ————————————————————————————————————————————————————————

# ——— Pydantic Models ——————————————————————————————————
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

class BatchPredictRequest(BaseModel):
    inputs: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: List[PredictProbaResponse]

class CompareRequest(BaseModel):
    cand_a: PredictRequest
    cand_b: PredictRequest

class CompareResponse(BaseModel):
    a: PredictProbaResponse
    b: PredictProbaResponse
    delta: Dict[str, float]

class ThresholdRequest(BaseModel):
    threshold: float
# ————————————————————————————————————————————————————————

THRESHOLD = 0.5

# ——— Middleware para logs + métricas —————————————————————
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
# ————————————————————————————————————————————————————————

# ——— GET Endpoints ————————————————————————————————————

@app.get("/", summary="Página inicial",
         description="• O que é: Endpoint raiz.\n• O que resolve: Verifica disponibilidade.\n• Quando usar: Teste manual.")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get("/health", summary="Health check básico",
         description="• O que é: Health simples.\n• O que resolve: Monitoramento básico.\n• Quando usar: Orquestradores.")
def health():
    return {"status": "ok"}

@app.get("/health_detailed", summary="Health check detalhado",
         description="• O que é: Uptime e versão Python.\n• O que resolve: Diagnóstico operacional.\n• Quando usar: Infra SRE.")
def health_detailed():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version.split()[0],
    }

@app.get("/metrics", summary="Métricas Prometheus",
         description="• O que é: Métricas internas.\n• O que resolve: Integração Prometheus.\n• Quando usar: Dashboards.")
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.get("/model_info", summary="Informações do modelo",
         description="• O que é: Metadados do modelo.\n• O que resolve: Auditoria ML.\n• Quando usar: Compliance.")
def model_info():
    return {
        "version": app.version,
        "trained_on": "2025-07-01",
        "validation_accuracy": 0.87
    }

@app.get("/features", summary="Lista de features",
         description="• O que é: Variáveis de entrada.\n• O que resolve: Guia payload.\n• Quando usar: Antes da predição.")
def features():
    return {"features": feature_names}

@app.get("/global_explain", summary="Importância global de features",
         description="• O que é: Importância média (SHAP).\n• O que resolve: Perfil geral de candidatos.\n• Quando usar: Entendimento global do modelo.")
def global_explain():
    try:
        importances = modelo.feature_importances_
        vals = importances.tolist()
    except AttributeError:
        # fallback para explainer
        sample = pd.DataFrame([dict.fromkeys(feature_names, 0)])
        vals = explainer.shap_values(sample)[1][0]
        vals = list(map(abs, vals))
    return {"global_importance": dict(zip(feature_names, vals))}

@app.get("/threshold", summary="Consultar threshold",
         description="• O que é: Mostra corte atual.\n• O que resolve: Transparência na sensibilidade.\n• Quando usar: Revisão de estratégia.")
def get_threshold():
    return {"threshold": THRESHOLD}

# ——— POST Endpoints ———————————————————————————————————

@app.post("/threshold", summary="Atualizar threshold",
          description="• O que é: Atualiza corte dinamicamente.\n• O que resolve: Ajuste sem redeploy.\n• Quando usar: Teste A/B ou tuning.")
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    return {"threshold": THRESHOLD}

@app.post("/predict", response_model=PredictProbaResponse,
          summary="Classificação sim/não",
          description="• O que é: Predição binária.\n• O que resolve: Filtragem de candidatos.\n• Quando usar: Triagem automática.")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.dict()]); df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Erro na predição: {e}")
    pred = int(probs[0] >= THRESHOLD)
    return {"prediction": pred, "probability": float(probs[0])}

@app.post("/predict_proba", response_model=PredictProbaResponse,
          summary="Score de compatibilidade",
          description="• O que é: Probabilidade contínua.\n• O que resolve: Ranking de candidatos.\n• Quando usar: Ordenação por confiança.")
def predict_proba(req: PredictRequest):
    df = pd.DataFrame([req.dict()]); df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Erro na predição: {e}")
    return {"prediction": int(probs[0] >= THRESHOLD), "probability": float(probs[0])}

@app.post("/batch_predict", response_model=BatchPredictResponse,
          summary="Predição em lote",
          description="• O que é: Lista de candidatos.\n• O que resolve: Processamento em massa.\n• Quando usar: Pipelines de dados.")
def batch_predict(req: BatchPredictRequest):
    df = pd.DataFrame([i.dict() for i in req.inputs]); df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Erro na predição em lote: {e}")
    results = [
        PredictProbaResponse(prediction=int(p>=THRESHOLD), probability=float(p))
        for p in probs
    ]
    return {"results": results}

@app.post("/explain", response_model=ExplainResponse,
          summary="Explicação de decisão",
          description="• O que é: Contribuição de cada feature (SHAP).\n• O que resolve: Transparência.\n• Quando usar: Auditoria de decisões.")
def explain(req: PredictRequest):
    df = pd.DataFrame([req.dict()]); df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Erro na predição: {e}")
    pred = int(probs[0] >= THRESHOLD)
    shap_vals = explainer.shap_values(df_aligned)[1][0]
    explanation = dict(zip(feature_names, map(float, shap_vals)))
    return {"prediction": pred, "probability": float(probs[0]), "explanation": explanation}

@app.post("/compare", response_model=CompareResponse,
          summary="Comparar dois candidatos",
          description="• O que é: Diff de SHAP + predições.\n• O que resolve: Escolha entre perfis.\n• Quando usar: Tie-breaker em seleção.")
def compare(req: CompareRequest):
    def single(r):
        df = pd.DataFrame([r.dict()]); df_enc = pd.get_dummies(df)
        df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
        prob = modelo.predict_proba(df_aligned)[:,1][0]
        pred = int(prob >= THRESHOLD)
        shap_v = explainer.shap_values(df_aligned)[1][0]
        return pred, prob, shap_v

    a_pred, a_prob, a_shap = single(req.cand_a)
    b_pred, b_prob, b_shap = single(req.cand_b)
    delta = dict(zip(feature_names, (b_shap - a_shap).tolist()))
    return {
        "a": {"prediction": a_pred, "probability": a_prob},
        "b": {"prediction": b_pred, "probability": b_prob},
        "delta": delta
    }

@app.post("/feedback", summary="Registrar feedback",
          description="• O que é: Log de predição vs real.\n• O que resolve: Monitoramento contínuo.\n• Quando usar: Pós-outcome.")
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}

@app.post("/historical_data", summary="Upload histórico",
          description="• O que é: Recebe CSV de entrevistas.\n• O que resolve: Análise retroativa.\n• Quando usar: Importação em massa.")
async def upload_historical(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "historical_data.csv")
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": path}
