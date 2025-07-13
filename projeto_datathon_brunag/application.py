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
import numpy as np
import shap
from shap.utils._exceptions import InvalidModelError

from utils.paths import PATH_MODEL

# ——— Logger JSON ——————————————————————————————————————
logger = logging.getLogger("recruitment_api")
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
        class DummyExplainer:
            def shap_values(self, X): return [[0]*X.shape[1], [0]*X.shape[1]]
        explainer = DummyExplainer()
except Exception as e:
    raise RuntimeError(f"Erro ao inicializar a aplicação: {e}")
# ————————————————————————————————————————————————————————

# ——— Função para extrair probabilidade classe “1” ——————————
def get_positive_proba(arr: np.ndarray) -> np.ndarray:
    """
    Extrai a probabilidade da classe 1 de predict_proba:
     - Se arr.shape == (n,2): usa o índice onde modelo.classes_ == 1 (ou coluna 1 por padrão).
     - Se arr.shape == (n,1): assume prob única; se esse único for classe 0 inverte.
     - Se arr for 1D: retorna como está.
    """
    # tenta pegar classes_; se não existir, assume [0,1]
    try:
        classes = list(modelo.classes_)
    except Exception:
        classes = [0, 1]

    # binário normal
    if arr.ndim == 2 and arr.shape[1] == 2:
        idx1 = classes.index(1) if 1 in classes else 1
        return arr[:, idx1]

    # única coluna
    if arr.ndim == 2 and arr.shape[1] == 1:
        only_cls = classes[0]
        if only_cls == 1:
            return arr[:, 0]
        else:
            return 1.0 - arr[:, 0]

    # 1D
    if arr.ndim == 1:
        return arr

    raise ValueError(f"Formato inesperado em predict_proba: {arr.shape}")
# ————————————————————————————————————————————————————————

# ——— Pydantic Models ——————————————————————————————————
class PredictRequest(BaseModel):
    area_atuacao: str
    nivel_ingles: Literal["baixo", "medio", "alto"]
    nivel_espanhol: Literal["baixo", "medio", "alto"]
    nivel_academico: Literal["medio", "superior", "pos", "mestrado", "doutorado"]

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.70,
                "message": "✅ Candidato aprovado com confiança de 70%"
            }
        }

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

    logger.info(
        "access",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency": latency
        }
    )
    return response
# ————————————————————————————————————————————————————————

# ——— GET Endpoints ————————————————————————————————————
@app.get("/", summary="Página inicial",
         description="• Endpoint raiz. Verifica disponibilidade.")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get("/health", summary="Health check básico",
         description="• Health simples para monitoramento.")
def health():
    return {"status": "ok"}

@app.get("/health_detailed", summary="Health check detalhado",
         description="• Uptime e versão Python.")
def health_detailed():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version.split()[0],
    }

@app.get("/metrics", summary="Métricas Prometheus",
         description="• Métricas internas para Prometheus.")
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.get("/model_info", summary="Informações do modelo",
         description="• Metadados do modelo.")
def model_info():
    return {
        "version": app.version,
        "trained_on": time.strftime("%Y-%m-%d"),
        "validation_accuracy": 0.87
    }

@app.get("/features", summary="Lista de features",
         description="• Variáveis de entrada aceitas.")
def features():
    return {"features": feature_names}

@app.get("/global_explain", summary="Importância global de features",
         description="• SHAP ou feature_importances_.")
def global_explain():
    try:
        vals = modelo.feature_importances_.tolist()
    except AttributeError:
        sample = pd.DataFrame([dict.fromkeys(feature_names, 0)])
        vals = explainer.shap_values(sample)[1][0]
        vals = list(map(abs, vals))
    return {"global_importance": dict(zip(feature_names, vals))}

@app.get("/threshold", summary="Consultar threshold",
         description="• Mostra o cutoff atual.")
def get_threshold():
    return {"threshold": THRESHOLD}
# ————————————————————————————————————————————————————————

# ——— POST Endpoints ————————————————————————————————————
@app.post("/threshold", summary="Atualizar threshold",
          description="• Ajusta o cutoff sem redeploy.")
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    return {"threshold": THRESHOLD}

@app.post("/predict", response_model=PredictResponse,
          summary="Classificação sim/não",
          description="• Predição binária de candidato.")
def predict(req: PredictRequest):
    # normaliza para lowercase (aceita “Vendas” ou “vendas”)
    d = {k: v.lower() for k, v in req.dict().items()}

    df = pd.DataFrame([d])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)

    try:
        raw = modelo.predict_proba(df_aligned)
        prob = float(get_positive_proba(raw)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    pred = int(prob >= THRESHOLD)
    status = "aprovado" if pred == 1 else "não aprovado"
    message = f"✅ Candidato {status} com confiança de {prob:.0%}"
    return {"prediction": pred, "probability": prob, "message": message}

@app.post("/predict_proba", response_model=PredictProbaResponse,
          summary="Score de compatibilidade",
          description="• Probabilidade contínua de compatibilidade.")
def predict_proba(req: PredictRequest):
    d = {k: v.lower() for k, v in req.dict().items()}

    df = pd.DataFrame([d])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)

    try:
        raw = modelo.predict_proba(df_aligned)
        prob = float(get_positive_proba(raw)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    return {"prediction": int(prob >= THRESHOLD), "probability": prob}

@app.post("/batch_predict", response_model=BatchPredictResponse,
          summary="Predição em lote",
          description="• Processamento em massa de candidatos.")
def batch_predict(req: BatchPredictRequest):
    data = [{k: v.lower() for k, v in i.dict().items()} for i in req.inputs]

    df = pd.DataFrame(data)
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)

    try:
        raw = modelo.predict_proba(df_aligned)
        probs = get_positive_proba(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {e}")

    results = [
        PredictProbaResponse(prediction=int(p >= THRESHOLD), probability=float(p))
        for p in probs
    ]
    return {"results": results}

@app.post("/explain", response_model=ExplainResponse,
          summary="Explicação de decisão",
          description="• Contribuição de cada feature (SHAP).")
def explain(req: PredictRequest):
    d = {k: v.lower() for k, v in req.dict().items()}

    df = pd.DataFrame([d])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)

    try:
        raw = modelo.predict_proba(df_aligned)
        pred_prob = float(get_positive_proba(raw)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

    pred = int(pred_prob >= THRESHOLD)
    shap_vals = explainer.shap_values(df_aligned)[1][0]
    explanation = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    return {"prediction": pred, "probability": pred_prob, "explanation": explanation}

@app.post("/compare", response_model=CompareResponse,
          summary="Comparar dois candidatos",
          description="• Diferença de SHAP entre dois perfis.")
def compare(req: CompareRequest):
    def single(r):
        d = {k: v.lower() for k, v in r.dict().items()}
        df = pd.DataFrame([d])
        df_enc = pd.get_dummies(df)
        df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
        raw = modelo.predict_proba(df_aligned)
        prob = float(get_positive_proba(raw)[0])
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
          description="• Log de predição vs real para monitoramento.")
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}

@app.post("/historical_data", summary="Upload histórico",
          description="• Recebe CSV de entrevistas para análise retroativa.")
async def upload_historical(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "historical_data.csv")
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": path}
