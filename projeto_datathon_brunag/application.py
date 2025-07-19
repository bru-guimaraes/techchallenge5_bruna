import sys
import time
import logging
import os
import json
from typing import Literal, List, Dict, Optional

import joblib
import pandas as pd
import numpy as np
import shap
from shap.utils._exceptions import InvalidModelError
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, field_validator, model_validator

from utils.paths import PATH_MODEL
from utils.feature_engineering import processar_features_inference

# ——— Logger JSON ——————————————————————————————————————————————————————————
logger = logging.getLogger("recruitment_api")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ——— Métricas Prometheus ———————————————————————————————————————————————
REQUEST_COUNT   = Counter('request_count', 'Total HTTP requests', ['method','endpoint','http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'HTTP request latency', ['endpoint'])

# ——— Instância FastAPI ———————————————————————————————————————————————
app = FastAPI(
    title="Decision Recruitment API",
    version="1.0.0",
    description=(
        "• Classificação binária de candidatos (match sim/não)\n"
        "• Retorno de scores contínuos\n"
        "• Explicações via SHAP\n"
        "• Ajuste dinâmico de threshold e histórico"
    )
)

# ——— Carrega modelo, features e explainer ————————————————————————————
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

# ——— Função utilitária ————————————————————————————————————————————————
def get_positive_proba(arr: np.ndarray) -> np.ndarray:
    try:
        classes = list(modelo.classes_)
    except:
        classes = [0,1]
    if arr.ndim == 2 and arr.shape[1] == 2:
        idx1 = classes.index(1) if 1 in classes else 1
        return arr[:, idx1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        only_cls = classes[0]
        return arr[:,0] if only_cls == 1 else 1.0 - arr[:,0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Formato inesperado em predict_proba: {arr.shape}")

# ——— Pydantic Models ————————————————————————————————————————————————
class PredictRequest(BaseModel):
    area_atuacao: str
    nivel_ingles: Literal["baixo","medio","alto"]
    nivel_espanhol: Literal["baixo","medio","alto"]
    nivel_academico: Literal["medio","superior","pos","mestrado","doutorado"]
    areas_atuacao: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def set_areas(cls, values):
        if not values.get("areas_atuacao"):
            values["areas_atuacao"] = values.get("area_atuacao")
        return values

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    message: str

class PredictProbaResponse(BaseModel):
    prediction: int
    probability: float

class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    explanation: Dict[str, float]

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

# ——— Threshold e tempo de uptime ————————————————————————————————————
THRESHOLD = 0.5
START_TIME = time.time()

# ——— Middleware para métricas ——————————————————————————————————————
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_LATENCY.labels(request.url.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    logger.info("access", extra={
        "method": request.method,
        "path":   request.url.path,
        "status": response.status_code,
        "latency": latency
    })
    return response

# ——— Endpoints ——————————————————————————————————————————————————————————

@app.get("/")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health_detailed")
def health_detailed():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version
    }

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/model_info")
def model_info():
    return {"version": "1.0.0", "trained_on": "YYYY-MM-DD", "validation_accuracy": 0.85}

@app.get("/features")
def get_features():
    return {"features": feature_names}

@app.get("/global_explain")
def global_explain():
    try:
        if hasattr(modelo, "feature_importances_"):
            importances = modelo.feature_importances_
            glob = dict(zip(feature_names, map(float, importances)))
        elif hasattr(modelo, "get_booster"):
            booster = modelo.get_booster()
            score_dict = booster.get_score(importance_type="weight")
            glob = {feat: float(score_dict.get(feat, 0.0)) for feat in feature_names}
        else:
            glob = {feat: 0.0 for feat in feature_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"global_importance": glob}

@app.get("/threshold")
def get_threshold():
    return {"threshold": THRESHOLD}

@app.post("/threshold")
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    return {"threshold": THRESHOLD}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        data = req.model_dump()
        mlb_path = os.path.join(os.path.dirname(PATH_MODEL), "area_atuacao_mlb.joblib")
        df = processar_features_inference(data, feature_names, mlb_path)
        prob = float(get_positive_proba(modelo.predict_proba(df))[0])
        pred = int(prob >= THRESHOLD)
        msg = f"✅ Aprovado com confiança de {int(prob*100)}%" if pred else f"❌ Reprovado com confiança de {int((1-prob)*100)}%"
        return PredictResponse(prediction=pred, probability=prob, message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(req: PredictRequest):
    try:
        data = req.model_dump()
        mlb_path = os.path.join(os.path.dirname(PATH_MODEL), "area_atuacao_mlb.joblib")
        df = processar_features_inference(data, feature_names, mlb_path)
        prob = float(get_positive_proba(modelo.predict_proba(df))[0])
        pred = int(prob >= THRESHOLD)
        return PredictProbaResponse(prediction=pred, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest):
    try:
        mlb_path = os.path.join(os.path.dirname(PATH_MODEL), "area_atuacao_mlb.joblib")
        dfs = [processar_features_inference(d.model_dump(), feature_names, mlb_path) for d in req.inputs]
        df_all = pd.concat(dfs, ignore_index=True)
        probs = get_positive_proba(modelo.predict_proba(df_all))
        results = [PredictProbaResponse(prediction=int(p >= THRESHOLD), probability=float(p)) for p in probs]
        return BatchPredictResponse(results=results)
    except Exception as e:
        logger.exception("Erro interno em /batch_predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    try:
        data = req.model_dump()
        mlb_path = os.path.join(os.path.dirname(PATH_MODEL), "area_atuacao_mlb.joblib")
        df = processar_features_inference(data, feature_names, mlb_path)
        prob = float(get_positive_proba(modelo.predict_proba(df))[0])
        pred = int(prob >= THRESHOLD)
        shap_vals = explainer.shap_values(df)[1][0]
        explanation = dict(zip(feature_names, map(float, shap_vals)))
        return ExplainResponse(prediction=pred, probability=prob, explanation=explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    try:
        mlb_path = os.path.join(os.path.dirname(PATH_MODEL), "area_atuacao_mlb.joblib")
        def process(d):
            data = d.model_dump()
            df = processar_features_inference(data, feature_names, mlb_path)
            prob = float(get_positive_proba(modelo.predict_proba(df))[0])
            pred = int(prob >= THRESHOLD)
            shap_v = explainer.shap_values(df)[1][0]
            return pred, prob, shap_v

        a_pred, a_prob, a_shap = process(req.cand_a)
        b_pred, b_prob, b_shap = process(req.cand_b)
        delta = dict(zip(feature_names, (b_shap - a_shap).tolist()))
        return CompareResponse(
            a=PredictProbaResponse(prediction=a_pred, probability=a_prob),
            b=PredictProbaResponse(prediction=b_pred, probability=b_prob),
            delta=delta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(fb: dict):
    path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fb, ensure_ascii=False) + "\n")
    return {"status": "ok"}

@app.post("/historical_data")
async def upload_historical(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", "historical_data.csv")
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": dest}
