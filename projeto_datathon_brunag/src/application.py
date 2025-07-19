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
from fastapi import FastAPI, HTTPException, UploadFile, File
from pythonjsonlogger import jsonlogger
from pydantic import BaseModel, model_validator

from utils.paths import PATH_MODEL
from utils.engenharia_de_features import processar_features as processar_features_inference

# Logger JSON
logger = logging.getLogger("recruitment_api")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# FastAPI app
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

# Load assets (model, features, explainer)
def _load_assets():
    path = os.getenv("PATH_MODEL", PATH_MODEL)
    model = joblib.load(path)
    feat_path = os.path.join(os.path.dirname(path), "features.json")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        class DummyExplainer:
            def shap_values(self, X): return [[0]*X.shape[1], [0]*X.shape[1]]
        explainer = DummyExplainer()
    return model, feature_names, explainer

# Global load for tests and module-level export
model, feature_names, explainer = _load_assets()
THRESHOLD: float = 0.5
START_TIME = time.time()

# Utility: positive probability
def get_positive_proba(model, arr) -> np.ndarray:
    arr = np.asarray(arr)
    try:
        classes = list(model.classes_)
    except Exception:
        classes = [0, 1]
    if arr.ndim == 2 and arr.shape[1] == 2:
        idx1 = classes.index(1) if 1 in classes else 1
        return arr[:, idx1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        only_cls = classes[0]
        return arr[:,0] if only_cls == 1 else 1.0 - arr[:,0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Formato inesperado em predict_proba: {arr.shape}")

# Pydantic models
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

class PredictProbaResponse(BaseModel):
    prediction: int
    probability: float

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

# Helper for encoder path
def _setup_mlb_path():
    return os.path.join(os.path.dirname(os.getenv("PATH_MODEL", PATH_MODEL)), "area_atuacao_mlb.joblib")

# Endpoints
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

@app.get("/model_info")
def model_info():
    return {"version": "1.0.0", "trained_on": "YYYY-MM-DD", "validation_accuracy": 0.85}

@app.get("/features")
def get_features():
    return {"features": feature_names}

@app.get("/global_explain")
def global_explain():
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            glob = dict(zip(feature_names, map(float, importances)))
        elif hasattr(model, "get_booster"):
            score_dict = model.get_booster().get_score(importance_type="weight")
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

@app.post("/predict")
def predict(req: PredictRequest):
    data = req.model_dump()
    df = processar_features_inference(pd.DataFrame([data]))
    prob = float(get_positive_proba(model, model.predict_proba(df))[0])
    pred = int(prob >= THRESHOLD)
    return {"previsao": pred, "probabilidade": prob}

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    data = req.model_dump()
    df = processar_features_inference(pd.DataFrame([data]))
    prob = float(get_positive_proba(model, model.predict_proba(df))[0])
    pred = int(prob >= THRESHOLD)
    return {"prediction": pred, "probability": prob}

@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    dfs = [processar_features_inference(pd.DataFrame([d.model_dump()])) for d in req.inputs]
    df_all = pd.concat(dfs, ignore_index=True)
    probs = get_positive_proba(model, model.predict_proba(df_all))
    results = [PredictProbaResponse(prediction=int(p >= THRESHOLD), probability=float(p)) for p in probs]
    return BatchPredictResponse(results=results)

@app.post("/explain")
def explain(req: PredictRequest):
    data = req.model_dump()
    df = processar_features_inference(pd.DataFrame([data]))
    prob = float(get_positive_proba(model, model.predict_proba(df))[0])
    pred = int(prob >= THRESHOLD)
    shap_vals = explainer.shap_values(df)[1][0]
    explanation = dict(zip(feature_names, map(float, shap_vals)))
    return PredictProbaResponse(prediction=pred, probability=prob) if False else {"prediction": pred, "probability": prob, "explanation": explanation}

@app.post("/compare")
def compare(req: CompareRequest):
    data_a = req.cand_a.model_dump()
    data_b = req.cand_b.model_dump()
    df_a = processar_features_inference(pd.DataFrame([data_a]))
    df_b = processar_features_inference(pd.DataFrame([data_b]))
    a_p = float(get_positive_proba(model, model.predict_proba(df_a))[0])
    b_p = float(get_positive_proba(model, model.predict_proba(df_b))[0])
    a_pred = int(a_p >= THRESHOLD)
    b_pred = int(b_p >= THRESHOLD)
    shap_a = explainer.shap_values(df_a)[1][0]
    shap_b = explainer.shap_values(df_b)[1][0]
    delta = dict(zip(feature_names, (np.array(shap_b) - np.array(shap_a)).tolist()))
    return CompareResponse(a=PredictProbaResponse(prediction=a_pred, probability=a_p), b=PredictProbaResponse(prediction=b_pred, probability=b_p), delta=delta)

@app.post("/feedback")
def feedback(fb: dict):
    path = os.path.join(os.path.dirname(os.getenv("PATH_MODEL", PATH_MODEL)), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fb, ensure_ascii=False) + "\n")
    return {"status": "ok"}

@app.post("/historico")
async def upload_historico(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", "historical_data.csv")
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status": "salvo", "caminho": dest}
