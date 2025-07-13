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
    explainer = shap.TreeExplainer(modelo)
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

# ——— GET ENDPOINTS (lista primeiro) —————————————————————————————————

@app.get("/", summary="Página inicial",
         description="• O que é: Endpoint raiz para checar disponibilidade.\n"
                     "• Quando usar: Teste manual ou link na documentação.\n"
                     "• Que dor resolve: Confirma se a API está online.")
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get("/health", summary="Verificação de saúde básica",
         description="• O que é: Retorna status simples.\n"
                     "• Quando usar: Health checks em orquestradores.\n"
                     "• Que dor resolve: Monitoramento básico de uptime.")
def health():
    return {"status": "ok"}

@app.get("/health_detailed", summary="Verificação de saúde detalhada",
         description="• O que é: Uptime e versão Python.\n"
                     "• Quando usar: Diagnóstico aprofundado.\n"
                     "• Que dor resolve: Debug e observabilidade de ambiente.")
def health_detailed():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version.split()[0],
    }

@app.get("/metrics", summary="Métricas Prometheus",
         description="• O que é: Métricas internas em formato Prometheus.\n"
                     "• Quando usar: Integração com Prometheus/Grafana.\n"
                     "• Que dor resolve: Coleta de estatísticas de uso e performance.")
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.get("/model_info", summary="Informações do modelo",
         description="• O que é: Metadados do artefato ML.\n"
                     "• Quando usar: Auditoria e compliance.\n"
                     "• Que dor resolve: Documentação de versão e acurácia.")
def model_info():
    return {
        "version": app.version,
        "trained_on": "2025-07-01",
        "validation_accuracy": 0.87
    }

@app.get("/features", summary="Lista de features",
         description="• O que é: Variáveis de entrada esperadas.\n"
                     "• Quando usar: Construção de payload de predição.\n"
                     "• Que dor resolve: Minimiza erros de contrato de API.")
def features():
    return {"features": feature_names}

@app.get("/global_explain", summary="Importância global de features",
         description="• O que é: Importância média de cada feature.\n"
                     "• Quando usar: Entender perfil ideal de candidato.\n"
                     "• Que dor resolve: Transparência global do modelo.")
def global_explain():
    try:
        importances = modelo.feature_importances_
        return {"global_importance": dict(zip(feature_names, importances.tolist()))}
    except AttributeError:
        sample = pd.DataFrame([dict.fromkeys(feature_names, 0)])
        vals = explainer.shap_values(sample)[1][0]
        return {"global_importance": dict(zip(feature_names, map(abs, vals)))}

@app.get("/threshold", summary="Consultar threshold atual",
         description="• O que é: Valor de corte para decisão match.\n"
                     "• Quando usar: Antes de ajustar sensibilidade.\n"
                     "• Que dor resolve: Transparência no filtro de candidatos.")
def get_threshold():
    return {"threshold": THRESHOLD}

# ——— POST ENDPOINTS ———————————————————————————————————————

@app.post("/predict", response_model=PredictProbaResponse,
          summary="Decisão de contratação (sim/não)",
          description="• O que é: Classifica candidato como match (1) ou no-match (0).\n"
                      "• Quando usar: Triagem automática inicial.\n"
                      "• Que dor resolve: Agiliza filtragem de grandes volumes.")
def predict(req: PredictRequest):
    df_enc = pd.get_dummies(pd.DataFrame([req.dict()]))
    aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= THRESHOLD)
    return {"prediction": pred, "probability": float(probs[0])}

@app.post("/predict_proba", response_model=PredictProbaResponse,
          summary="Score de compatibilidade (0–1)",
          description="• O que é: Probabilidade bruta de match.\n"
                      "• Quando usar: Ranking por confiança.\n"
                      "• Que dor resolve: Priorização de candidatos por score.")
def predict_proba(req: PredictRequest):
    df_enc = pd.get_dummies(pd.DataFrame([req.dict()]))
    aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    return {"prediction": int(probs[0] >= THRESHOLD), "probability": float(probs[0])}

@app.post("/batch_predict", response_model=BatchPredictResponse,
          summary="Predição em lote",
          description="• O que é: Processa múltiplos candidatos de uma vez.\n"
                      "• Quando usar: Importação em pipelines em massa.\n"
                      "• Que dor resolve: Eficiência em lotes e integrações.")
def batch_predict(req: BatchPredictRequest):
    df_enc = pd.get_dummies(pd.DataFrame([i.dict() for i in req.inputs]))
    aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro em lote: {e}")
    results = [PredictProbaResponse(prediction=int(p >= THRESHOLD), probability=float(p))
               for p in probs]
    return {"results": results}

@app.post("/explain", response_model=ExplainResponse,
          summary="Explicação de decisão via SHAP",
          description="• O que é: Contribuição de cada feature para o resultado.\n"
                      "• Quando usar: Auditoria interna e feedback.\n"
                      "• Que dor resolve: Transparência local do modelo.")
def explain(req: PredictRequest):
    df_enc = pd.get_dummies(pd.DataFrame([req.dict()]))
    aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= THRESHOLD)
    shap_vals = explainer.shap_values(aligned)[1][0]
    explanation = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    return {"prediction": pred, "probability": float(probs[0]), "explanation": explanation}

@app.post("/compare", response_model=CompareResponse,
          summary="Comparar dois candidatos",
          description="• O que é: Compara predições, scores e SHAP delta.\n"
                      "• Quando usar: Decisão entre dois perfis.\n"
                      "• Que dor resolve: Auxílio em tie-breaker de seleção.")
def compare(req: CompareRequest):
    def run(r):
        enc = pd.get_dummies(pd.DataFrame([r.dict()]))
        aligned = enc.reindex(columns=feature_names, fill_value=0)
        prob = modelo.predict_proba(aligned)[:, 1][0]
        pred = int(prob >= THRESHOLD)
        shap_v = explainer.shap_values(aligned)[1][0]
        return pred, prob, dict(zip(feature_names, shap_v))
    a_pred, a_prob, a_shap = run(req.cand_a)
    b_pred, b_prob, b_shap = run(req.cand_b)
    delta = {f: b_shap[f] - a_shap[f] for f in feature_names}
    return {"a": {"prediction": a_pred, "probability": a_prob},
            "b": {"prediction": b_pred, "probability": b_prob},
            "delta": delta}

@app.post("/threshold", summary="Atualizar threshold de decisão",
          description="• O que é: Modifica corte sem redeploy.\n"
                      "• Quando usar: Testes A/B ou ajustes rápidos.\n"
                      "• Que dor resolve: Flexibilidade operacional.")
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    return {"threshold": THRESHOLD}

@app.post("/feedback", summary="Registrar feedback",
          description="• O que é: Grava predição vs resultado real.\n"
                      "• Quando usar: Logging contínuo de outcomes.\n"
                      "• Que dor resolve: Monitoramento e melhoria do modelo.")
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}

@app.post("/historical_data", summary="Upload de dados históricos",
          description="• O que é: Recebe CSV de entrevistas antigas.\n"
                      "• Quando usar: Importação de massa para re-treino.\n"
                      "• Que dor resolve: Centraliza e padroniza seu histórico.")
async def upload_historical(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "historical_data.csv")
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": path}
