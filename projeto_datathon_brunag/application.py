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
    with open(features_path, "r") as f:
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

# ——— 0) Root e health ——————————————————————————————————
@app.get(
    "/",
    summary="Página inicial",
    description=(
        "O que é: Endpoint raiz que confirma que a API está pronta.\n"
        "O que resolve: Verifica rapidamente a disponibilidade.\n"
        "Quando usar: Teste manual ou link na documentação."
    )
)
def root():
    return {"mensagem": "API funcionando com sucesso 🚀"}

@app.get(
    "/health",
    summary="Verificação de saúde básica",
    description=(
        "O que é: Retorna status simples de operação.\n"
        "O que resolve: Permite monitorar via checks de health.\n"
        "Quando usar: Orquestradores (K8s, ELB) verificam este endpoint."
    )
)
def health():
    return {"status": "ok"}

@app.get(
    "/health_detailed",
    summary="Verificação de saúde detalhada",
    description=(
        "O que é: Informe de uptime e versão do Python.\n"
        "O que resolve: Ajuda SREs a monitorar e debugar ambiente.\n"
        "Quando usar: Diagnóstico avançado e dashboards de operações."
    )
)
def health_detailed():
    return {
        "status": "ok",
        "uptime_seconds": time.time() - START_TIME,
        "python_version": sys.version.split()[0],
    }
# ————————————————————————————————————————————————————————

# ——— 1) /metrics —————————————————————————————————————
@app.get(
    "/metrics",
    summary="Métricas Prometheus",
    description=(
        "O que é: Métricas internas em formato Prometheus.\n"
        "O que resolve: Integração com Prometheus/Grafana.\n"
        "Quando usar: Para coletar estatísticas de uso e performance."
    )
)
def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)
# ————————————————————————————————————————————————————————

# ——— 2) /model_info —————————————————————————————————————
@app.get(
    "/model_info",
    summary="Informações do modelo",
    description=(
        "O que é: Metadados estáticos do modelo (versão, data, acurácia).\n"
        "O que resolve: Documenta o artefato em produção.\n"
        "Quando usar: Auditoria e compliance de ML ops."
    )
)
def model_info():
    return {
        "version": app.version,
        "trained_on": "2025-07-01",
        "validation_accuracy": 0.87
    }
# ————————————————————————————————————————————————————————

# ——— 3) /features —————————————————————————————————————
@app.get(
    "/features",
    summary="Lista de features",
    description=(
        "O que é: Exibe as variáveis de entrada esperadas.\n"
        "O que resolve: Ajuda integradores a construir payload correto.\n"
        "Quando usar: Antes de enviar dados para predição."
    )
)
def features():
    return {"features": feature_names}
# ————————————————————————————————————————————————————————

# ——— 4) /predict —————————————————————————————————————
@app.post(
    "/predict",
    response_model=PredictProbaResponse,
    summary="Decisão de contratação (sim/não)",
    description=(
        "O que é: Classifica candidato como 1 (match) ou 0 (no match).\n"
        "O que resolve: Filtra rapidamente candidatos para seleção.\n"
        "Quando usar: Implementar regras automáticas de triagem."
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
    pred = int(probs[0] >= THRESHOLD)
    return {"prediction": pred, "probability": float(probs[0])}
# ————————————————————————————————————————————————————————

# ——— 5) /predict_proba —————————————————————————————————
@app.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    summary="Score de compatibilidade (0–1)",
    description=(
        "O que é: Retorna probabilidade bruta de match.\n"
        "O que resolve: Permite classificar por ranking de confiança.\n"
        "Quando usar: Análise de candidatos em ordem decrescente de score."
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
    return {"prediction": int(probs[0] >= THRESHOLD), "probability": float(probs[0])}
# ————————————————————————————————————————————————————————

# ——— 6) /batch_predict —————————————————————————————————
@app.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    summary="Predição em lote",
    description=(
        "O que é: Recebe lista de candidatos e devolve predições em massa.\n"
        "O que resolve: Processa múltiplos registros de forma eficiente.\n"
        "Quando usar: Importação em lote de planilhas ou pipelines de dados."
    )
)
def batch_predict(req: BatchPredictRequest):
    df = pd.DataFrame([i.dict() for i in req.inputs])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {e}")
    results = [
        PredictProbaResponse(prediction=int(p >= THRESHOLD), probability=float(p))
        for p in probs
    ]
    return {"results": results}
# ————————————————————————————————————————————————————————

# ——— 7) /explain —————————————————————————————————————
@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explicação de decisão via SHAP",
    description=(
        "O que é: Detalha contribuição de cada feature para o resultado.\n"
        "O que resolve: Transparência e confiança no critério de seleção.\n"
        "Quando usar: Auditoria de decisões ou feedback aos gestores."
    )
)
def explain(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    df_enc = pd.get_dummies(df)
    df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
    try:
        probs = modelo.predict_proba(df_aligned)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
    pred = int(probs[0] >= THRESHOLD)
    shap_vals = explainer.shap_values(df_aligned)[1][0]
    explanation = {feat: float(val) for feat, val in zip(feature_names, shap_vals)}
    return {"prediction": pred, "probability": float(probs[0]), "explanation": explanation}
# ————————————————————————————————————————————————————————

# ——— 8) /global_explain —————————————————————————————————
@app.get(
    "/global_explain",
    summary="Importância global de features",
    description=(
        "O que é: Exibe importância média de cada feature.\n"
        "O que resolve: Identifica atributos mais relevantes do modelo.\n"
        "Quando usar: Para entender perfil ideal de candidato a nível geral."
    )
)
def global_explain():
    try:
        importances = modelo.feature_importances_
        return {"global_importance": dict(zip(feature_names, importances.tolist()))}
    except AttributeError:
        sample = pd.DataFrame([dict.fromkeys(feature_names, 0)])
        vals = explainer.shap_values(sample)[1][0]
        return {"global_importance": dict(zip(feature_names, map(abs, vals)))}
# ————————————————————————————————————————————————————————

# ——— 9) /compare —————————————————————————————————————
@app.post(
    "/compare",
    response_model=CompareResponse,
    summary="Comparar dois candidatos",
    description=(
        "O que é: Compara predições, scores e diferenças SHAP.\n"
        "O que resolve: Ajuda a escolher entre duas opções de perfil.\n"
        "Quando usar: Entrevista comparativa ou tie-breaker de seleção."
    )
)
def compare(req: CompareRequest):
    def single(r):
        df = pd.DataFrame([r.dict()])
        df_enc = pd.get_dummies(df)
        df_aligned = df_enc.reindex(columns=feature_names, fill_value=0)
        prob = modelo.predict_proba(df_aligned)[:, 1][0]
        pred = int(prob >= THRESHOLD)
        shap_v = explainer.shap_values(df_aligned)[1][0]
        return pred, prob, dict(zip(feature_names, shap_v))

    a_pred, a_prob, a_shap = single(req.cand_a)
    b_pred, b_prob, b_shap = single(req.cand_b)
    delta = {f: b_shap[f] - a_shap[f] for f in feature_names}
    return {
        "a": {"prediction": a_pred, "probability": a_prob},
        "b": {"prediction": b_pred, "probability": b_prob},
        "delta": delta
    }
# ————————————————————————————————————————————————————————

# ——— 10) /threshold ————————————————————————————————————
@app.get(
    "/threshold",
    summary="Consultar threshold atual",
    description=(
        "O que é: Exibe corte usado para decidir sim/não.\n"
        "O que resolve: Transparência na sensibilidade do filtro.\n"
        "Quando usar: Antes de ajustar ou revisar política de contratação."
    )
)
def get_threshold():
    return {"threshold": THRESHOLD}

@app.post(
    "/threshold",
    summary="Atualizar threshold de decisão",
    description=(
        "O que é: Modifica o valor de corte sem redeploy.\n"
        "O que resolve: Ajusta sensibilidade (mais rigoroso ou mais flexível).\n"
        "Quando usar: Testes A/B ou mudanças de estratégia de seleção."
    )
)
def set_threshold(req: ThresholdRequest):
    global THRESHOLD
    THRESHOLD = req.threshold
    return {"threshold": THRESHOLD}
# ————————————————————————————————————————————————————————

# ——— 11) /feedback ————————————————————————————————————
@app.post(
    "/feedback",
    summary="Registrar feedback",
    description=(
        "O que é: Grava predição vs resultado real para monitoring.\n"
        "O que resolve: Permite avaliar e melhorar performance ao longo do tempo.\n"
        "Quando usar: Sempre que tiver o outcome definitivo do candidato."
    )
)
def feedback(fb: FeedbackRequest):
    record = fb.dict()
    log_path = os.path.join(os.path.dirname(PATH_MODEL), "feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}
# ————————————————————————————————————————————————————————

# ——— 12) /historical_data —————————————————————————————
@app.post(
    "/historical_data",
    summary="Upload de dados históricos",
    description=(
        "O que é: Recebe CSV com histórico de entrevistas e outcomes.\n"
        "O que resolve: Facilita análise retroativa e novos treinamentos.\n"
        "Quando usar: Importação em massa de dados já consolidados."
    )
)
async def upload_historical(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "historical_data.csv")
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "saved", "path": path}
# ————————————————————————————————————————————————————————
