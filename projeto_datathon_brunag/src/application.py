import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ---------------------------
# 1) Definições de Pydantic
# ---------------------------

class Profile(BaseModel):
    area_atuacao: str           = Field(..., example="TI - Desenvolvimento")
    cliente: str                = Field(..., example="Empresa X")
    conhecimentos_tecnicos: str = Field(..., example="Python, AWS, Docker")
    eh_sap: bool                = Field(..., example=False)
    idioma_requerido: str       = Field(..., example="Inglês")
    nivel_academico: str        = Field(..., example="Superior")
    nivel_espanhol: str         = Field(..., example="Básico")
    nivel_ingles: str           = Field(..., example="Avançado")
    nivel_profissional: str     = Field(..., example="Sênior")

class CandidateJobItem(BaseModel):
    candidate: Profile = Field(..., description="Dados do candidato")
    job:       Profile = Field(..., description="Dados da vaga")

class PredictRequest(BaseModel):
    data: List[CandidateJobItem] = Field(
        ..., 
        description="Lista de pares `candidate` + `job`"
    )

class PredictResponseItem(BaseModel):
    status:    str   = Field(..., description="‘aprovado’ ou ‘reprovado’")
    score:     float = Field(..., description="Probabilidade bruta (0–1)")
    threshold: float = Field(..., description="Valor de corte usado")

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


# ---------------------------
# 2) Carrega pipeline
# ---------------------------

MODEL_PATH = os.getenv("PATH_MODEL", "model/pipeline.joblib")
try:
    pipeline = joblib.load(MODEL_PATH)
    expected_cols = list(pipeline.feature_names_in_)
except Exception as e:
    raise RuntimeError(f"Não foi possível carregar o modelo: {e}")


# ---------------------------
# 3) FastAPI
# ---------------------------

app = FastAPI(title="API de Previsão de Contratação")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Monta DataFrame a partir de cada par (candidate + job)
    rows = []
    for item in req.data:
        c = item.candidate.dict()
        j = item.job.dict()
        merged = {**c, **j}
        rows.append(merged)
    df = pd.DataFrame(rows)

    # 2) Preencher colunas faltantes com valor neutro
    for col in expected_cols:
        if col not in df.columns:
            # se for booleano, poderia inferir tipo, mas vamos usar 0/False
            df[col] = 0

    # 3) Cortar colunas extras e reordenar
    df = df[expected_cols]

    # 4) Chama o pipeline
    try:
        probs = pipeline.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"erro ao predizer: {e}")

    # 5) Define threshold
    threshold = float(os.getenv("PREDICTION_THRESHOLD", 0.5))

    # 6) Monta resposta
    results = []
    for p in probs:
        status = "aprovado" if p >= threshold else "reprovado"
        results.append(
            PredictResponseItem(
                status=status,
                score=round(float(p), 4),
                threshold=threshold
            )
        )

    return PredictResponse(results=results)
