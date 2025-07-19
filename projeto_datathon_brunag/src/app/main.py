from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from utils.paths import (
    PATH_PARQUET_APPLICANTS,
    PATH_PARQUET_PROSPECTS,
    PATH_PARQUET_VAGAS,
    PATH_MODEL
)
from utils.data_merger import merge_dataframes

app = FastAPI()

# Carrega modelo e dados uma única vez
pipeline = joblib.load(PATH_MODEL)
applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)

@app.get("/predict/{candidate_code}/{vaga_id}")
def predict_candidate(candidate_code: str, vaga_id: int):
    """
    Retorna a probabilidade de contratação para um candidato específico em uma vaga.
    """
    df = merge_dataframes(applicants, prospects, vagas)
    row = df[(df['codigo'] == candidate_code) & (df['vaga_id'] == vaga_id)]
    if row.empty:
        raise HTTPException(404, "Candidato ou vaga não encontrados.")
    proba = pipeline.predict_proba(row)[0,1]
    return {"candidate": candidate_code, "vaga_id": vaga_id, "probability": float(proba)}

@app.get("/match/{vaga_id}")
def match_vaga(vaga_id: int, top_k: int = 10):
    """
    Retorna os top_k candidatos para a vaga ordenados pela probabilidade de contratação.
    """
    df = merge_dataframes(applicants, prospects, vagas)
    subset = df[df['vaga_id'] == vaga_id]
    if subset.empty:
        raise HTTPException(404, "Nenhum prospect encontrado para esta vaga.")
    probs = pipeline.predict_proba(subset)[:,1]
    subset = subset.copy()
    subset['score'] = probs
    top = subset.nlargest(top_k, 'score')
    return top[['codigo', 'nome', 'score']].to_dict(orient='records')
