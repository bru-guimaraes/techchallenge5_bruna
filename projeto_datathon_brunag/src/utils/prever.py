import os
import joblib
import pandas as pd
import json

# Diretório onde pipeline.joblib e features.json estão gerados
BASE_DIR = os.getenv("PATH_MODEL", "model")
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")

# Carrega pipeline e lista de features
pipeline = joblib.load(PIPELINE_PATH)
with open(FEATURES_PATH, encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

def predict_from_pipeline(df: pd.DataFrame) -> pd.Series:
    """
    Recebe um DataFrame já com as colunas esperadas,
    aplica o pipeline e retorna uma Series com as previsões.
    """
    # verifica se não faltam colunas
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise KeyError(f"columns are missing: {missing}")
    # seleciona só as colunas que o pipeline espera
    X = df[FEATURE_NAMES]
    # retorna Série de ints (0/1)
    return pd.Series(pipeline.predict(X))
