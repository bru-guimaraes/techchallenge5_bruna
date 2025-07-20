import os
import sys
import joblib
import yaml
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils.carregar_dados_para_treino import carregar_dados_para_treino

def main():
    # 1) Carrega configurações
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    path_unificado = (
        next((a for a in sys.argv[1:] if a.endswith(".parquet")), None)
        or cfg["paths"]["parquet_treino_unificado"]
    )
    target_col = cfg["features"]["target_column"]
    rf_params  = cfg["model"]["random_forest"].copy()
    rnd_state  = cfg.get("train", {}).get("random_state", 42)
    rf_params.pop("random_state", None)

    # 2) Carrega dados
    df = carregar_dados_para_treino(path_unificado)
    if target_col not in df.columns:
        raise KeyError(f"Coluna alvo '{target_col}' não existe em {path_unificado}")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3) Todas as colunas object são categóricas
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # 4) Pipeline: OneHot em todas as categóricas + RandomForest
    ct = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ], remainder="passthrough")  # boolean passa "as is"

    pipeline = Pipeline([
        ("preproc", ct),
        ("clf", RandomForestClassifier(**rf_params, random_state=rnd_state))
    ])

    # 5) Treina no dataset completo
    pipeline.fit(X, y)

    # 6) Salva pipeline
    model_dir = os.getenv("PATH_MODEL", "model/pipeline.joblib")
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    joblib.dump(pipeline, model_dir)
    print(f"✅ Pipeline salvo em {model_dir}")

if __name__ == "__main__":
    main()
