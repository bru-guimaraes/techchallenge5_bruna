import os
import sys
import joblib
import yaml
import pandas as pd
import json
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

    # 3) Identifica colunas categóricas e numéricas
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    # o resto (bool, int, float) ficará em remainder

    # 4) Pipeline: OneHot em todas as categóricas + RandomForest
    ct = ColumnTransformer([
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ),
    ], remainder="passthrough")  # mantém booleans e numéricas

    pipeline = Pipeline([
        ("preproc", ct),
        ("clf", RandomForestClassifier(**rf_params, random_state=rnd_state))
    ])

    # 5) Treina no dataset completo
    pipeline.fit(X, y)

    # 6) Extrai nomes de todas as features após pré‑processamento
    preproc: ColumnTransformer = pipeline.named_steps["preproc"]
    # gera array com nomes de todas as colunas transformadas + remainder
    feat_names = preproc.get_feature_names_out(input_features=X.columns)
    # garante que não sobrescreva o target por engano
    feat_list = [f for f in feat_names if f != target_col]

    # 7) Salva features.json
    features_path = Path(os.getenv("FEATURES_JSON_PATH", "model/features.json"))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feat_list, f, ensure_ascii=False, indent=2)
    print(f"✅ features.json gerado em {features_path}")

    # 8) Salva pipeline
    model_path = os.getenv("PATH_MODEL", "model/pipeline.joblib")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"✅ Pipeline salvo em {model_path}")


if __name__ == "__main__":
    main()
