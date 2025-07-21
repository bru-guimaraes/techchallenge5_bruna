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

class MatchScoreMixin:
    """Mixin para computar uma coluna match_score antes do pipeline."""
    @staticmethod
    def compute_match_score(df: pd.DataFrame) -> pd.Series:
        keys = [
            "area_atuacao",
            "cliente",
            "eh_sap",
            "idioma_requerido",
            "nivel_academico",
            "nivel_espanhol",
            "nivel_ingles",
            "nivel_profissional"
        ]
        def score(row):
            count = sum(
                row.get(f"candidate_{k}") == row.get(f"job_{k}")
                for k in keys
            )
            return count / len(keys)
        return df.apply(score, axis=1)

def main():
    # 1) Carrega configurações do YAML
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    path_unificado = (
        next((a for a in sys.argv[1:] if a.endswith(".parquet")), None)
        or cfg["paths"]["parquet_treino_unificado"]
    )
    target_col = cfg["features"]["target_column"]
    rf_params  = cfg["model"]["random_forest"].copy()
    rnd_state  = cfg.get("train", {}).get("random_state", 42)
    rf_params.pop("random_state", None)

    # 2) Carrega dados unificados de treino
    df = carregar_dados_para_treino(path_unificado)
    if target_col not in df.columns:
        raise KeyError(f"Coluna alvo '{target_col}' não existe em {path_unificado}")

    # 3) Balanceamento das classes (undersampling)
    class_counts = df[target_col].value_counts()
    min_class = class_counts.idxmin()
    min_count = class_counts.min()
    df_balanced = pd.concat([
        df[df[target_col] == c].sample(n=min_count, random_state=rnd_state)
        for c in class_counts.index
    ], ignore_index=True)

    # 4) Calcula match_score no dataframe balanceado
    df_balanced["match_score"] = MatchScoreMixin.compute_match_score(df_balanced)

    # 5) Define explicitamente as features a serem usadas pelo modelo
    FEATURES = [
        "cliente",
        "nivel_profissional",
        "idioma_requerido",
        "area_atuacao",
        "nivel_ingles",
        "nivel_espanhol",
        "nivel_academico",
        "conhecimentos_tecnicos",
        "eh_sap",        # Mantenha apenas se fará sentido usar como feature
        "match_score"    # Mantenha apenas se usará nas massas de teste
    ]

    # 6) Cria X e y, garantindo que só as features corretas vão para o modelo
    X = df_balanced[FEATURES].copy()
    y = df_balanced[target_col]

    # 7) Identifica colunas categóricas para o OneHotEncoder
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # 8) Define pipeline: OneHot em categóricas + RandomForest
    ct = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols
            ),
        ],
        remainder="passthrough"  # Passa as demais (numéricas/bool) direto
    )

    pipeline = Pipeline([
        ("preproc", ct),
        ("clf", RandomForestClassifier(**rf_params, random_state=rnd_state))
    ])

    # 9) Treina o pipeline
    pipeline.fit(X, y)

    # 10) Extrai nomes de features já transformadas (OHE + passthrough)
    preproc: ColumnTransformer = pipeline.named_steps["preproc"]
    feat_names = preproc.get_feature_names_out(input_features=X.columns)
    # >>>>>>>>>>>> Correção fundamental <<<<<<<<<<<<<<
    # Remove o prefixo 'remainder__' das features que são passthrough
    feat_list = [f.replace("remainder__", "") for f in feat_names if f != target_col]

    # 11) Salva features.json
    features_path = Path(os.getenv("FEATURES_JSON_PATH", "model/features.json"))
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feat_list, f, ensure_ascii=False, indent=2)
    print(f"✅ features.json gerado em {features_path}")

    # 12) Salva pipeline treinado
    model_path = os.getenv("PATH_MODEL", "model/pipeline.joblib")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"✅ Pipeline salvo em {model_path}")

if __name__ == "__main__":
    main()
