# src/utils/paths.py

import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

def get_features_json_path() -> str:
    """
    Caminho para o features.json gerado pelo pipeline.
    Usa a variável de ambiente FEATURES_JSON_PATH, se definida,
    ou o arquivo padrão em model/features.json.
    """
    return os.environ.get(
        "FEATURES_JSON_PATH",
        str(BASE_DIR / "model" / "features.json")
    )

def get_pipeline_path() -> str:
    """
    Caminho para o pipeline.joblib gerado pelo treinamento.
    Usa a variável de ambiente PIPELINE_PATH, se definida,
    ou o arquivo padrão em model/pipeline.joblib.
    """
    return os.environ.get(
        "PIPELINE_PATH",
        str(BASE_DIR / "model" / "pipeline.joblib")
    )

# Alias para compatibilidade
def get_model_path() -> str:
    return get_pipeline_path()

def load_feature_names(path: str | None = None) -> list[str]:
    """
    Carrega a lista de features salvas em JSON.
    """
    p = path or get_features_json_path()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# Alias para uso direto em testes
PATH_MODEL = get_pipeline_path()
