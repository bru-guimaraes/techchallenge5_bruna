import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Default (fallback) — usado apenas se não houver VAR de ambiente
DEFAULT_MODEL_PATH = str(BASE_DIR / "model" / "modelo_classificador.joblib")

# Constante (para não quebrar import tests)
PATH_MODEL = DEFAULT_MODEL_PATH

def get_model_path() -> str:
    """
    Retorna o caminho real do modelo, dando preferência
    à variável de ambiente PATH_MODEL se estiver definida.
    """
    return os.getenv("PATH_MODEL") or DEFAULT_MODEL_PATH

def load_feature_names() -> list:
    """
    Carrega a lista de features de features.json
    usando o mesmo diretório de onde saiu o modelo.
    """
    model_path = get_model_path()
    feat_file = Path(model_path).parent / "features.json"
    return json.loads(feat_file.read_text(encoding="utf-8"))
