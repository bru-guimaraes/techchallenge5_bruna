import os, json

from pathlib import Path

PATH_MODEL = os.getenv("PATH_MODEL", "/app/model/modelo_classificador.pkl")

def load_feature_names():
    f = Path(PATH_MODEL).parent / "features.json"
    return json.loads(f.read_text(encoding="utf-8"))


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

PATH_PARQUET_APPLICANTS = DATA_DIR / "parquet" / "applicants"
PATH_PARQUET_PROSPECTS = DATA_DIR / "parquet" / "prospects"
PATH_PARQUET_VAGAS     = DATA_DIR / "parquet" / "vagas"
