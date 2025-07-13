import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

PATH_PARQUET_APPLICANTS = DATA_DIR / "parquet" / "applicants"
PATH_PARQUET_PROSPECTS = DATA_DIR / "parquet" / "prospects"
PATH_PARQUET_VAGAS     = DATA_DIR / "parquet" / "vagas"
PATH_MODEL = os.getenv("MODEL_PATH", "/app/model/modelo_classificador.pkl")
# (opcional) PATH_LE = MODEL_DIR / "label_encoder_contratado.pkl"
