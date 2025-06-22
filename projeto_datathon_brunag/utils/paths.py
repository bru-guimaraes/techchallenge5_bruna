from pathlib import Path

# Diretório raiz do projeto (dois níveis acima deste arquivo)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Diretórios de dados e modelos
DATA_DIR = ROOT_DIR / "data" / "parquet"
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos Parquet
PATH_PARQUET_APPLICANTS = DATA_DIR / "applicants" / "applicants.parquet"
PATH_PARQUET_PROSPECTS = DATA_DIR / "prospects" / "prospects.parquet"
PATH_PARQUET_VAGAS     = DATA_DIR / "vagas" / "vagas.parquet"

# Caminhos de saída do modelo e (opcional) encoder de rótulos
PATH_MODEL = MODEL_DIR / "modelo_classificador.pkl"
PATH_LE    = MODEL_DIR / "label_encoder_contratado.pkl"
