from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Caminhos dos arquivos Parquet
PATH_PARQUET_APPLICANTS = ROOT_DIR / "data" / "parquet" / "applicants" / "applicants.parquet"
PATH_PARQUET_PROSPECTS = ROOT_DIR / "data" / "parquet" / "prospects" / "prospects.parquet"
PATH_PARQUET_VAGAS = ROOT_DIR / "data" / "parquet" / "vagas" / "vagas.parquet"

# Caminho para salvar o modelo treinado
PATH_MODELO_SAIDA = ROOT_DIR / "model" / "modelo_classificador.pkl"

# Caminho para salvar o LabelEncoder
PATH_ENCODER_SAIDA = ROOT_DIR / "model" / "label_encoder_contratado.pkl"
