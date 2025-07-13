import pandas as pd

parquet_paths = [
    "data/parquet/applicants/applicants.parquet",
    "data/parquet/prospects/prospects.parquet",
    "data/parquet/vagas/vagas.parquet"
]


for path in parquet_paths:
    print(f"\n==== {path} ====")
    try:
        df = pd.read_parquet(path)
        print("Colunas:", list(df.columns))
        print(df.head(2))
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")