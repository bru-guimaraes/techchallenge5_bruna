import pandas as pd

# Carrega os arquivos Parquet
applicants = pd.read_parquet("data/parquet/applicants/applicants.parquet")
prospects = pd.read_parquet("data/parquet/prospects/prospects.parquet")
vagas = pd.read_parquet("data/parquet/vagas/vagas.parquet")  # Adicionado para inspecionar vagas

# Exibe colunas dos DataFrames carregados
print("📋 Colunas do DataFrame 'applicants':")
print(applicants.columns.tolist())

print("\n📋 Colunas do DataFrame 'prospects':")
print(prospects.columns.tolist())

print("\n📋 Colunas do DataFrame 'vagas':")
print(vagas.columns.tolist())  # Verifica qual coluna usar no merge
