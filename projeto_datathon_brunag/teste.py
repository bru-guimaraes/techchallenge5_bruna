import pandas as pd

applicants = pd.read_parquet("data/parquet/applicants/applicants.parquet")
prospects = pd.read_parquet("data/parquet/prospects/prospects.parquet")
vagas     = pd.read_parquet("data/parquet/vagas/vagas.parquet")

print("Colunas em 'applicants':", applicants.columns.tolist())
print("Colunas em 'prospects':", prospects.columns.tolist())
print("Colunas em 'vagas':",     vagas.columns.tolist())
