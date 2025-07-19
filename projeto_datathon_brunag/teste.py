import pandas as pd

# Carrega arquivos individualmente
df_app = pd.read_parquet("data/parquet/applicants/applicants.parquet")
df_pro = pd.read_parquet("data/parquet/prospects/prospects.parquet")

print("Shape applicants:", df_app.shape)
print("Shape prospects:", df_pro.shape)

print("\nColunas applicants:", df_app.columns.tolist())
print("Colunas prospects:", df_pro.columns.tolist())