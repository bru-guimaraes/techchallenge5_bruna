import pandas as pd
from utils.paths import PATH_PARQUET_APPLICANTS, PATH_PARQUET_PROSPECTS, PATH_PARQUET_VAGAS
from utils.data_merger import merge_dataframes

# carrega e mescla como no run_train
applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
df = merge_dataframes(applicants, prospects, vagas)
df['contratado'] = (df['situacao_candidado']=='Contratado pela Decision').astype(int)

# checa a proporção
print(df['contratado'].value_counts(normalize=True))
