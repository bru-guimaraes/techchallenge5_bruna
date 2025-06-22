import pandas as pd
from pathlib import Path

def carregar_dados_para_treino(
    path_applicants='data/parquet/applicants',
    path_prospects='data/parquet/prospects',
) -> pd.DataFrame:
    # Carrega os arquivos Parquet
    df_applicants = pd.concat([pd.read_parquet(p) for p in Path(path_applicants).glob("*.parquet")])
    df_prospects = pd.concat([pd.read_parquet(p) for p in Path(path_prospects).glob("*.parquet")])

    # Corrige o nome da coluna se necessário
    if "codigo_profissional" not in df_applicants.columns and "codigo" in df_applicants.columns:
        df_applicants = df_applicants.rename(columns={"codigo": "codigo_profissional"})

    # Adiciona coluna de contratação (target)
    df_prospects['contratado'] = df_prospects['situacao_candidado'].str.contains('contratado', case=False, na=False).astype(int)

    # Junta os dados dos candidatos com seus respectivos prospects
    df_final = df_prospects.merge(
        df_applicants,
        how='left',
        left_on='codigo',
        right_on='codigo_profissional'
    )

    # Seleciona as colunas úteis para o modelo
    colunas_uteis = [
        'codigo_profissional',
        'sexo',
        'estado_civil',
        'nivel_academico',
        'area_atuacao',
        'remuneracao',
        'nivel_ingles',
        'nivel_espanhol',
        'contratado'
    ]
    df_final = df_final[[col for col in colunas_uteis if col in df_final.columns]].dropna()

    # Tratamento básico de tipos
    df_final['remuneracao'] = pd.to_numeric(df_final['remuneracao'], errors='coerce')
    df_final = df_final.dropna()

    return df_final
