"""
utils/carregar_dados_para_treino.py

Carrega Parquet de candidatos e prospects, une e gera a coluna alvo.
"""

import pandas as pd
from pathlib import Path

def carregar_dados_para_treino(
    caminho_candidatos: str = "data/parquet/applicants",
    caminho_prospects: str  = "data/parquet/prospects"
) -> pd.DataFrame:
    """Carrega e retorna o DataFrame pronto para treino.

    Concatena todos os Parquet de candidatos e prospects,
    renomeia colunas, gera coluna `contratado` e filtra nulos.

    Args:
        caminho_candidatos: Pasta com arquivos applicants.parquet.
        caminho_prospects:  Pasta com arquivos prospects.parquet.

    Returns:
        DataFrame sem valores nulos, com coluna `contratado`.
    """
    df_cand = pd.concat(
        [pd.read_parquet(p) for p in Path(caminho_candidatos).glob("*.parquet")]
    )
    df_prop = pd.concat(
        [pd.read_parquet(p) for p in Path(caminho_prospects).glob("*.parquet")]
    )

    if "codigo_profissional" not in df_cand.columns and "codigo" in df_cand.columns:
        df_cand = df_cand.rename(columns={"codigo": "codigo_profissional"})

    df_cand["contratado"] = (
        df_cand["codigo_profissional"]
        .isin(df_prop["codigo_profissional"])
        .astype(int)
    )

    col_uteis = [
        "codigo_profissional",
        "id_vaga",
        "nivel_academico",
        "area_atuacao",
        "remuneracao",
        "nivel_ingles",
        "nivel_espanhol",
        "contratado"
    ]
    df_final = df_cand.merge(df_prop, on="codigo_profissional", how="inner")
    df_final = df_final[[c for c in col_uteis if c in df_final.columns]].dropna()
    df_final["remuneracao"] = pd.to_numeric(df_final["remuneracao"], errors="coerce")
    df_final = df_final.dropna()

    return df_final
