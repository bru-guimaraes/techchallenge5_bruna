import pandas as pd

def mesclar_dataframes(candidatos: pd.DataFrame,
                       prospects: pd.DataFrame,
                       vagas: pd.DataFrame) -> pd.DataFrame:
    """Mescla candidatos, prospects e vagas retornando DataFrame unificado."""
    # 0) Se não há candidatos ou prospects, retorna vazio com colunas de vagas
    if candidatos.empty or prospects.empty:
        return pd.DataFrame(columns=vagas.columns)

    # 1) Mescla candidatos ↔ prospects
    comuns = set(candidatos.columns) & set(prospects.columns)
    chave = list(comuns)[0] if comuns else "codigo_profissional"
    try:
        df = candidatos.merge(prospects, on=chave, how="inner")
    except KeyError:
        return pd.DataFrame(columns=vagas.columns)

    if df.empty:
        return pd.DataFrame(columns=vagas.columns)

    # 2) Mescla com vagas
    comuns = set(df.columns) & set(vagas.columns)
    if comuns:
        chave2 = list(comuns)[0]
        return df.merge(vagas, on=chave2, how="inner")

    # Sem coluna comum, retorna vazio
    return pd.DataFrame(columns=vagas.columns)
