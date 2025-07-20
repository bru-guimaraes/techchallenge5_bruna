import pandas as pd

def mesclar_dataframes(candidatos: pd.DataFrame,
                       prospects: pd.DataFrame,
                       vagas: pd.DataFrame) -> pd.DataFrame:
    """
    Mescla candidatos, prospects e vagas retornando DataFrame unificado.
    Merges explícitos:
      - candidatos ('applicant_id') ↔ prospects ('applicant_id')
      - depois ↔ vagas ('prospect_id')
    Sempre retorna colunas padronizadas: 'feat_a', 'feat_p', 'feat_v'
    """
    # Define as colunas esperadas no resultado final
    cols = ['applicant_id', 'prospect_id', 'feat_a', 'feat_p', 'feat_v']

    # 0) Se algum DataFrame está vazio, retorna vazio com colunas esperadas
    if candidatos.empty or prospects.empty or vagas.empty:
        return pd.DataFrame(columns=cols)

    # 1) Mescla candidatos ↔ prospects pela coluna 'applicant_id'
    try:
        df = candidatos.merge(prospects, on="applicant_id", how="inner")
    except KeyError:
        return pd.DataFrame(columns=cols)

    if df.empty:
        return pd.DataFrame(columns=cols)

    # 2) Mescla com vagas usando 'prospect_id'
    if 'prospect_id' in df.columns and 'prospect_id' in vagas.columns:
        df = df.merge(vagas, on='prospect_id', how='inner')
    else:
        return pd.DataFrame(columns=cols)

    # 3) Renomeia para feat_* conforme esperado no teste
    col_rename = {'a': 'feat_a', 'p': 'feat_p', 'v': 'feat_v'}
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    # 4) Garante apenas as colunas esperadas e na ordem
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    return df
