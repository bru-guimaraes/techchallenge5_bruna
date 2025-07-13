import pandas as pd

def merge_dataframes(applicants: pd.DataFrame,
                     prospects: pd.DataFrame,
                     vagas: pd.DataFrame) -> pd.DataFrame:
    """
    Mescla três DataFrames em duas etapas:
      1) prospects ↔ applicants:
         - Normaliza applicants.id -> applicants.applicant_id.
         - Tenta qualquer coluna comum.
         - Se não, tenta:
             a) applicant_id ↔ codigo_profissional
             b) codigo ↔ codigo_profissional
             c) codigo_profissional ↔ codigo_profissional
      2) resultado ↔ vagas:
         - Tenta qualquer coluna comum.
         - Se não, tenta prospect_id ↔ prospect_id.
         - Se não, tenta vaga_id ↔ vaga_id.
    """

    # 0) Normalização: id → applicant_id
    if 'id' in applicants.columns and 'applicant_id' not in applicants.columns:
        applicants = applicants.rename(columns={'id': 'applicant_id'})

    # --- 1) prospects ↔ applicants ---
    common = set(prospects.columns) & set(applicants.columns)
    if common:
        left_key = right_key = common.pop()
    else:
        # a) applicant_id ↔ codigo_profissional
        if 'applicant_id' in prospects.columns and 'codigo_profissional' in applicants.columns:
            left_key, right_key = 'applicant_id', 'codigo_profissional'
        # b) codigo ↔ codigo_profissional
        elif 'codigo' in prospects.columns and 'codigo_profissional' in applicants.columns:
            left_key, right_key = 'codigo', 'codigo_profissional'
        # c) codigo_profissional ↔ codigo_profissional
        elif 'codigo_profissional' in prospects.columns and 'codigo_profissional' in applicants.columns:
            left_key = right_key = 'codigo_profissional'
        else:
            raise KeyError(
                f"Nenhuma chave para merge prospects↔applicants:\n"
                f"prospects cols={list(prospects.columns)}\n"
                f"applicants cols={list(applicants.columns)}"
            )

    print(f"Mesclando prospects ({left_key}) com applicants ({right_key})…")
    df = prospects.merge(
        applicants,
        left_on=left_key,
        right_on=right_key,
        how='left',
        suffixes=("", "_cand")
    )

    # --- 2) resultado ↔ vagas ---
    common2 = set(df.columns) & set(vagas.columns)
    if common2:
        merge_key = common2.pop()
    elif 'prospect_id' in df.columns and 'prospect_id' in vagas.columns:
        merge_key = 'prospect_id'
    elif 'vaga_id' in df.columns and 'vaga_id' in vagas.columns:
        merge_key = 'vaga_id'
    else:
        raise KeyError(
            f"Nenhuma chave para merge resultado↔vagas:\n"
            f"df cols={list(df.columns)}\n"
            f"vagas cols={list(vagas.columns)}"
        )

    print(f"Mesclando resultado ({merge_key}) com vagas…")
    df = df.merge(
        vagas,
        on=merge_key,
        how='left',
        suffixes=("", "_vaga")
    )

    print(f"Dados mesclados: {df.shape}")
    return df
