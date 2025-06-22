import pandas as pd

def merge_dataframes(applicants: pd.DataFrame, prospects: pd.DataFrame, vagas: pd.DataFrame) -> pd.DataFrame:
    print("Mesclando prospects com applicants...")
    df = prospects.merge(
        applicants,
        left_on='codigo',
        right_on='codigo_profissional',
        how='left',
        suffixes=('', '_cand')
    )

    print("Mesclando prospects com vagas...")
    df = df.merge(
        vagas,
        on='vaga_id',  # Correto agora
        how='left',
        suffixes=('', '_vaga')
    )

    print(f"Dados mesclados: {df.shape}")
    return df
