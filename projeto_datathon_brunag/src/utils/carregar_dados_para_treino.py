import pandas as pd
from pathlib import Path

def carregar_dados_para_treino(caminho_unificado: str) -> pd.DataFrame:
    """
    Carrega o parquet unificado; gera coluna 'y' (alvo):
      0) se 'y' já existe, retorna;
      1) se 'contratado' existe, transfere para 'y';
      2) se 'situacao_candidado' indicar contratação/seleção/aprovação, usa isso;
      3) senão, se houver prospect_id/applicant_id, faz o fallback daí.
    """
    p = Path(caminho_unificado)
    if not p.exists():
        raise FileNotFoundError(f"Parquet unificado não encontrado em {p}")

    df = pd.read_parquet(p)

    # 0) se 'y' já existe, retorna
    if "y" in df.columns:
        return df

    # 1) se 'contratado' existe, usar como y
    if "contratado" in df.columns:
        df["y"] = df["contratado"].astype(int)
        return df

    # 2) preferir gerar y a partir de situacao_candidado
    if "situacao_candidado" in df.columns:
        df["y"] = (
            df["situacao_candidado"]
              .fillna("")
              .str.lower()
              .str.contains(r"contratad|selecionad|aprovad")
              .astype(int)
        )
        return df

    # 3) fallback pelo funil (applicant_id vs prospect_id)
    if "applicant_id" in df.columns and "prospect_id" in df.columns:
        df["y"] = df["applicant_id"].isin(df["prospect_id"]).astype(int)
        return df

    raise KeyError(
        "Não encontrei nem 'y', nem 'contratado', nem 'situacao_candidado' nem 'prospect_id/applicant_id' para gerar y."
    )
