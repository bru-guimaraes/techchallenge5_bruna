# src/utils/engenharia_de_features.py

"""
utils/engenharia_de_features.py

Transformações de features e alinhamento com encoder salvo (se houver).
"""

import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PATH_TREINO = BASE_DIR / "data" / "parquet" / "treino_unificado.parquet"
PATH_ENCODER = BASE_DIR / "model" / "area_atuacao_mlb.joblib"


def processar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza dummy-encoding das colunas do DataFrame e, se existir um encoder
    salvo em PATH_ENCODER, alinha as colunas de saída a partir do encoder.

    Args:
        df: DataFrame cru com colunas originais.

    Retorna:
        DataFrame codificado e potencialmente alinhado.
    """
    # 1) Cópia dos dados
    df_sel = df.copy()

    # 2) Dummy-encode de todas as colunas categóricas
    df_enc = pd.get_dummies(df_sel)

    # 3) Tenta carregar encoder pré-treinado para alinhamento
    try:
        encoder_obj = joblib.load(PATH_ENCODER)
    except Exception:
        # Se não encontrar ou falhar no load, retorna dummies sem alinhamento
        return df_enc

    # 4) Se for sklearn transformer com atributo feature_names_in_, usa-o
    if hasattr(encoder_obj, "feature_names_in_"):
        cols = encoder_obj.feature_names_in_
        return df_enc.reindex(columns=cols, fill_value=0)

    # 5) Se for um dict (por exemplo, dict de classes), usa as chaves como colunas
    if isinstance(encoder_obj, dict):
        cols = list(encoder_obj.keys())
        return df_enc.reindex(columns=cols, fill_value=0)

    # 6) Caso genérico, retorna dummies sem alteração
    return df_enc


if __name__ == "__main__":
    # Quando executado diretamente, gera parquet de features para inspeção
    df = pd.read_parquet(PATH_TREINO)
    df_proc = processar_features(df)
    df_proc.to_parquet(
        BASE_DIR / "data" / "parquet" / "treino_features.parquet",
        index=False
    )
    print("✅ Features processadas e salvas!")
