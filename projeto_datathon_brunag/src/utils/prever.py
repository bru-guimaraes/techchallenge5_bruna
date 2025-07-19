import joblib
import pandas as pd
from pathlib import Path

from utils.paths import get_model_path, load_feature_names

# lista de features (vai apontar para o features.json correto)
_FEATURES = load_feature_names()
_MLB = None

def _load_mlb():
    """
    Carrega (ou recarrega) o MultiLabelBinarizer
    a partir do mesmo diretório do modelo.
    Usa VAR de ambiente PATH_MODEL automaticamente.
    """
    global _MLB
    if _MLB is None:
        mlb_path = Path(get_model_path()).parent / "area_atuacao_mlb.joblib"
        _MLB = joblib.load(str(mlb_path))
    return _MLB

def _get_model():
    """
    Carrega (ou recarrega) o modelo do disco,
    usando VAR de ambiente PATH_MODEL.
    """
    return joblib.load(get_model_path())

def preprocessar_requisicao(dados: dict) -> pd.DataFrame:
    """
    Converte dict de entrada em DataFrame pronto para predição.
    """
    df = pd.DataFrame([dados])

    # 1) split + strip
    df["area_atuacao"] = (
        df["area_atuacao"]
        .str.split(",")
        .apply(lambda lst: [s.strip() for s in lst])
    )

    # 2) one‑hot multilabel
    mlb = _load_mlb()
    arr = mlb.transform(df["area_atuacao"])
    df_area = pd.DataFrame(arr, columns=mlb.classes_)

    # 3) concat + dummies dos outros campos
    df_rest = df.drop(columns=["area_atuacao"])
    df1 = pd.concat([df_rest.reset_index(drop=True), df_area], axis=1)
    df_cod = pd.get_dummies(df1)

    # 4) alinha com as features esperadas
    return df_cod.reindex(columns=_FEATURES, fill_value=0)
