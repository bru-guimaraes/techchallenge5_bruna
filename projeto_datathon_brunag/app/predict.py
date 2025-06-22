import joblib
import os
import pandas as pd

MODELO_PATH = "model/modelo_rf.pkl"

def carregar_modelo():
    return joblib.load(MODELO_PATH)

modelo = carregar_modelo()

def fazer_previsao(dados_input: dict) -> int:
    df = pd.DataFrame([dados_input])
    pred = modelo.predict(df)
    return int(pred[0])
