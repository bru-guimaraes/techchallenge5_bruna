# src/run_train.py

"""
Pipeline de treino:
1) Carrega config.yaml
2) Carrega parquet
3) Se não há dados ou coluna target ausente, gera modelo dummy
4) Processa features e target
5) Split + SMOTE
6) Treina RandomForest
7) Avaliação
8) Salva modelo e features.json (tanto em model/… quanto em ./features.json)
"""
import os
import yaml
import json
import joblib
import shutil
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

from utils.paths import get_model_path
from utils.engenharia_de_features import processar_features

def main():
    # 1) Carrega config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) Carrega dados
    df = pd.read_parquet(cfg["paths"]["parquet_treino_unificado"])

    # 3) Define paths de saída
    model_path = get_model_path()                  # ex: /tmp/.../mymodel.joblib ou /app/model/area_atuacao_mlb.joblib
    model_dir  = Path(model_path).parent
    feat_model = model_dir / "features.json"       # para testes
    feat_root  = Path("features.json")             # para Dockerfile COPY

    os.makedirs(model_dir, exist_ok=True)

    # 4) Se não há dados ou coluna-target ausente → dummy + features vazias
    target_col = cfg["features"]["target_column"]
    if df.empty or target_col not in df.columns:
        print(f"⚠️ Dataset vazio ou sem coluna '{target_col}'. Gerando modelo dummy.")
        dummy = RandomForestClassifier()
        joblib.dump(dummy, model_path)

        # features.json vazio no model_dir
        with open(feat_model, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
        # copia para root para o Dockerfile
        shutil.copy(str(feat_model), str(feat_root))

        print(f"💾 Dummy model salvo em {model_path}")
        print(f"💾 Features (vazio) salvo em {feat_model} e em {feat_root}")
        return

    # 5) Processa features e target
    X = processar_features(df)
    y = df[target_col]

    # 6) Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        stratify=y
    )

    # 7) Balanceamento com SMOTE
    sm = SMOTE(random_state=cfg["train"]["random_state"])
    X_tr, y_tr = sm.fit_resample(X_train, y_train)
    print("🛠️ Dados balanceados:", Counter(y_tr))

    # 8) Treina RandomForest
    model = RandomForestClassifier(
        **cfg["model"]["random_forest"],
        random_state=cfg["train"]["random_state"]
    )
    model.fit(X_tr, y_tr)

    # 9) Avaliação
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 10) Salva modelo
    joblib.dump(model, model_path)
    print(f"💾 Model salvo em {model_path}")

    # 11) Salva lista de features no model_dir
    with open(feat_model, "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, ensure_ascii=False)
    # 12) Copia para root para o Dockerfile
    shutil.copy(str(feat_model), str(feat_root))

    print(f"💾 Features salvo em {feat_model} e em {feat_root}")

if __name__ == "__main__":
    main()
