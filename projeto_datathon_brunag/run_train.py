# run_train.py

import os
import json
import yaml
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

from utils.paths import PATH_MODEL as CONST_PATH_MODEL
from utils.feature_engineering import processar_features_inference

def main():
    print("🚀 Starting training script")

    # 1) Paths (can override via env)
    model_path   = os.getenv("PATH_MODEL", CONST_PATH_MODEL)
    parquet_path = os.getenv("PATH_TREINO_PARQUET", "data/parquet/treino_unificado.parquet")

    # 2) Load RF hyperparameters
    print("🔧 Loading hyperparameter config...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rf_params = cfg.get("random_forest", {})
    if rf_params.get("max_features") == "auto":
        rf_params["max_features"] = "sqrt"
    print(f"📋 RandomForest params: {rf_params}")

    # 3) Load unified parquet
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet de treino não encontrado em {parquet_path}. Abort.")
        return
    df = pd.read_parquet(parquet_path)
    print(f"🔢 Dataset shape: {df.shape}")

    # 4) Define base features + drop nulls
    base_feats = [
        'informacoes_profissionais.area_atuacao',
        'formacao_e_idiomas.nivel_academico',
        'formacao_e_idiomas.nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol',
        'nivel_academico',
        'nivel_ingles',
        'nivel_espanhol',
        'areas_atuacao'
    ]
    df = df.dropna(subset=base_feats + ['contratado'])

    # 5) One‐hot encode & save feature list
    print("📦 One-hot encoding and saving feature list...")
    X = pd.get_dummies(df[base_feats]).astype(float)
    y = df['contratado']
    feature_names = X.columns.tolist()
    features_path = os.path.join(os.path.dirname(model_path), "features.json")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)
    print(f"✅ Features saved to {features_path}")

    # 6) Train/test split
    print("🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 7) SMOTE
    print("⚖️ Applying SMOTE...")
    try:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied: {Counter(y_train_res)}")
    except ValueError as e:
        print(f"⚠️ SMOTE skipped: {e}")
        X_train_res, y_train_res = X_train, y_train

    # 8) Train RF
    print("🛠️ Training RandomForestClassifier...")
    model = RandomForestClassifier(**rf_params, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 9) Evaluate
    print("\n📊 Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 10) Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    main()
