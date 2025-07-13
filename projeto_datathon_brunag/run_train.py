import pandas as pd
import joblib
import os
import json
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

from utils.data_merger import merge_dataframes
from utils.paths import (
    PATH_PARQUET_APPLICANTS,
    PATH_PARQUET_PROSPECTS,
    PATH_PARQUET_VAGAS,
    PATH_MODEL as CONST_PATH_MODEL,
)

def main():
    print("🚀 Starting training script")

    # Respeita variável de ambiente PATH_MODEL, senão usa o valor padrão
    model_path = os.getenv("PATH_MODEL", CONST_PATH_MODEL)

    # 0) Carrega configuração de hiper-parâmetros
    print("🔧 Loading hyperparameter config...")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    rf_params = cfg.get("random_forest", {})
    # Ajusta max_features 'auto' → 'sqrt'
    if rf_params.get("max_features") == "auto":
        rf_params["max_features"] = "sqrt"
    print(f"📋 RandomForest params: {rf_params}")

    # 1) Carrega dados
    print("📥 Loading parquet files...")
    applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
    prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
    vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
    print(f"🔢 Shapes — applicants: {applicants.shape}, prospects: {prospects.shape}, vagas: {vagas.shape}")

    # 2) Merge
    print("🔗 Merging dataframes...")
    df = merge_dataframes(applicants, prospects, vagas)
    print(f"➡️  Merged shape: {df.shape}")

    # 3) Renomeia colunas aninhadas
    print("✏️ Renaming columns...")
    df.rename(columns={
        'informacoes_profissionais.area_atuacao': 'area_atuacao',
        'formacao_e_idiomas.nivel_academico':     'nivel_academico',
        'formacao_e_idiomas.nivel_ingles':        'nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol':      'nivel_espanhol'
    }, inplace=True)

    # 4) Gera target
    print("🎯 Generating target 'contratado'...")
    df['contratado'] = (df['situacao_candidado'] == 'Contratado pela Decision').astype(int)

    # 5) Define features e filtra NAs
    features = ['area_atuacao', 'nivel_ingles', 'nivel_espanhol', 'nivel_academico']
    df = df.dropna(subset=features + ['contratado'])

    # 6) One-hot + salva lista de features
    print("📦 One-hot encoding and saving feature list...")
    X = pd.get_dummies(df[features]).astype(float)
    y = df['contratado']
    feature_names = X.columns.tolist()
    features_path = os.path.join(os.path.dirname(model_path), "features.json")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)
    print(f"✅ Features saved to {features_path}")

    # 7) Train/test split
    print("🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 8) SMOTE com fallback
    print("⚖️ Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied: {Counter(y_train_res)}")
    except ValueError as e:
        print(f"⚠️ SMOTE skipped (single class): {e}")
        X_train_res, y_train_res = X_train, y_train

    # 9) Treino com RF
    print("🛠️ Training RandomForestClassifier...")
    model = RandomForestClassifier(**rf_params, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 10) Avaliação
    print("\n📊 Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 11) Importância de features
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("🌟 Feature importances:\n", importances.sort_values(ascending=False).head(10))

    # 12) Salva o modelo treinado
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    main()
