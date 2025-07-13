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

from utils.paths import PATH_MODEL as CONST_PATH_MODEL

def main():
    print("🚀 Starting training script")

    # Permite sobrescrever o caminho do modelo por variável de ambiente
    model_path = os.getenv("PATH_MODEL", CONST_PATH_MODEL)

    # Permite sobrescrever o parquet de treino via variável de ambiente
    parquet_path = os.getenv("PATH_TREINO_PARQUET", "data/parquet/treino_unificado.parquet")

    # 0) Carrega configuração de hiperparâmetros do RandomForest
    print("🔧 Loading hyperparameter config...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rf_params = cfg.get("random_forest", {})
    if rf_params.get("max_features") == "auto":
        rf_params["max_features"] = "sqrt"
    print(f"📋 RandomForest params: {rf_params}")

    # 1) Carrega dataset de treino já unificado
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet de treino não encontrado em {parquet_path}. Execute o pré-processamento antes.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"🔢 Shape do dataset unificado: {df.shape}")

    # Colunas base para features (ajuste conforme evolução do projeto)
    base_features = [
        'informacoes_profissionais.area_atuacao',
        'formacao_e_idiomas.nivel_academico',
        'formacao_e_idiomas.nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol',
        'nivel_academico',
        'nivel_ingles',
        'nivel_espanhol',
        'areas_atuacao'
    ]

    # 2) Remove linhas com valores nulos em qualquer feature essencial ou target
    df = df.dropna(subset=base_features + ['contratado'])

    # 3) One-hot encoding e salva lista de features
    print("📦 One-hot encoding and saving feature list...")
    X = pd.get_dummies(df[base_features]).astype(float)
    y = df['contratado']
    feature_names = X.columns.tolist()
    features_path = os.path.join(os.path.dirname(model_path), "features.json")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False)
    print(f"✅ Features saved to {features_path}")

    # 4) Train/test split
    print("🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5) SMOTE para balancear as classes
    print("⚖️ Applying SMOTE...")
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied: {Counter(y_train_res)}")
    except ValueError as e:
        print(f"⚠️ SMOTE skipped (single class): {e}")
        X_train_res, y_train_res = X_train, y_train

    # 6) Treina o RandomForest
    print("🛠️ Training RandomForestClassifier...")
    model = RandomForestClassifier(**rf_params, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 7) Avaliação no conjunto de teste
    print("\n📊 Classification Report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 8) Importância das features
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("🌟 Feature importances:\n", importances.sort_values(ascending=False).head(10))

    # 9) Salva o modelo treinado
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == "__main__":
    main()
