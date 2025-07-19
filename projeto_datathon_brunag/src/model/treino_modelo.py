import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def treinar_modelo(df: pd.DataFrame, caminho_saida: str):
    print("🧪 Iniciando treino do modelo...")

    colunas_uteis = [
        "area_atuacao",
        "nivel_ingles",
        "nivel_espanhol",
        "nivel_academico",
    ]

    if "contratado" not in df.columns:
        print("❌ Coluna 'contratado' não encontrada. Abortando treino.")
        return

    # Filtra apenas colunas úteis + alvo
    df_filtrado = df[colunas_uteis + ["contratado"]].dropna()

    if df_filtrado.empty:
        print("⚠️ Nenhum dado válido encontrado para treino.")
        return

    print(f"🔢 Total de amostras válidas: {len(df_filtrado)}")

    X = df_filtrado[colunas_uteis]
    y = df_filtrado["contratado"]

    # One-hot encoding
    X = pd.get_dummies(X)

    # Aplica SMOTE para balancear classes
    print("⚖️ Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"✅ Dados balanceados: {X_resampled.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    # Treinamento
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Avaliação
    y_pred = modelo.predict(X_test)
    print("\n📊 Relatório de Classificação:\n")
    print(classification_report(y_test, y_pred))

    # Salvando modelo
    joblib.dump(modelo, caminho_saida)
    print(f"✅ Modelo salvo em: {caminho_saida}")
