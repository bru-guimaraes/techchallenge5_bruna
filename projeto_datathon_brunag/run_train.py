import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

from utils.data_merger import merge_dataframes
from utils.paths import (
    PATH_PARQUET_APPLICANTS,
    PATH_PARQUET_PROSPECTS,
    PATH_PARQUET_VAGAS,
    PATH_MODELO_SAIDA,
)


def main():
    print("Starting training script")

    # 1) Carregar parquet
    print("Loading parquet files...")
    applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
    prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
    vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
    print(f"Data shapes: applicants={applicants.shape}, prospects={prospects.shape}, vagas={vagas.shape}")

    # 2) Mesclar
    print("Merging dataframes...")
    df = merge_dataframes(applicants, prospects, vagas)
    print(f"Merged dataframe shape: {df.shape}")

    # 3) Gerar coluna-alvo
    print("Generating target column 'contratado' from 'situacao_candidado'...")
    df['contratado'] = df['situacao_candidado'].apply(
        lambda x: 1 if 'Contratado' in str(x) else 0
    )
    print("Value counts for target:")
    print(df['contratado'].value_counts())

    # 4) Selecionar features
    features = ['area_atuacao', 'nivel_ingles', 'nivel_espanhol', 'nivel_academico']
    df_filtered = df.dropna(subset=features + ['contratado'])
    if df_filtered.empty:
        print("No valid data for training. Exiting.")
        return

    # 5) Preparar X e y
    X = pd.get_dummies(df_filtered[features]).astype(float)
    y = df_filtered['contratado']
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
        os.makedirs(os.path.dirname(PATH_MODELO_SAIDA), exist_ok=True)
        joblib.dump(le, PATH_MODELO_SAIDA.replace('.pkl', '_label_encoder.pkl'))
        print("Label encoder saved.")

    # 6) Split
    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 7) SMOTE (balanceamento)
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 8) Treinar modelo
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 9) Avaliar
    print("Classification report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 10) Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print("Feature importances:")
    print(importances)

    # 11) Salvar modelo
    os.makedirs(os.path.dirname(PATH_MODELO_SAIDA), exist_ok=True)
    joblib.dump(model, PATH_MODELO_SAIDA)
    print(f"Model saved to {PATH_MODELO_SAIDA}")


if __name__ == "__main__":
    main()
