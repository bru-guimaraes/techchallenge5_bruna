import pandas as pd
import joblib
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from utils.data_merger import merge_dataframes
from utils.paths import (
    PATH_PARQUET_APPLICANTS,
    PATH_PARQUET_PROSPECTS,
    PATH_PARQUET_VAGAS,
    PATH_MODEL,
)

def main():
    print("Starting training script")

    # 1) Carrega os arquivos Parquet
    print("Loading parquet files...")
    applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
    prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
    vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
    print(f"Data shapes: applicants={applicants.shape}, prospects={prospects.shape}, vagas={vagas.shape}")

    # 2) Faz o merge de todos os dados
    print("Merging dataframes...")
    df = merge_dataframes(applicants, prospects, vagas)
    print(f"Merged dataframe shape: {df.shape}")

    # 3) Renomeia as colunas aninhadas para nomes simples de feature
    print("Renaming feature columns to unified names...")
    df.rename(columns={
        'informacoes_profissionais.area_atuacao': 'area_atuacao',
        'formacao_e_idiomas.nivel_academico':     'nivel_academico',
        'formacao_e_idiomas.nivel_ingles':        'nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol':      'nivel_espanhol'
    }, inplace=True)
    print("Columns after rename:", [c for c in df.columns 
                                   if c in ['area_atuacao','nivel_academico','nivel_ingles','nivel_espanhol']])

    # 4) Gera coluna alvo binária a partir de 'situacao_candidado'
    print("Generating target column 'contratado'...")
    df['contratado'] = (df['situacao_candidado'] == 'Contratado pela Decision').astype(int)
    print(df['contratado'].value_counts())

    # 5) Define lista de features
    features = ['area_atuacao', 'nivel_ingles', 'nivel_espanhol', 'nivel_academico']

    # 6) Descarta linhas com NA nas features ou no alvo
    print("Filtering rows with missing values in features or target...")
    df = df.dropna(subset=features + ['contratado'])
    print("After dropna:", df.shape)

    # 7) Prepara X e y
    print("Preparing feature matrix and target vector...")
    X = pd.get_dummies(df[features]).astype(float)
    y = df['contratado']

    # ——— AQUI ADICIONAMOS O SALVAMENTO DAS FEATURES PARA A API ———
    feature_names = X.columns.tolist()
    features_path = os.path.join(os.path.dirname(PATH_MODEL), "features.json")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "w") as f:
        json.dump(feature_names, f)
    print(f"✅ Feature names saved to {features_path}")
    # ————————————————————————————————————————————————————————————

    # 8) Divide treino/teste
    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 9) Balanceia classes com SMOTE
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 10) Treina o classificador
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    # 11) Avalia no conjunto de teste
    print("Classification report:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 12) Exibe importância das features
    print("Feature importances:")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    # 13) Salva o modelo treinado
    os.makedirs(os.path.dirname(PATH_MODEL), exist_ok=True)
    joblib.dump(model, PATH_MODEL)
    print(f"✅ Model saved to {PATH_MODEL}")

if __name__ == "__main__":
    main()
