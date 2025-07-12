import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
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
    # 1) carga
    applicants = pd.read_parquet(PATH_PARQUET_APPLICANTS)
    prospects  = pd.read_parquet(PATH_PARQUET_PROSPECTS)
    vagas      = pd.read_parquet(PATH_PARQUET_VAGAS)
    df = merge_dataframes(applicants, prospects, vagas)
    df.rename(columns={
        'informacoes_profissionais.area_atuacao':'area_atuacao',
        'formacao_e_idiomas.nivel_academico':'nivel_academico',
        'formacao_e_idiomas.nivel_ingles':'nivel_ingles',
        'formacao_e_idiomas.nivel_espanhol':'nivel_espanhol'
    }, inplace=True)
    df['contratado'] = (df['situacao_candidado']=='Contratado pela Decision').astype(int)
    features = ['area_atuacao','nivel_ingles','nivel_espanhol','nivel_academico']
    df = df.dropna(subset=features+['contratado'])
    X = pd.get_dummies(df[features]).astype(float)
    y = df['contratado']

    # 2) split e SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # 3) grid search
    param_grid = {
        'n_estimators':[50,100,200],
        'max_depth':[None,10,20],
        'min_samples_split':[2,5]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
    )
    gs.fit(X_res, y_res)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # 4) avaliação
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 5) salvar modelo + features
    MODEL_DIR = Path(PATH_MODEL).parent
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best, PATH_MODEL)
    feature_names = X.columns.tolist()
    with open(MODEL_DIR/"features.json","w") as f:
        json.dump(feature_names, f)
    print(f"Modelo e features salvos em {MODEL_DIR}")

if __name__=="__main__":
    main()
